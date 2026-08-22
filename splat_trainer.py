"""Small gsplat optimizer initialized directly from VGGT geometry and poses."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Sequence

from backends import ReconstructionResult


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def train_splat_bytes(
    config: dict[str, Any],
    poses: list[dict[str, Any]],
    reconstruction: ReconstructionResult,
    images: Sequence[Path],
) -> bytes:
    import numpy as np
    import torch
    from gsplat import rasterization
    from vggt.utils.load_fn import load_and_preprocess_images

    if not poses or not images:
        raise ValueError("Splat training requires at least one posed image")
    device = "cuda"
    point_values = np.asarray(reconstruction.points, dtype=np.float32)
    color_values = np.asarray(reconstruction.colors, dtype=np.float32) / 255.0
    limit = int(config["max_splat_points"])
    if len(point_values) > limit:
        indices = np.linspace(0, len(point_values) - 1, limit, dtype=np.int64)
        point_values, color_values = point_values[indices], color_values[indices]

    means = torch.nn.Parameter(torch.from_numpy(point_values).to(device))
    color_logits = torch.nn.Parameter(
        torch.logit(torch.from_numpy(color_values).to(device).clamp(0.001, 0.999))
    )
    centered = means - means.detach().median(dim=0).values
    scene_extent = torch.quantile(torch.linalg.vector_norm(centered, dim=1), 0.8).detach()
    initial_scale = torch.clamp(scene_extent / max(math.sqrt(len(means)), 1.0), min=1e-4)
    log_scales = torch.nn.Parameter(torch.ones_like(means) * torch.log(initial_scale))
    quats = torch.nn.Parameter(torch.zeros((len(means), 4), device=device))
    with torch.no_grad():
        quats[:, 0] = 1.0
    opacity_logits = torch.nn.Parameter(torch.full((len(means),), _logit(0.7), device=device))

    rate = float(config["splat_learning_rate"])
    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": rate * 0.08},
            {"params": [color_logits], "lr": rate},
            {"params": [log_scales], "lr": rate * 0.5},
            {"params": [quats], "lr": rate * 0.2},
            {"params": [opacity_logits], "lr": rate * 0.5},
        ]
    )
    # Reuse VGGT's exact crop/resize transform so K and target pixels remain
    # aligned. Padding phone images to a square would require principal-point
    # offsets and roughly doubles attention/render work for 16:9 captures.
    target_batch = load_and_preprocess_images([str(path) for path in images]).to(device)
    source_height, source_width = target_batch.shape[-2:]
    requested_size = int(config["splat_train_resolution"])
    resize_scale = min(1.0, requested_size / max(source_height, source_width))
    height = max(1, round(source_height * resize_scale))
    width = max(1, round(source_width * resize_scale))
    if (height, width) != (source_height, source_width):
        target_batch = torch.nn.functional.interpolate(
            target_batch, size=(height, width), mode="bilinear", align_corners=False
        )
    targets = target_batch.permute(0, 2, 3, 1).contiguous()
    viewmats = torch.tensor(
        [np.linalg.inv(np.asarray(camera["T_wc"], dtype=np.float32)) for camera in poses],
        device=device,
    )
    intrinsics = torch.tensor([camera["K"] for camera in poses], dtype=torch.float32, device=device)
    intrinsics[:, 0, :] *= width / source_width
    intrinsics[:, 1, :] *= height / source_height
    iterations = int(config["gsplat_iterations"])

    for step in range(iterations):
        camera_index = step % min(len(poses), len(targets))
        rendered, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(color_logits),
            viewmats=viewmats[camera_index : camera_index + 1],
            Ks=intrinsics[camera_index : camera_index + 1],
            width=width,
            height=height,
            packed=True,
            sh_degree=None,
            backgrounds=torch.zeros((1, 3), device=device),
        )
        visible = alpha[0].detach().clamp_min(0.05)
        loss = ((rendered[0] - targets[camera_index]).abs() * visible).mean()
        loss = loss + 0.001 * torch.exp(log_scales).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == iterations:
            print(f"[gsplat] iteration {step + 1}/{iterations}, loss={loss.item():.5f}", flush=True)

    means_np = means.detach().cpu().numpy()
    colors_np = torch.sigmoid(color_logits).detach().cpu().numpy()
    scales_np = log_scales.detach().cpu().numpy()
    quats_np = torch.nn.functional.normalize(quats.detach(), dim=-1).cpu().numpy()
    opacity_np = opacity_logits.detach().cpu().numpy()
    sh0_np = (colors_np - 0.5) / 0.28209479177387814
    out = io.StringIO()
    properties = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    out.write("ply\nformat ascii 1.0\ncomment ROOMSCAN standard 3DGS\n")
    out.write(f"element vertex {len(means_np)}\n")
    for property_name in properties:
        out.write(f"property float {property_name}\n")
    out.write("end_header\n")
    for index in range(len(means_np)):
        row = [
            *means_np[index], 0.0, 0.0, 0.0, *sh0_np[index], opacity_np[index],
            *scales_np[index], *quats_np[index],
        ]
        out.write(" ".join(f"{float(value):.7g}" for value in row) + "\n")
    return out.getvalue().encode("ascii")
