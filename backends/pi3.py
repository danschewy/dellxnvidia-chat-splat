"""CUDA-only π³ geometry fallback using only local weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import ReconstructionResult
from models import load_pi3, load_pi3x


def _estimate_pinhole_intrinsics(
    local_points: Any,
    confidences: Any,
    fallback_focal: float,
) -> list[list[list[float]]]:
    """Fit per-view pinhole intrinsics to Pi3's predicted camera-space rays."""
    import numpy as np

    point_values = np.asarray(local_points, dtype=np.float64)
    confidence_values = np.asarray(confidences, dtype=np.float64)
    if confidence_values.ndim == point_values.ndim:
        confidence_values = confidence_values[..., 0]
    view_count, height, width = point_values.shape[:3]
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    outputs = []
    for view in range(view_count):
        points = point_values[view]
        confidence = confidence_values[view]
        z = points[..., 2]
        valid = (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(confidence)
            & (z > 1e-6)
        )
        if int(valid.sum()) >= 32:
            threshold = float(np.quantile(confidence[valid], 0.5))
            valid &= confidence >= threshold
        indices = np.flatnonzero(valid)
        if len(indices) > 50_000:
            indices = indices[
                np.linspace(0, len(indices) - 1, 50_000, dtype=np.int64)
            ]

        fx = fy = float(fallback_focal)
        cx, cy = width / 2.0, height / 2.0
        if len(indices) >= 32:
            flat_points = points.reshape(-1, 3)[indices]
            weights = np.sqrt(np.maximum(confidence.reshape(-1)[indices], 1e-6))

            def fit(pixel: Any, numerator: Any) -> tuple[float, float]:
                ratio = numerator / flat_points[:, 2]
                design = np.column_stack((ratio, np.ones_like(ratio)))
                fitted, *_ = np.linalg.lstsq(
                    design * weights[:, None], pixel.reshape(-1)[indices] * weights, rcond=None
                )
                return float(fitted[0]), float(fitted[1])

            estimated_fx, estimated_cx = fit(pixel_x, flat_points[:, 0])
            estimated_fy, estimated_cy = fit(pixel_y, flat_points[:, 1])
            plausible = (
                0.2 * width <= estimated_fx <= 4.0 * width
                and 0.2 * height <= estimated_fy <= 4.0 * width
                and -0.25 * width <= estimated_cx <= 1.25 * width
                and -0.25 * height <= estimated_cy <= 1.25 * height
            )
            if plausible:
                fx, fy, cx, cy = (
                    estimated_fx,
                    estimated_fy,
                    estimated_cx,
                    estimated_cy,
                )
        outputs.append(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        )
    return outputs


def reconstruct_pi3(
    config: dict[str, Any], weights_dir: Path, images: Sequence[Path]
) -> ReconstructionResult:
    import numpy as np
    from PIL import Image
    import torch

    resolution = int(config["vggt_resolution"])
    loaded = []
    for path in images:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
            loaded.append(image.copy())
    max_width = max(image.width for image in loaded)
    max_height = max(image.height for image in loaded)
    arrays = []
    for image in loaded:
        canvas = Image.new("RGB", (max_width, max_height), (255, 255, 255))
        canvas.paste(image, ((max_width - image.width) // 2, (max_height - image.height) // 2))
        arrays.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to("cuda")
    geometry_model = str(config.get("geometry_model", "pi3")).lower()
    if geometry_model == "pi3x":
        model = load_pi3x(weights_dir, "cuda")
    elif geometry_model == "pi3":
        model = load_pi3(weights_dir, "cuda")
    else:
        raise ValueError("Pi3 reconstruction requires geometry_model pi3 or pi3x")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(tensor[None])
    points = predictions["points"].detach().float().cpu().numpy().squeeze(0)
    local_points = predictions["local_points"].detach().float().cpu().numpy().squeeze(0)
    point_layout = tuple(int(value) for value in points.shape[:3])
    confidences = torch.sigmoid(predictions["conf"]).detach().float().cpu().numpy().squeeze(0)
    poses = predictions["camera_poses"].detach().float().cpu().numpy().squeeze(0)
    colors = (np.stack(arrays) * 255.0).clip(0, 255).astype(np.uint8)
    intrinsics = _estimate_pinhole_intrinsics(
        local_points, confidences, float(config["pi3_focal_length"])
    )
    cameras = [
        {
            "frame": path.name,
            "T_wc": poses[index].tolist(),
            "K": intrinsics[index],
            "confidence": float(np.median(confidences[index])),
        }
        for index, path in enumerate(images)
    ]
    return ReconstructionResult(
        points=points.reshape(-1, 3),
        colors=colors.reshape(-1, 3),
        confidences=confidences.reshape(-1),
        cameras=cameras,
        point_layout=point_layout,
    )
