"""CUDA-only π³ geometry fallback using only local weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import ReconstructionResult
from models import load_pi3


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
    model = load_pi3(weights_dir, "cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(tensor[None])
    points = predictions["points"].detach().float().cpu().numpy().squeeze(0)
    point_layout = tuple(int(value) for value in points.shape[:3])
    confidences = torch.sigmoid(predictions["conf"]).detach().float().cpu().numpy().squeeze(0)
    poses = predictions["camera_poses"].detach().float().cpu().numpy().squeeze(0)
    colors = (np.stack(arrays) * 255.0).clip(0, 255).astype(np.uint8)
    focal = float(config["pi3_focal_length"])
    intrinsic = [[focal, 0.0, max_width / 2], [0.0, focal, max_height / 2], [0.0, 0.0, 1.0]]
    cameras = [
        {
            "frame": path.name,
            "T_wc": poses[index].tolist(),
            "K": intrinsic,
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
