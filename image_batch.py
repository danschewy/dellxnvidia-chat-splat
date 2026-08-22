"""Load already-resized reconstruction images without applying a second crop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def load_prepared_image_batch(images: Sequence[Path]) -> Any:
    """Stack prepared RGB images, centering white padding only when shapes differ."""
    if not images:
        raise ValueError("At least one prepared image is required")
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    tensors = []
    for path in images:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            values = np.array(rgb, dtype=np.float32, copy=True) / 255.0
        tensors.append(torch.from_numpy(values).permute(2, 0, 1))

    max_height = max(int(tensor.shape[1]) for tensor in tensors)
    max_width = max(int(tensor.shape[2]) for tensor in tensors)
    padded = []
    for tensor in tensors:
        height_padding = max_height - int(tensor.shape[1])
        width_padding = max_width - int(tensor.shape[2])
        if height_padding or width_padding:
            top = height_padding // 2
            left = width_padding // 2
            tensor = torch.nn.functional.pad(
                tensor,
                (left, width_padding - left, top, height_padding - top),
                mode="constant",
                value=1.0,
            )
        padded.append(tensor)
    return torch.stack(padded)
