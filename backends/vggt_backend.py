"""Shared VGGT and YOLO implementation for CUDA and MPS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import ReconstructionResult
from models import load_vggt, load_yolo
from roomscan_io import resolve_weights_dir


def _invert_batched_se3(extrinsic: Any, inverse_fn: Any) -> Any:
    """Adapt VGGT's BxS pose batch to its Nx3x4 inverse helper."""
    if len(extrinsic.shape) < 3 or tuple(extrinsic.shape[-2:]) not in {
        (3, 4),
        (4, 4),
    }:
        raise ValueError(f"Expected batched 3x4 or 4x4 extrinsics, got {tuple(extrinsic.shape)}")
    leading_shape = tuple(extrinsic.shape[:-2])
    flattened = extrinsic.reshape((-1,) + tuple(extrinsic.shape[-2:]))
    inverted = inverse_fn(flattened)
    return inverted.reshape(leading_shape + (4, 4))


class VggtBackend:
    name = "vggt"

    def __init__(self, config: dict[str, Any], root: Path, device: str):
        self.config = config
        self.root = root
        self.device = device
        self.weights_dir = resolve_weights_dir(config, root)
        self._vggt: Any = None
        self._yolo: Any = None

    @property
    def vggt(self) -> Any:
        if self._vggt is None:
            self._vggt = load_vggt(self.weights_dir, self.device)
        return self._vggt

    @property
    def yolo(self) -> Any:
        if self._yolo is None:
            self._yolo = load_yolo(self.weights_dir)
        return self._yolo

    def reconstruct(self, images: Sequence[Path]) -> ReconstructionResult:
        import numpy as np
        import torch
        from vggt.utils.geometry import closed_form_inverse_se3
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        tensors = load_and_preprocess_images([str(path) for path in images]).to(self.device)
        if tensors.shape[-1] > int(self.config["vggt_resolution"]) or tensors.shape[-2] > int(self.config["vggt_resolution"]):
            tensors = torch.nn.functional.interpolate(
                tensors,
                size=(int(self.config["vggt_resolution"]), int(self.config["vggt_resolution"])),
                mode="bilinear",
                align_corners=False,
            )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.autocast(device_type="mps", dtype=torch.float16)
        )
        with torch.inference_mode(), autocast:
            predictions = self.vggt(tensors)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], tensors.shape[-2:]
            )

        def array(name: str) -> Any:
            value = predictions[name].detach().float().cpu().numpy()
            return value.squeeze(0)

        points = array("world_points")
        point_layout = tuple(int(value) for value in points.shape[:3])
        confidences = array("world_points_conf")
        colors = array("images")
        if colors.ndim == 4 and colors.shape[1] == 3:
            colors = colors.transpose(0, 2, 3, 1)
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        intrinsic_np = intrinsic.detach().float().cpu().numpy().squeeze(0)
        camera_to_world = (
            _invert_batched_se3(extrinsic, closed_form_inverse_se3)
            .detach()
            .float()
            .cpu()
            .numpy()
            .squeeze(0)
        )
        cameras = []
        for index, image in enumerate(images):
            cameras.append(
                {
                    "frame": image.name,
                    "T_wc": camera_to_world[index].tolist(),
                    "K": intrinsic_np[index].tolist(),
                    "confidence": float(np.median(confidences[index])),
                }
            )
        return ReconstructionResult(
            points=points.reshape(-1, 3),
            colors=colors.reshape(-1, 3),
            confidences=confidences.reshape(-1),
            cameras=cameras,
            point_layout=point_layout,
        )

    def segment_people(self, images: Sequence[Path]) -> list[Any]:
        import cv2
        import numpy as np

        results = self.yolo.predict(
            source=[str(path) for path in images],
            classes=[0],
            device=self.device,
            verbose=False,
            stream=False,
        )
        outputs: list[Any] = []
        kernel = np.ones((21, 21), np.uint8)
        for image_path, result in zip(images, results):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image for masking: {image_path}")
            union = np.zeros(image.shape[:2], dtype=np.uint8)
            masks = getattr(result, "masks", None)
            if masks is not None:
                for raw in masks.data.detach().cpu().numpy():
                    mask = cv2.resize(raw, (image.shape[1], image.shape[0]))
                    union |= (mask > 0.5).astype(np.uint8)
            union = cv2.dilate(union, kernel, iterations=1)
            outputs.append(union.astype(bool))
        return outputs
