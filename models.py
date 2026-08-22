"""The only model-loading module in ROOMSCAN.

Every path is resolved under ``weights_dir`` and every loader is forced offline.
No inference code calls a model hub or accepts a remote identifier.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
LOCAL_FILES_ONLY = True


class MissingWeightError(FileNotFoundError):
    pass


def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise MissingWeightError(f"Missing {label}. Expected local file or directory: {path}")
    return path


def vggt_path(weights_dir: Path) -> Path:
    candidates = [weights_dir / "meta" / "VGGT", weights_dir / "facebook" / "VGGT-1B"]
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / name).is_file() for name in ("model.pt", "model.safetensors")):
            return candidate
    return _require(candidates[0], "VGGT weights (facebook/VGGT-1B)")


def yolo_path(weights_dir: Path) -> Path:
    candidates = [
        weights_dir / "yolo" / "yolo11s-seg.pt",
        weights_dir / "ultralytics" / "yolo11s-seg.pt",
        weights_dir / "yolo11s-seg.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return _require(candidates[0], "YOLO people-segmentation weights (yolo11s-seg.pt)")


def load_vggt(weights_dir: Path, device: str) -> Any:
    local_dir = vggt_path(weights_dir)
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:
        raise RuntimeError("VGGT Python package is not installed; install requirements for this target") from exc

    # The local path plus local_files_only is deliberate defense in depth.
    try:
        model = VGGT.from_pretrained(str(local_dir), local_files_only=LOCAL_FILES_ONLY)
    except TypeError:
        # Older huggingface_hub mixins do not expose local_files_only. A local
        # directory remains incapable of resolving to huggingface.co.
        model = VGGT.from_pretrained(str(local_dir))
    return model.eval().to(device)


def load_yolo(weights_dir: Path) -> Any:
    weight_file = yolo_path(weights_dir)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed; cannot load yolo11s-seg") from exc
    return YOLO(str(weight_file))


def import_gsplat_checked() -> Any:
    try:
        import gsplat
        from gsplat import rasterization
        # Importing the public function is lazy and does not prove that the
        # CUDA extension exists. _C forces the prebuilt module load or JIT
        # build, which is exactly what M0 must fail fast on.
        from gsplat.cuda._backend import _C
    except Exception as exc:
        raise RuntimeError(f"GSPLAT CUDA EXTENSION UNAVAILABLE: {exc}") from exc
    if not callable(rasterization) or _C is None:
        raise RuntimeError("GSPLAT CUDA EXTENSION UNAVAILABLE: compiled module is missing")
    return gsplat
