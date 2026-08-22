from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import ReconstructionResult
from .vggt_backend import VggtBackend
from models import import_gsplat_checked


class CudaBackend(VggtBackend):
    name = "cuda"

    def __init__(self, config: dict[str, Any], root: Path):
        super().__init__(config, root, "cuda")

    def train_splat(
        self,
        poses: list[dict[str, Any]],
        points: ReconstructionResult,
        images: Sequence[Path],
    ) -> bytes | None:
        import_gsplat_checked()
        from splat_trainer import train_splat_bytes

        return train_splat_bytes(self.config, poses, points, images)
