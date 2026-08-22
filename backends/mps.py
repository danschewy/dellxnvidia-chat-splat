from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import ReconstructionResult
from .vggt_backend import VggtBackend


class MpsBackend(VggtBackend):
    name = "mps"

    def __init__(self, config: dict[str, Any], root: Path):
        super().__init__(config, root, "mps")

    def reconstruct(self, images: Sequence[Path]) -> ReconstructionResult:
        if str(self.config.get("geometry_model", "vggt")).lower() != "vggt":
            raise RuntimeError("Pi3 fallback is supported only by CudaBackend")
        return super().reconstruct(images)

    def train_splat(
        self,
        poses: list[dict[str, Any]],
        points: ReconstructionResult,
        images: Sequence[Path],
    ) -> bytes | None:
        return None
