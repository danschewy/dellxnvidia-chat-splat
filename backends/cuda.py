from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import ReconstructionResult
from .pi3 import reconstruct_pi3
from .vggt_backend import VggtBackend
from models import import_gsplat_checked, pi3_path


class CudaBackend(VggtBackend):
    name = "cuda"

    def __init__(self, config: dict[str, Any], root: Path):
        super().__init__(config, root, "cuda")

    def reconstruct(self, images: Sequence[Path]) -> ReconstructionResult:
        geometry_model = str(self.config.get("geometry_model", "vggt")).lower()
        if geometry_model == "pi3":
            return reconstruct_pi3(self.config, self.weights_dir, images)
        if geometry_model != "vggt":
            raise ValueError("geometry_model must be vggt or pi3")
        try:
            return super().reconstruct(images)
        except RuntimeError as vggt_error:
            if "out of memory" in str(vggt_error).lower():
                raise
            if not bool(self.config.get("pi3_fallback_on_error", True)):
                raise
            try:
                pi3_path(self.weights_dir)
            except Exception as pi3_error:
                raise RuntimeError(
                    f"VGGT failed ({vggt_error}); Pi3 fallback is unavailable ({pi3_error})"
                ) from vggt_error
            print(f"[reconstruct] VGGT failed; activating local Pi3 fallback: {vggt_error}", flush=True)
            return reconstruct_pi3(self.config, self.weights_dir, images)

    def train_splat(
        self,
        poses: list[dict[str, Any]],
        points: ReconstructionResult,
        images: Sequence[Path],
    ) -> bytes | None:
        import_gsplat_checked()
        from splat_trainer import train_splat_bytes

        return train_splat_bytes(self.config, poses, points, images)
