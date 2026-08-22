"""Capability-selected reconstruction backends.

Heavy dependencies are imported only after a backend has been selected. This is
what allows the stub path to run on a machine with no ML stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class ReconstructionResult:
    """Backend-neutral reconstruction payload.

    ``points`` is an iterable of XYZ triples, ``colors`` RGB byte triples,
    ``confidences`` scalar values, and ``cameras`` follows the cameras.json
    contract. Real backends may return NumPy arrays; the stub uses Python lists.
    """

    points: Any
    colors: Any
    confidences: Any
    cameras: list[dict[str, Any]]


class Backend(Protocol):
    name: str

    def reconstruct(self, images: Sequence[Path]) -> ReconstructionResult: ...

    def train_splat(
        self,
        poses: list[dict[str, Any]],
        points: ReconstructionResult,
        images: Sequence[Path],
    ) -> bytes | None: ...

    def segment_people(self, images: Sequence[Path]) -> list[Any]: ...


def _torch_capabilities() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False
    cuda = bool(torch.cuda.is_available())
    mps = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )
    return cuda, mps


def select_backend(config: dict[str, Any], root: Path | None = None) -> Backend:
    """Select CUDA, MPS, then stub, with an explicit stub escape hatch."""

    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    requested = os.getenv("ROOMSCAN_BACKEND", str(config.get("backend_override", "auto")))
    requested = requested.strip().lower()
    cuda, mps = _torch_capabilities()

    if requested not in {"auto", "stub", "cuda", "mps"}:
        raise ValueError("ROOMSCAN_BACKEND/backend_override must be auto, stub, cuda, or mps")

    selected = requested
    if selected == "auto":
        selected = "cuda" if cuda else "mps" if mps else "stub"
    if selected == "cuda" and not cuda:
        raise RuntimeError("CUDA backend was requested but torch.cuda.is_available() is false")
    if selected == "mps" and not mps:
        raise RuntimeError("MPS backend was requested but torch.backends.mps is unavailable")

    if selected == "stub":
        from .stub import StubBackend

        backend: Backend = StubBackend(config, project_root)
    elif selected == "cuda":
        from .cuda import CudaBackend

        backend = CudaBackend(config, project_root)
    else:
        from .mps import MpsBackend

        backend = MpsBackend(config, project_root)

    banner = f"ROOMSCAN BACKEND: {backend.name.upper()}"
    print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}\n", flush=True)
    return backend


__all__ = ["Backend", "ReconstructionResult", "select_backend"]
