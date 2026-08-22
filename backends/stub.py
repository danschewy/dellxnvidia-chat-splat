from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from . import ReconstructionResult
from roomscan_io import read_points_ply


class StubBackend:
    name = "stub"

    def __init__(self, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root
        self.fixtures = root / "sample_data"

    def reconstruct(self, images: Sequence[Path]) -> ReconstructionResult:
        points_path = self.fixtures / "points.ply"
        cameras_path = self.fixtures / "cameras.json"
        if not points_path.is_file() or not cameras_path.is_file():
            raise FileNotFoundError(
                f"Stub fixtures are incomplete. Expected {points_path} and {cameras_path}"
            )
        points, colors = read_points_ply(points_path)
        with cameras_path.open("r", encoding="utf-8") as handle:
            fixture_cameras = json.load(handle)
        cameras = []
        for index, image in enumerate(images):
            source = fixture_cameras[index % len(fixture_cameras)]
            camera = dict(source)
            camera["frame"] = image.name
            cameras.append(camera)
        return ReconstructionResult(
            points=points,
            colors=colors,
            confidences=[0.95] * len(points),
            cameras=cameras,
        )

    def train_splat(
        self,
        poses: list[dict[str, Any]],
        points: ReconstructionResult,
        images: Sequence[Path],
    ) -> bytes | None:
        return None

    def segment_people(self, images: Sequence[Path], out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for image in images:
            target = out_dir / image.name
            shutil.copy2(image, target)
            outputs.append(target)
        return outputs
