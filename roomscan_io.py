"""Durable file-contract helpers shared by CLI, server, and backends."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(__file__).with_name("config.json")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "max_frames",
        "vggt_resolution",
        "vggt_preprocess_mode",
        "confidence_threshold",
        "blur_threshold",
        "frames_per_client",
        "frame_selection",
        "video_upload",
        "video_bits_per_second",
        "max_video_upload_bytes",
        "max_video_decode_seconds",
        "max_video_source_fps",
        "upload_timeout_seconds",
        "video_worker_count",
        "inference_queue_limit",
        "live_updates",
        "live_update_debounce_seconds",
        "live_update_max_wait_seconds",
        "viewer_refresh_seconds",
        "live_update_train_splat",
        "quality_max_camera_step_ratio",
        "quality_min_camera_steps",
        "model_revision_retention",
        "mask_people",
        "gsplat_iterations",
        "splat_max_screen_size",
        "splat_exposure",
        "point_size",
        "weights_dir",
        "backend_override",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return config


def resolve_weights_dir(config: dict[str, Any], root: Path) -> Path:
    env_value = os.getenv("ROOMSCAN_WEIGHTS_DIR")
    value = Path(env_value or str(config["weights_dir"])).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")
    return images


def evenly_subsample(items: Sequence[Any], limit: int) -> list[Any]:
    if limit < 1:
        raise ValueError("frame limit must be positive")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_copy_file(source: Path, target: Path) -> None:
    """Copy a potentially large artifact and expose it only after fsync."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def _rows(values: Any) -> Iterable[Any]:
    return values.tolist() if hasattr(values, "tolist") else values


def write_points_ply(path: Path, points: Any, colors: Any) -> None:
    point_rows = list(_rows(points))
    color_rows = list(_rows(colors))
    if len(point_rows) != len(color_rows):
        raise ValueError("point and color counts differ")
    lines = [
        "ply",
        "format ascii 1.0",
        "comment ROOMSCAN XYZ RGB",
        f"element vertex {len(point_rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point, color in zip(point_rows, color_rows):
        if len(point) != 3 or len(color) != 3:
            raise ValueError("points and colors must contain triples")
        if not all(math.isfinite(float(value)) for value in point):
            continue
        rgb = [max(0, min(255, int(round(float(value))))) for value in color]
        lines.append(
            f"{float(point[0]):.7g} {float(point[1]):.7g} {float(point[2]):.7g} "
            f"{rgb[0]} {rgb[1]} {rgb[2]}"
        )
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("ascii"))


def read_points_ply(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    with path.open("r", encoding="ascii") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = None
        is_ascii = False
        for line in handle:
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                is_ascii = True
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.rsplit(" ", 1)[1])
            if stripped == "end_header":
                break
        if not is_ascii or vertex_count is None:
            raise ValueError(f"Stub fixture must be an ASCII vertex PLY: {path}")
        points: list[list[float]] = []
        colors: list[list[int]] = []
        for _ in range(vertex_count):
            fields = handle.readline().split()
            if len(fields) < 6:
                raise ValueError(f"Malformed vertex row in {path}")
            points.append([float(value) for value in fields[:3]])
            colors.append([int(value) for value in fields[3:6]])
    return points, colors
