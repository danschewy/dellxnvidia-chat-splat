#!/usr/bin/env python3
"""Offline ROOMSCAN pipeline: images on disk to durable reconstruction files."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator, Sequence

from backends import Backend, ReconstructionResult, select_backend
from roomscan_io import atomic_write_bytes, atomic_write_json, evenly_subsample, list_images, load_config, write_points_ply


ROOT = Path(__file__).resolve().parent


@contextmanager
def timed_stage(meta: dict[str, Any], name: str) -> Iterator[None]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    meta.setdefault("stages", {})[name] = record
    print(f"[stage:{name}] started", flush=True)
    try:
        yield
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        record["status"] = "ok"
    finally:
        record["seconds"] = round(time.perf_counter() - started, 4)
        print(f"[stage:{name}] {record['status']} in {record['seconds']:.3f}s", flush=True)


def _persist_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    atomic_write_json(out_dir / "meta.json", meta)


def _copy_inputs(images: Sequence[Path], frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if images and all(source.parent.resolve() == frames_dir.resolve() for source in images):
        return list(images)
    copied = []
    for index, source in enumerate(images):
        target = frames_dir / f"{index:03d}_{source.stem}.jpg"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def _resize_for_vggt(images: Sequence[Path], out_dir: Path, resolution: int) -> list[Path]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for CUDA/MPS image preparation") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in images:
        target = out_dir / source.name
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
            image.save(target, format="JPEG", quality=90, optimize=True)
        outputs.append(target)
    return outputs


def _write_masked_images(images: Sequence[Path], masks: Sequence[Any], out_dir: Path) -> list[Path]:
    if len(images) != len(masks):
        raise ValueError("segmentation backend returned the wrong number of masks")
    out_dir.mkdir(parents=True, exist_ok=True)
    if all(mask is None for mask in masks):
        outputs = []
        for image in images:
            target = out_dir / image.name
            shutil.copy2(image, target)
            outputs.append(target)
        return outputs
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to persist segmentation masks") from exc
    masks_dir = out_dir.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for image_path, mask in zip(images, masks):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image for masking: {image_path}")
        boolean_mask = np.zeros(image.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if boolean_mask.shape != image.shape[:2]:
            raise ValueError(f"Mask shape does not match frame: {image_path}")
        image[boolean_mask] = 0
        mask_path = masks_dir / f"{image_path.stem}.png"
        target = out_dir / image_path.name
        if not cv2.imwrite(str(mask_path), boolean_mask.astype(np.uint8) * 255):
            raise OSError(f"Could not write person mask: {mask_path}")
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"Could not write masked frame: {target}")
        outputs.append(target)
    return outputs


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _reconstruct_with_backoff(
    backend: Backend, images: list[Path], minimum: int = 4
) -> tuple[ReconstructionResult, list[Path]]:
    attempt = images
    while True:
        try:
            return backend.reconstruct(attempt), attempt
        except RuntimeError as exc:
            if not _is_oom(exc) or len(attempt) <= minimum:
                raise
            next_count = max(minimum, len(attempt) // 2)
            print(f"[reconstruct] accelerator OOM with {len(attempt)} frames; retrying with {next_count}", flush=True)
            attempt = evenly_subsample(attempt, next_count)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def _filter_points(result: ReconstructionResult, threshold: float) -> ReconstructionResult:
    if not 0.0 <= threshold < 1.0:
        raise ValueError("confidence_threshold must be in [0, 1)")
    try:
        import numpy as np
    except ImportError:
        keep = [index for index, value in enumerate(result.confidences) if float(value) >= threshold]
        return ReconstructionResult(
            points=[result.points[index] for index in keep],
            colors=[result.colors[index] for index in keep],
            confidences=[result.confidences[index] for index in keep],
            cameras=result.cameras,
        )
    points = np.asarray(result.points)
    colors = np.asarray(result.colors)
    confidences = np.asarray(result.confidences).reshape(-1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidences)
    valid_conf = confidences[finite]
    if not len(valid_conf):
        raise ValueError("VGGT produced no finite points")
    # VGGT confidence is not normalized. The 0..1 knob is the percentile
    # rejected, matching the upstream VGGT visualizer's convention.
    cutoff = np.quantile(valid_conf, threshold)
    keep = finite & (confidences >= cutoff)
    return ReconstructionResult(points[keep], colors[keep], confidences[keep], result.cameras)


def _exclude_masked_points(
    result: ReconstructionResult, masks: Sequence[Any]
) -> tuple[ReconstructionResult, int]:
    """Remove pixels identified as people from per-view dense geometry."""
    if not any(mask is not None for mask in masks):
        return result, 0
    if result.point_layout is None:
        raise ValueError("Reconstruction backend did not report its per-frame point layout")
    view_count, height, width = result.point_layout
    if view_count != len(masks):
        raise ValueError("Person-mask count does not match reconstructed view count")

    import cv2
    import numpy as np

    flattened_masks = []
    for mask in masks:
        if mask is None:
            resized = np.zeros((height, width), dtype=bool)
        else:
            values = np.asarray(mask, dtype=np.uint8)
            resized = cv2.resize(
                values, (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        flattened_masks.append(resized.reshape(-1))
    person_points = np.concatenate(flattened_masks)
    if len(person_points) != len(result.points):
        raise ValueError("Person masks do not align with reconstructed points")
    keep = ~person_points
    removed = int(person_points.sum())
    return ReconstructionResult(
        points=np.asarray(result.points)[keep],
        colors=np.asarray(result.colors)[keep],
        confidences=np.asarray(result.confidences).reshape(-1)[keep],
        cameras=result.cameras,
    ), removed


def run_pipeline(image_dir: Path, out_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    backend = select_backend(config, ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "backend": backend.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "input_frame_count": 0,
        "selected_frame_count": 0,
        "point_count": 0,
        "splat_available": False,
        "stages": {},
    }
    try:
        with timed_stage(meta, "ingest"):
            all_images = list_images(image_dir)
            meta["input_frame_count"] = len(all_images)
            selected = evenly_subsample(all_images, int(config["max_frames"]))
            frames = _copy_inputs(selected, out_dir / "frames")
            meta["selected_frame_count"] = len(frames)
        _persist_meta(out_dir, meta)

        with timed_stage(meta, "resize"):
            prepared = frames if backend.name == "stub" else _resize_for_vggt(
                frames, out_dir / "work" / "resized", int(config["vggt_resolution"])
            )
        _persist_meta(out_dir, meta)

        with timed_stage(meta, "mask_people"):
            masks_by_frame: dict[str, Any] = {}
            if bool(config["mask_people"]):
                masks = backend.segment_people(prepared)
                prepared = _write_masked_images(prepared, masks, out_dir / "work" / "masked")
                masks_by_frame = {
                    image.name: mask for image, mask in zip(prepared, masks)
                }
            else:
                meta["stages"]["mask_people"]["detail"] = "disabled in config"
        _persist_meta(out_dir, meta)

        with timed_stage(meta, "reconstruct"):
            reconstruction, used_images = _reconstruct_with_backoff(backend, list(prepared))
            reconstruction, removed_people_points = _exclude_masked_points(
                reconstruction,
                [masks_by_frame.get(image.name) for image in used_images],
            )
            reconstruction = _filter_points(reconstruction, float(config["confidence_threshold"]))
            write_points_ply(out_dir / "points.ply", reconstruction.points, reconstruction.colors)
            atomic_write_json(out_dir / "cameras.json", reconstruction.cameras)
            meta["selected_frame_count"] = len(used_images)
            meta["point_count"] = len(reconstruction.points)
            meta["masked_point_count"] = removed_people_points
        _persist_meta(out_dir, meta)

        # Point and camera artifacts have been fsynced before this optional stage.
        try:
            with timed_stage(meta, "train_splat"):
                splat = backend.train_splat(reconstruction.cameras, reconstruction, used_images)
                if splat is not None:
                    atomic_write_bytes(out_dir / "splat.ply", splat)
                    meta["splat_available"] = True
                else:
                    meta["stages"]["train_splat"]["detail"] = "unsupported by selected backend"
        except BaseException as exc:
            print(f"[stage:train_splat] non-fatal fallback to points.ply: {exc}", file=sys.stderr, flush=True)
        _persist_meta(out_dir, meta)
    except BaseException:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        _persist_meta(out_dir, meta)
        raise

    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["total_seconds"] = round(sum(float(stage.get("seconds", 0.0)) for stage in meta["stages"].values()), 4)
    _persist_meta(out_dir, meta)
    print(f"ROOMSCAN complete: {meta['point_count']} points, backend={backend.name}, total={meta['total_seconds']:.3f}s", flush=True)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args()
    run_pipeline(arguments.image_dir.resolve(), arguments.out_dir.resolve(), arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
