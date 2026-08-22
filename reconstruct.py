#!/usr/bin/env python3
"""Offline ROOMSCAN pipeline: images on disk to durable reconstruction files."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import math
import statistics
import shutil
import sys
import time
from typing import Any, Callable, Iterator, Sequence

from backends import Backend, ReconstructionResult, select_backend
from multiphone import (
    DenseSubmap,
    align_submaps,
    concatenate_reconstructions,
    frame_identity,
    group_images_by_capture,
    group_images_by_device,
    identity_similarity,
    select_capture_aware,
    transform_reconstruction,
)
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


def _resize_for_vggt(
    images: Sequence[Path],
    out_dir: Path,
    resolution: int,
    mode: str,
    portrait_height: int | None = None,
) -> list[Path]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for CUDA/MPS image preparation") from exc
    if mode not in {"crop", "pad"}:
        raise ValueError("vggt_preprocess_mode must be crop or pad")
    portrait_height = resolution if portrait_height is None else portrait_height
    if portrait_height < resolution or portrait_height % 14:
        raise ValueError("vggt_portrait_height must be >= vggt_resolution and divisible by 14")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in images:
        target = out_dir / source.name
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            if mode == "crop":
                target_width = resolution
                target_height = max(
                    14, round(height * (target_width / width) / 14) * 14
                )
                image = image.resize(
                    (target_width, target_height), Image.Resampling.BICUBIC
                )
                if target_height > portrait_height:
                    top = (target_height - portrait_height) // 2
                    image = image.crop((0, top, resolution, top + portrait_height))
            else:
                if width >= height:
                    target_width = resolution
                    target_height = max(
                        14, round(height * (target_width / width) / 14) * 14
                    )
                else:
                    target_height = resolution
                    target_width = max(
                        14, round(width * (target_height / height) / 14) * 14
                    )
                image = image.resize(
                    (target_width, target_height), Image.Resampling.BICUBIC
                )
                canvas = Image.new("RGB", (resolution, resolution), (255, 255, 255))
                canvas.paste(
                    image,
                    ((resolution - target_width) // 2, (resolution - target_height) // 2),
                )
                image = canvas
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


def _filter_points(
    result: ReconstructionResult,
    threshold: float,
    max_points: int | None = None,
) -> ReconstructionResult:
    if not 0.0 <= threshold < 1.0:
        raise ValueError("confidence_threshold must be in [0, 1)")
    if max_points is not None and max_points < 1:
        raise ValueError("max_point_cloud_points must be positive")
    try:
        import numpy as np
    except ImportError:
        keep = [index for index, value in enumerate(result.confidences) if float(value) >= threshold]
        if max_points is not None and len(keep) > max_points:
            step = (len(keep) - 1) / max(max_points - 1, 1)
            keep = [keep[round(index * step)] for index in range(max_points)]
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
    # rejected, matching the upstream VGGT visualizer's convention. Difficult
    # footage commonly has millions of values tied at VGGT's exact 1.0 floor.
    # Select an exact bounded count: keep every value above the boundary, then
    # fill the remaining quota evenly across the tie. This preserves wall
    # coverage without accidentally retaining every floor-saturated pixel.
    valid_indices = np.flatnonzero(finite)
    all_equal = bool(np.all(valid_conf == valid_conf[0]))
    desired = len(valid_indices) if all_equal else max(1, math.ceil(len(valid_indices) * (1 - threshold)))
    if max_points is not None:
        desired = min(desired, max_points)
    if desired >= len(valid_indices):
        selected = valid_indices
    else:
        boundary = np.partition(valid_conf, len(valid_conf) - desired)[len(valid_conf) - desired]
        higher = valid_indices[valid_conf > boundary]
        tied = valid_indices[valid_conf == boundary]
        tie_count = desired - len(higher)
        if tie_count >= len(tied):
            selected_ties = tied
        elif tie_count == 1:
            selected_ties = tied[[len(tied) // 2]]
        else:
            tie_positions = np.linspace(0, len(tied) - 1, tie_count, dtype=np.int64)
            selected_ties = tied[tie_positions]
        selected = np.sort(np.concatenate((higher, selected_ties)))
    return ReconstructionResult(
        points[selected], colors[selected], confidences[selected], result.cameras
    )


def _camera_path_quality(
    cameras: Sequence[dict[str, Any]],
    max_step_ratio: float,
    minimum_steps: int,
) -> dict[str, Any]:
    """Measure trajectory jumps within clips without comparing different phones."""
    if max_step_ratio <= 1:
        raise ValueError("quality_max_camera_step_ratio must be greater than 1")
    if minimum_steps < 1:
        raise ValueError("quality_min_camera_steps must be positive")

    captures: dict[str, list[list[float]]] = {}
    all_positions: list[list[float]] = []
    for index, camera in enumerate(cameras):
        transform = camera.get("T_wc")
        if not isinstance(transform, list) or len(transform) < 3:
            continue
        try:
            position = [float(transform[axis][3]) for axis in range(3)]
        except (IndexError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in position):
            continue
        frame = Path(str(camera.get("frame", index)))
        identity = frame_identity(frame)
        capture = identity.capture_id
        captures.setdefault(capture, []).append(position)
        all_positions.append(position)

    eligible = {
        capture: positions
        for capture, positions in captures.items()
        if len(positions) - 1 >= minimum_steps
    }
    if not eligible and len(all_positions) - 1 >= minimum_steps:
        eligible = {"__sequence__": all_positions}

    per_capture = []
    for capture, positions in eligible.items():
        steps = [math.dist(first, second) for first, second in zip(positions, positions[1:])]
        if not steps:
            continue
        median_step = statistics.median(steps)
        largest_step = max(steps)
        ratio = largest_step / max(median_step, 1e-9)
        per_capture.append(
            {
                "capture": capture,
                "step_count": len(steps),
                "median_step": round(median_step, 7),
                "max_step": round(largest_step, 7),
                "max_step_ratio": round(ratio, 4),
            }
        )

    worst = max(per_capture, key=lambda item: item["max_step_ratio"], default=None)
    observed_ratio = float(worst["max_step_ratio"]) if worst else None
    return {
        "evaluated_capture_count": len(per_capture),
        "max_allowed_step_ratio": max_step_ratio,
        "max_step_ratio": observed_ratio,
        "worst_capture": worst["capture"] if worst else None,
        "continuous": observed_ratio is None or observed_ratio <= max_step_ratio,
        "captures": per_capture,
    }


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


def _select_pipeline_inputs(
    images: Sequence[Path], config: dict[str, Any], backend_name: str
) -> tuple[list[Path], dict[str, Any]]:
    device_groups = group_images_by_device(images)
    use_submaps = (
        bool(config["multi_device_submaps"])
        and backend_name != "stub"
        and len(device_groups) > 1
    )
    if not use_submaps:
        selected = select_capture_aware(
            images,
            int(config["max_frames"]),
            int(config["submap_frames_per_capture"]),
        )
        return selected, {
            "mode": "single_sequence",
            "devices": sorted(device_groups),
            "selected_frames_by_device": {
                device: sum(frame_identity(path).device_id == device for path in selected)
                for device in device_groups
            },
        }

    selected = []
    selected_counts = {}
    for device, paths in device_groups.items():
        capture_groups = list(group_images_by_capture(paths).items())
        chosen_captures = evenly_subsample(
            capture_groups,
            min(int(config["submaps_per_device"]), len(capture_groups)),
        )
        device_selection = []
        for _capture, capture_paths in chosen_captures:
            device_selection.extend(
                select_capture_aware(
                    capture_paths,
                    int(config["submap_max_frames"]),
                    int(config["submap_frames_per_capture"]),
                )
            )
        selected.extend(device_selection)
        selected_counts[device] = len(device_selection)
    return selected, {
        "mode": "per_capture_submaps",
        "devices": sorted(device_groups),
        "selected_frames_by_device": selected_counts,
    }


def _reconstruct_capture_submaps(
    backend: Backend,
    prepared: Sequence[Path],
    masks_by_frame: dict[str, Any],
    config: dict[str, Any],
) -> tuple[ReconstructionResult, list[Path], int, dict[str, Any]]:
    grouped = group_images_by_capture(prepared)
    submaps = []
    for capture, images in grouped.items():
        dense, used_images = _reconstruct_with_backoff(backend, list(images))
        submaps.append(
            DenseSubmap(capture, frame_identity(images[0]).device_id, dense, used_images)
        )

    try:
        transforms, alignment = align_submaps(submaps, config)
    except BaseException as exc:
        anchor = max(submaps, key=lambda item: (len(item.images), item.submap_id))
        transforms = {anchor.submap_id: identity_similarity()}
        alignment = {
            "mode": "per_capture_submaps",
            "anchor_submap": anchor.submap_id,
            "anchor_device": anchor.device_id,
            "submap_count": len(submaps),
            "device_count": len({item.device_id for item in submaps}),
            "aligned_submaps": [anchor.submap_id],
            "skipped_submaps": sorted(
                item.submap_id for item in submaps if item.submap_id != anchor.submap_id
            ),
            "aligned_devices": [anchor.device_id],
            "skipped_devices": sorted(
                {
                    item.device_id
                    for item in submaps
                    if item.device_id != anchor.device_id
                }
            ),
            "links": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    by_submap = {submap.submap_id: submap for submap in submaps}
    point_quota = max(1, int(config["max_point_cloud_points"]) // len(transforms))
    aligned_reconstructions = []
    aligned_images = []
    removed_people_points = 0
    for submap_id in alignment["aligned_submaps"]:
        submap = by_submap[submap_id]
        cleaned, removed = _exclude_masked_points(
            submap.reconstruction,
            [masks_by_frame.get(image.name) for image in submap.images],
        )
        cleaned = _filter_points(
            cleaned,
            float(config["confidence_threshold"]),
            max_points=point_quota,
        )
        aligned_reconstructions.append(
            transform_reconstruction(cleaned, transforms[submap_id])
        )
        aligned_images.extend(submap.images)
        removed_people_points += removed
    return (
        concatenate_reconstructions(aligned_reconstructions),
        aligned_images,
        removed_people_points,
        alignment,
    )


def run_pipeline(
    image_dir: Path,
    out_dir: Path,
    config_path: Path | None = None,
    geometry_ready: Callable[[Path], None] | None = None,
    train_splat: bool = True,
    backend: Backend | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    backend = backend or select_backend(config, ROOT)
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
            selected, selection = _select_pipeline_inputs(
                all_images, config, backend.name
            )
            frames = _copy_inputs(selected, out_dir / "frames")
            meta["selected_frame_count"] = len(frames)
            meta["selection"] = selection
        _persist_meta(out_dir, meta)

        with timed_stage(meta, "resize"):
            prepared = frames if backend.name == "stub" else _resize_for_vggt(
                frames,
                out_dir / "work" / "resized",
                int(config["vggt_resolution"]),
                str(config["vggt_preprocess_mode"]),
                portrait_height=int(config["vggt_portrait_height"]),
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
            prepared_groups = group_images_by_device(prepared)
            if (
                meta["selection"]["mode"] == "per_capture_submaps"
                and len(prepared_groups) > 1
            ):
                (
                    reconstruction,
                    used_images,
                    removed_people_points,
                    alignment,
                ) = _reconstruct_capture_submaps(
                    backend, prepared, masks_by_frame, config
                )
            else:
                reconstruction, used_images = _reconstruct_with_backoff(
                    backend, list(prepared)
                )
                reconstruction, removed_people_points = _exclude_masked_points(
                    reconstruction,
                    [masks_by_frame.get(image.name) for image in used_images],
                )
                reconstruction = _filter_points(
                    reconstruction,
                    float(config["confidence_threshold"]),
                    max_points=int(config["max_point_cloud_points"]),
                )
                only_device = next(iter(prepared_groups), "__sequence__")
                alignment = {
                    "mode": "single_sequence",
                    "anchor_device": only_device,
                    "device_count": 1,
                    "aligned_devices": [only_device],
                    "skipped_devices": [],
                    "links": [],
                }
            write_points_ply(out_dir / "points.ply", reconstruction.points, reconstruction.colors)
            atomic_write_json(out_dir / "cameras.json", reconstruction.cameras)
            meta["selected_frame_count"] = len(used_images)
            meta["point_count"] = len(reconstruction.points)
            meta["masked_point_count"] = removed_people_points
            camera_quality = _camera_path_quality(
                reconstruction.cameras,
                float(config["quality_max_camera_step_ratio"]),
                int(config["quality_min_camera_steps"]),
            )
            warnings = [
                f"unaligned_submap:{submap}"
                for submap in alignment.get("skipped_submaps", [])
            ]
            blocking_warnings = []
            if not camera_quality["continuous"]:
                camera_warning = (
                    "camera_path_discontinuity: "
                    f"{camera_quality['worst_capture']} has step ratio "
                    f"{camera_quality['max_step_ratio']}"
                )
                warnings.append(camera_warning)
                blocking_warnings.append(camera_warning)
            meta["quality"] = {
                "camera_path": camera_quality,
                "alignment": alignment,
                "warnings": warnings,
                "blocking_warnings": blocking_warnings,
            }
        _persist_meta(out_dir, meta)
        if geometry_ready is not None:
            geometry_ready(out_dir)

        # Point and camera artifacts have been fsynced before this optional stage.
        try:
            with timed_stage(meta, "train_splat"):
                if train_splat:
                    splat = backend.train_splat(
                        reconstruction.cameras, reconstruction, used_images
                    )
                else:
                    splat = None
                    meta["stages"]["train_splat"]["detail"] = "disabled for live update"
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
    parser.add_argument(
        "--no-splat",
        action="store_true",
        help="persist point geometry and cameras without running optional splat training",
    )
    arguments = parser.parse_args()
    run_pipeline(
        arguments.image_dir.resolve(),
        arguments.out_dir.resolve(),
        arguments.config,
        train_splat=not arguments.no_splat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
