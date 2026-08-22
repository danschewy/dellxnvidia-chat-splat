"""Capture-aware sampling and robust alignment for independent phone submaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import re
from typing import Any, Sequence

from backends import ReconstructionResult
from roomscan_io import evenly_subsample


_COPY_PREFIX = re.compile(r"^\d{3}_(.+)$")


@dataclass(frozen=True)
class FrameIdentity:
    device_id: str
    capture_id: str
    frame_index: int


@dataclass(frozen=True)
class SimilarityTransform:
    """Map a point with ``target = scale * rotation @ source + translation``."""

    scale: float
    rotation: Any
    translation: Any


@dataclass
class DenseSubmap:
    submap_id: str
    device_id: str
    reconstruction: ReconstructionResult
    images: list[Path]


def frame_identity(path: Path | str) -> FrameIdentity:
    """Recover stable device/capture identity after pipeline copy prefixes."""

    stem = Path(path).stem
    while True:
        matched = _COPY_PREFIX.fullmatch(stem)
        if matched is None:
            break
        remainder = matched.group(1)
        prefix, separator, suffix = remainder.rpartition("_")
        if not separator or not suffix.isdigit():
            break
        stem = remainder

    capture_id, separator, frame_value = stem.rpartition("_")
    if not separator or not frame_value.isdigit():
        return FrameIdentity(stem, stem, 0)
    device_id, device_separator, capture_token = capture_id.rpartition("-")
    if not device_separator or not device_id or not capture_token:
        device_id = capture_id
    return FrameIdentity(device_id, capture_id, int(frame_value))


def group_images_by_device(images: Sequence[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image in images:
        grouped.setdefault(frame_identity(image).device_id, []).append(image)
    return {device: sorted(paths) for device, paths in sorted(grouped.items())}


def group_images_by_capture(images: Sequence[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image in images:
        grouped.setdefault(frame_identity(image).capture_id, []).append(image)
    return {capture: sorted(paths) for capture, paths in sorted(grouped.items())}


def select_capture_aware(
    images: Sequence[Path], limit: int, frames_per_capture: int
) -> list[Path]:
    """Keep short local motion sequences instead of isolated frames per clip."""

    if limit < 1:
        raise ValueError("frame limit must be positive")
    if frames_per_capture < 2:
        raise ValueError("submap_frames_per_capture must be at least 2")
    if len(images) <= limit:
        return list(images)

    captures: dict[str, list[Path]] = {}
    for image in sorted(images):
        captures.setdefault(frame_identity(image).capture_id, []).append(image)
    capture_groups = sorted(captures.items())
    capture_budget = max(1, limit // frames_per_capture)
    chosen_groups = evenly_subsample(capture_groups, min(capture_budget, len(capture_groups)))

    selected: list[Path] = []
    for _capture_id, paths in chosen_groups:
        remaining = limit - len(selected)
        if remaining <= 0:
            break
        selected.extend(evenly_subsample(paths, min(frames_per_capture, remaining)))

    if len(selected) < limit:
        selected_set = set(selected)
        unused = [image for image in sorted(images) if image not in selected_set]
        selected.extend(evenly_subsample(unused, min(limit - len(selected), len(unused))))
    return selected[:limit]


def identity_similarity() -> SimilarityTransform:
    import numpy as np

    return SimilarityTransform(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))


def compose_similarity(
    outer: SimilarityTransform, inner: SimilarityTransform
) -> SimilarityTransform:
    """Return a transform that applies ``inner`` and then ``outer``."""

    import numpy as np

    outer_rotation = np.asarray(outer.rotation, dtype=np.float64)
    inner_rotation = np.asarray(inner.rotation, dtype=np.float64)
    outer_translation = np.asarray(outer.translation, dtype=np.float64)
    inner_translation = np.asarray(inner.translation, dtype=np.float64)
    return SimilarityTransform(
        scale=float(outer.scale * inner.scale),
        rotation=outer_rotation @ inner_rotation,
        translation=(
            outer.scale * (outer_rotation @ inner_translation) + outer_translation
        ),
    )


def _umeyama_similarity(source: Any, target: Any) -> SimilarityTransform:
    import numpy as np

    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.shape != target_values.shape or source_values.ndim != 2:
        raise ValueError("Similarity correspondences must have matching Nx3 shapes")
    if source_values.shape[0] < 3 or source_values.shape[1] != 3:
        raise ValueError("At least three 3D correspondences are required")

    source_mean = source_values.mean(axis=0)
    target_mean = target_values.mean(axis=0)
    source_centered = source_values - source_mean
    target_centered = target_values - target_mean
    source_variance = float(np.sum(source_centered**2) / len(source_values))
    if source_variance <= 1e-12:
        raise ValueError("Source correspondences are degenerate")

    covariance = target_centered.T @ source_centered / len(source_values)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        sign[-1, -1] = -1
    rotation = left @ sign @ right_transpose
    scale = float(np.sum(singular_values * np.diag(sign)) / source_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated similarity scale is invalid")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def apply_similarity(points: Any, transform: SimilarityTransform) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(transform.rotation, dtype=np.float64)
    translation = np.asarray(transform.translation, dtype=np.float64)
    return transform.scale * (values @ rotation.T) + translation


def estimate_similarity_ransac(
    source: Any,
    target: Any,
    *,
    threshold: float,
    iterations: int,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
    scale_min: float,
    scale_max: float,
) -> tuple[SimilarityTransform, Any, float] | None:
    """Estimate a deterministic robust Sim(3) transform from 3D matches."""

    import numpy as np

    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    count = len(source_values)
    if (
        source_values.shape != target_values.shape
        or source_values.shape != (count, 3)
        or count < max(3, minimum_inliers)
    ):
        return None
    if threshold <= 0 or iterations < 1:
        raise ValueError("Alignment RANSAC threshold and iterations must be positive")

    generator = np.random.default_rng(0)
    best_inliers = None
    best_median = math.inf
    for _ in range(iterations):
        indices = generator.choice(count, size=3, replace=False)
        try:
            transform = _umeyama_similarity(source_values[indices], target_values[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not scale_min <= transform.scale <= scale_max:
            continue
        errors = np.linalg.norm(apply_similarity(source_values, transform) - target_values, axis=1)
        inliers = errors <= threshold
        inlier_count = int(inliers.sum())
        median = float(np.median(errors[inliers])) if inlier_count else math.inf
        if best_inliers is None or (inlier_count, -median) > (int(best_inliers.sum()), -best_median):
            best_inliers = inliers
            best_median = median

    if best_inliers is None:
        return None
    required = max(minimum_inliers, math.ceil(count * minimum_inlier_ratio))
    if int(best_inliers.sum()) < required:
        return None
    try:
        refined = _umeyama_similarity(source_values[best_inliers], target_values[best_inliers])
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not scale_min <= refined.scale <= scale_max:
        return None
    refined_errors = np.linalg.norm(
        apply_similarity(source_values, refined) - target_values, axis=1
    )
    refined_inliers = refined_errors <= threshold
    if int(refined_inliers.sum()) < required:
        return None
    median_error = float(np.median(refined_errors[refined_inliers]))
    return refined, refined_inliers, median_error


def _feature_candidates(submap: DenseSubmap, image_limit: int, feature_count: int) -> list[dict[str, Any]]:
    import cv2

    candidate_paths = select_capture_aware(
        submap.images, min(image_limit, len(submap.images)), 2
    )
    indices = {path: index for index, path in enumerate(submap.images)}
    detector = cv2.SIFT_create(nfeatures=feature_count)
    candidates = []
    for path in candidate_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 3:
            continue
        candidates.append(
            {
                "path": path,
                "view_index": indices[path],
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )
    return candidates


def _point_for_keypoint(
    reconstruction: ReconstructionResult, view_index: int, keypoint: Any
) -> tuple[Any, float] | None:
    import numpy as np

    if reconstruction.point_layout is None:
        return None
    view_count, height, width = reconstruction.point_layout
    x = int(round(float(keypoint.pt[0])))
    y = int(round(float(keypoint.pt[1])))
    if not (0 <= view_index < view_count and 0 <= x < width and 0 <= y < height):
        return None
    index = view_index * height * width + y * width + x
    point = np.asarray(reconstruction.points[index], dtype=np.float64)
    confidence = float(np.asarray(reconstruction.confidences).reshape(-1)[index])
    if not np.isfinite(point).all() or not math.isfinite(confidence):
        return None
    return point, confidence


def _scene_extent(points: Any) -> float:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    if len(values) > 50_000:
        indices = np.linspace(0, len(values) - 1, 50_000, dtype=np.int64)
        values = values[indices]
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return 0.0
    center = np.median(values, axis=0)
    radius = float(np.quantile(np.linalg.norm(values - center, axis=1), 0.9))
    return max(radius * 2.0, 1e-9)


def estimate_submap_alignment(
    target: DenseSubmap,
    source: DenseSubmap,
    config: dict[str, Any],
    *,
    target_features: list[dict[str, Any]] | None = None,
    source_features: list[dict[str, Any]] | None = None,
) -> tuple[SimilarityTransform, dict[str, Any]] | None:
    """Align ``source`` into ``target`` coordinates using shared visual points."""

    import cv2
    import numpy as np

    if target_features is None:
        target_features = _feature_candidates(
            target,
            int(config["alignment_max_images_per_submap"]),
            int(config["alignment_feature_count"]),
        )
    if source_features is None:
        source_features = _feature_candidates(
            source,
            int(config["alignment_max_images_per_submap"]),
            int(config["alignment_feature_count"]),
        )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pair_correspondences = []
    ratio = float(config["alignment_ratio_test"])
    minimum_pair_matches = int(config["alignment_min_pair_matches"])
    for source_view in source_features:
        for target_view in target_features:
            matches = matcher.knnMatch(
                source_view["descriptors"], target_view["descriptors"], k=2
            )
            good = [
                first
                for pair in matches
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < ratio * second.distance
            ]
            source_points = []
            target_points = []
            for match in good:
                source_point = _point_for_keypoint(
                    source.reconstruction,
                    source_view["view_index"],
                    source_view["keypoints"][match.queryIdx],
                )
                target_point = _point_for_keypoint(
                    target.reconstruction,
                    target_view["view_index"],
                    target_view["keypoints"][match.trainIdx],
                )
                if source_point is None or target_point is None:
                    continue
                source_points.append(source_point[0])
                target_points.append(target_point[0])
            if len(source_points) >= minimum_pair_matches:
                pair_correspondences.append(
                    (len(source_points), source_points, target_points)
                )

    pair_correspondences.sort(key=lambda item: item[0], reverse=True)
    retained_pairs = pair_correspondences[: int(config["alignment_max_image_pairs"])]
    if len(retained_pairs) < int(config["alignment_min_image_pairs"]):
        return None
    source_points = np.asarray(
        [point for _count, values, _targets in retained_pairs for point in values],
        dtype=np.float64,
    )
    target_points = np.asarray(
        [point for _count, _sources, values in retained_pairs for point in values],
        dtype=np.float64,
    )
    threshold = (
        _scene_extent(target.reconstruction.points)
        * float(config["alignment_ransac_threshold_ratio"])
    )
    estimated = estimate_similarity_ransac(
        source_points,
        target_points,
        threshold=threshold,
        iterations=int(config["alignment_ransac_iterations"]),
        minimum_inliers=int(config["alignment_min_inliers"]),
        minimum_inlier_ratio=float(config["alignment_min_inlier_ratio"]),
        scale_min=float(config["alignment_scale_min"]),
        scale_max=float(config["alignment_scale_max"]),
    )
    if estimated is None:
        return None
    transform, inliers, median_error = estimated
    return transform, {
        "source": source.submap_id,
        "target": target.submap_id,
        "image_pairs": len(retained_pairs),
        "matches": len(source_points),
        "inliers": int(np.asarray(inliers).sum()),
        "inlier_ratio": round(float(np.asarray(inliers).mean()), 4),
        "median_error_ratio": round(median_error / max(threshold, 1e-12), 4),
        "scale": round(float(transform.scale), 6),
    }


def align_submaps(
    submaps: Sequence[DenseSubmap], config: dict[str, Any]
) -> tuple[dict[str, SimilarityTransform], dict[str, Any]]:
    """Build a verified alignment graph rooted at the best-covered phone."""

    if not submaps:
        raise ValueError("At least one submap is required")
    ordered = sorted(submaps, key=lambda item: (-len(item.images), item.submap_id))
    by_submap = {submap.submap_id: submap for submap in ordered}
    anchor = ordered[0].submap_id
    transforms = {anchor: identity_similarity()}
    alignment_order = [anchor]
    pending = {submap.submap_id for submap in ordered[1:]}
    links = []
    feature_cache = {
        submap.submap_id: _feature_candidates(
            submap,
            int(config["alignment_max_images_per_submap"]),
            int(config["alignment_feature_count"]),
        )
        for submap in ordered
    }
    alignment_cache: dict[
        tuple[str, str], tuple[SimilarityTransform, dict[str, Any]] | None
    ] = {}

    while pending:
        best = None
        for source_submap in sorted(pending):
            for target_submap in sorted(transforms):
                cache_key = (target_submap, source_submap)
                if cache_key not in alignment_cache:
                    alignment_cache[cache_key] = estimate_submap_alignment(
                        by_submap[target_submap],
                        by_submap[source_submap],
                        config,
                        target_features=feature_cache[target_submap],
                        source_features=feature_cache[source_submap],
                    )
                estimated = alignment_cache[cache_key]
                if estimated is None:
                    continue
                transform, detail = estimated
                score = (detail["inliers"], detail["inlier_ratio"], -detail["median_error_ratio"])
                if best is None or score > best[0]:
                    best = (score, source_submap, target_submap, transform, detail)
        if best is None:
            break
        _score, source_submap, target_submap, transform, detail = best
        transforms[source_submap] = compose_similarity(
            transforms[target_submap], transform
        )
        alignment_order.append(source_submap)
        pending.remove(source_submap)
        links.append(detail)

    aligned_devices = sorted({by_submap[name].device_id for name in transforms})
    all_devices = {submap.device_id for submap in submaps}
    skipped_devices = sorted(all_devices - set(aligned_devices))
    return transforms, {
        "mode": "per_capture_submaps",
        "anchor_submap": anchor,
        "anchor_device": by_submap[anchor].device_id,
        "submap_count": len(submaps),
        "device_count": len({submap.device_id for submap in submaps}),
        "aligned_submaps": alignment_order,
        "skipped_submaps": sorted(pending),
        "aligned_devices": aligned_devices,
        "skipped_devices": skipped_devices,
        "links": links,
    }


def transform_reconstruction(
    reconstruction: ReconstructionResult, transform: SimilarityTransform
) -> ReconstructionResult:
    import numpy as np

    cameras = copy.deepcopy(reconstruction.cameras)
    rotation = np.asarray(transform.rotation, dtype=np.float64)
    translation = np.asarray(transform.translation, dtype=np.float64)
    for camera in cameras:
        matrix = np.asarray(camera["T_wc"], dtype=np.float64)
        matrix[:3, :3] = rotation @ matrix[:3, :3]
        matrix[:3, 3] = (
            transform.scale * (rotation @ matrix[:3, 3]) + translation
        )
        camera["T_wc"] = matrix.tolist()
    return ReconstructionResult(
        points=apply_similarity(reconstruction.points, transform).astype(np.float32),
        colors=np.asarray(reconstruction.colors),
        confidences=np.asarray(reconstruction.confidences),
        cameras=cameras,
    )


def concatenate_reconstructions(
    reconstructions: Sequence[ReconstructionResult],
) -> ReconstructionResult:
    import numpy as np

    if not reconstructions:
        raise ValueError("At least one aligned reconstruction is required")
    return ReconstructionResult(
        points=np.concatenate([np.asarray(item.points) for item in reconstructions]),
        colors=np.concatenate([np.asarray(item.colors) for item in reconstructions]),
        confidences=np.concatenate(
            [np.asarray(item.confidences).reshape(-1) for item in reconstructions]
        ),
        cameras=[camera for item in reconstructions for camera in item.cameras],
    )
