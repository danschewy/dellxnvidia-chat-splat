"""Offline phone-video ingestion into the durable ROOMSCAN frame contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roomscan_io import atomic_write_bytes


VIDEO_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}


@dataclass(frozen=True)
class ExtractionResult:
    frames: list[Path]
    decoded_frames: int
    sampled_frames: int
    passing_frames: int
    truncated: bool = False


def video_extension(mime_type: str) -> str:
    normalized = mime_type.split(";", 1)[0].strip().lower()
    try:
        return VIDEO_EXTENSIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported video MIME type: {mime_type}") from exc


def select_frame_candidates(
    candidates: list[tuple[float, int, bytes]],
    limit: int,
    mode: str,
) -> list[tuple[float, int, bytes]]:
    """Select sharp frames without discarding temporal bridges between views."""
    if limit < 1:
        raise ValueError("frame selection limit must be positive")
    if len(candidates) <= limit:
        return list(candidates)
    if mode == "sharpest":
        selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]
        return sorted(selected, key=lambda item: item[1])
    if mode != "temporal_sharpness":
        raise ValueError("frame_selection must be temporal_sharpness or sharpest")
    selected = []
    for window in range(limit):
        start = window * len(candidates) // limit
        end = (window + 1) * len(candidates) // limit
        selected.append(max(candidates[start:end], key=lambda item: item[0]))
    return selected


def _apply_video_orientation(frame: Any, degrees: int) -> Any:
    """Apply container rotation when the OpenCV backend does not do it."""
    import cv2

    normalized = degrees % 360
    if normalized == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_sharp_frames(
    video_path: Path,
    frames_dir: Path,
    client_id: str,
    config: dict[str, Any],
) -> ExtractionResult:
    """Decode, rank by sharpness, and persist selected frames in time order."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not decode uploaded video: {video_path.name}")

    orientation_meta_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    orientation_auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    orientation_degrees = (
        int(round(capture.get(orientation_meta_property)))
        if orientation_meta_property is not None
        else 0
    )
    orientation_is_automatic = False
    if orientation_auto_property is not None:
        capture.set(orientation_auto_property, 1)
        orientation_is_automatic = capture.get(orientation_auto_property) >= 0.5

    target_fps = float(config["capture_fps"])
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_limit = int(config["frames_per_client"])
    target_width = int(config["capture_width"])
    if target_fps <= 0 or frame_limit <= 0 or target_width <= 0:
        capture.release()
        raise ValueError("capture_fps, frames_per_client, and capture_width must be positive")
    jpeg_quality = int(round(float(config["jpeg_quality"]) * 100))
    blur_threshold = float(config["blur_threshold"])
    max_decode_seconds = float(config["max_video_decode_seconds"])
    max_source_fps = float(config["max_video_source_fps"])
    if max_decode_seconds <= 0 or max_source_fps <= 0:
        capture.release()
        raise ValueError("max_video_decode_seconds and max_video_source_fps must be positive")
    decode_fps = min(source_fps, max_source_fps) if source_fps > 0 else max_source_fps
    sample_stride = max(1, round(decode_fps / target_fps))
    max_decoded_frames = max(1, round(max_decode_seconds * decode_fps))
    candidates: list[tuple[float, int, bytes]] = []
    decoded = 0
    sampled = 0
    passing = 0
    truncated = False

    try:
        while True:
            if decoded >= max_decoded_frames:
                truncated = True
                break
            ok, frame = capture.read()
            if not ok:
                break
            if not orientation_is_automatic:
                frame = _apply_video_orientation(frame, orientation_degrees)
            frame_index = decoded
            decoded += 1
            if frame_index % sample_stride:
                continue
            sampled += 1
            height, width = frame.shape[:2]
            target_height = max(1, round(height * target_width / width))
            resized = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if score >= blur_threshold:
                passing += 1
            encoded, buffer = cv2.imencode(
                ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not encoded:
                raise OSError(f"Could not encode frame {frame_index} from {video_path.name}")
            candidates.append((score, frame_index, buffer.tobytes()))
    finally:
        capture.release()

    if not decoded or not candidates:
        raise ValueError(f"Uploaded video contains no decodable frames: {video_path.name}")

    selected = select_frame_candidates(
        candidates, frame_limit, str(config["frame_selection"])
    )

    frames_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    retained_names = set()
    for output_index, (_, _, payload) in enumerate(sorted(selected, key=lambda item: item[1])):
        target = frames_dir / f"{client_id}_{output_index:03d}.jpg"
        atomic_write_bytes(target, payload)
        outputs.append(target)
        retained_names.add(target.name)
    for stale in frames_dir.glob(f"{client_id}_*.jpg"):
        if stale.name not in retained_names:
            stale.unlink()

    return ExtractionResult(outputs, decoded, sampled, passing, truncated)
