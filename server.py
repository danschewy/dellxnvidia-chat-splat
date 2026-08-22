#!/usr/bin/env python3
"""Single-process HTTPS capture and reconstruction server for ROOMSCAN."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

from backends import select_backend
from reconstruct import run_pipeline
from roomscan_io import atomic_copy_file, atomic_write_bytes, atomic_write_json, load_config
from video_ingest import extract_sharp_frames, video_extension


ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "session"
STATIC = ROOT / "static"
SAMPLE = ROOT / "sample_data"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
config = load_config()
startup_backend = select_backend(config, ROOT)
clients: dict[str, dict[str, dict[str, Any]]] = {}
jobs: dict[str, dict[str, Any]] = {}
video_jobs: dict[str, dict[str, dict[str, Any]]] = {}
background_tasks: set[asyncio.Task[Any]] = set()
live_update_tasks: dict[str, asyncio.Task[Any]] = {}
live_update_state: dict[str, dict[str, Any]] = {}
video_decode_semaphore = asyncio.Semaphore(int(config["video_worker_count"]))
state_lock = threading.Lock()
inference_lock = threading.Lock()
inference_queue_limit = int(config["inference_queue_limit"])
if inference_queue_limit < 1:
    raise ValueError("inference_queue_limit must be positive")


class ReconstructionQualityHeld(RuntimeError):
    """An automatic revision was kept from replacing a known-good model."""


try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:
    raise SystemExit("FastAPI is missing. Install requirements-mac.txt or requirements-cuda.txt") from exc


app = FastAPI(title="ROOMSCAN", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/sample_data", StaticFiles(directory=SAMPLE), name="sample-data")
SESSIONS.mkdir(parents=True, exist_ok=True)
app.mount("/session", StaticFiles(directory=SESSIONS), name="sessions")


def checked_id(value: str, label: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, underscore, or dash")
    return value


@app.get("/")
def home_page(request: Request) -> FileResponse:
    page = "index.html" if request.query_params.get("session") else "lobby.html"
    return FileResponse(STATIC / page)


@app.get("/viewer")
def viewer_page() -> FileResponse:
    return FileResponse(STATIC / "viewer.html")


@app.get("/status")
def status_page() -> FileResponse:
    return FileResponse(STATIC / "status.html")


@app.get("/api/config")
def browser_config() -> JSONResponse:
    exposed = {
        key: config[key]
        for key in (
            "blur_threshold", "frames_per_client", "capture_seconds", "capture_fps",
            "frame_selection",
            "jpeg_quality", "capture_width", "motion_rotation_threshold",
            "motion_translation_threshold", "point_size",
            "video_upload", "video_bits_per_second", "max_video_upload_bytes",
            "upload_timeout_seconds",
            "live_updates", "viewer_refresh_seconds",
            "splat_max_screen_size",
            "splat_exposure",
        )
    }
    return JSONResponse(exposed, headers={"Cache-Control": "no-store"})


def session_snapshot(session_id: str) -> dict[str, Any]:
    frames_dir = SESSIONS / session_id / "frames"
    frame_names = [path.name for path in frames_dir.glob("*.jpg")] if frames_dir.is_dir() else []
    per_client: dict[str, int] = {}
    for name in frame_names:
        client_id = name.rsplit("_", 1)[0]
        per_client[client_id] = per_client.get(client_id, 0) + 1
    with state_lock:
        live = clients.get(session_id, {}).copy()
        job = jobs.get(session_id, {"status": "idle"}).copy()
        queued_videos = {
            key: value.copy() for key, value in video_jobs.get(session_id, {}).items()
        }
        update_state = live_update_state.get(session_id, {}).copy()
    session_dir = SESSIONS / session_id
    meta_path = session_dir / "meta.json"
    current_path = session_dir / "current.json"
    current = {}
    if current_path.is_file():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
    all_client_ids = sorted(set(per_client) | set(live) | set(queued_videos))
    client_rows = []
    for client_id in all_client_ids:
        video_state = queued_videos.get(client_id, {})
        state = str(video_state.get("status") or ("connected" if client_id in live else "uploaded"))
        client_rows.append(
            {
                "client_id": client_id,
                "frame_count": per_client.get(
                    client_id,
                    int(video_state.get("frame_count", live.get(client_id, {}).get("frame_count", 0))),
                ),
                "connected": client_id in live,
                "state": state,
                "error": video_state.get("error"),
            }
        )
    return {
        "session_id": session_id,
        "backend": startup_backend.name,
        "frame_count": len(frame_names),
        "clients": client_rows,
        "connected_clients": len(live),
        "processing_videos": sum(
            1 for value in queued_videos.values()
            if value.get("status") in {"uploading", "queued", "processing"}
        ),
        "job": job,
        "viewer_ready": (SESSIONS / session_id / "points.ply").is_file(),
        "model_version": str(
            current.get("version")
            or (meta_path.stat().st_mtime_ns if meta_path.is_file() else "")
        ),
        "model_path": str(current.get("path", "")),
        "live_updates": bool(config["live_updates"]),
        "update_pending": bool(update_state.get("dirty")),
    }


def session_card(session_id: str) -> dict[str, Any]:
    session_dir = SESSIONS / session_id
    snapshot = session_snapshot(session_id)
    frames = list((session_dir / "frames").glob("*.jpg"))
    thumbnail = max(frames, key=lambda path: path.stat().st_mtime_ns) if frames else None
    room = {}
    room_path = session_dir / "room.json"
    if room_path.is_file():
        try:
            room = json.loads(room_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            room = {}
    artifacts = [path for path in (room_path, session_dir / "meta.json", thumbnail) if path]
    updated_ns = max(
        (path.stat().st_mtime_ns for path in artifacts if path.is_file()),
        default=session_dir.stat().st_mtime_ns,
    )
    return {
        "session_id": session_id,
        "title": str(room.get("title") or session_id),
        "created_at": room.get("created_at"),
        "updated_at": datetime.fromtimestamp(
            updated_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat(),
        "thumbnail_url": (
            f"/session/{session_id}/frames/{thumbnail.name}?v={thumbnail.stat().st_mtime_ns}"
            if thumbnail else None
        ),
        "frame_count": snapshot["frame_count"],
        "client_count": len(snapshot["clients"]),
        "connected_clients": snapshot["connected_clients"],
        "processing_videos": snapshot["processing_videos"],
        "viewer_ready": snapshot["viewer_ready"],
        "job_status": snapshot["job"].get("status", "idle"),
        "update_pending": snapshot["update_pending"],
        "capture_url": f"/?session={session_id}",
        "viewer_url": f"/viewer?session={session_id}",
        "status_url": f"/status?session={session_id}",
    }


@app.get("/api/sessions")
def list_sessions() -> JSONResponse:
    limit = int(config["session_list_limit"])
    if limit < 1:
        raise HTTPException(500, "session_list_limit must be positive")
    cards = []
    for directory in SESSIONS.iterdir():
        if not directory.is_dir() or not SAFE_ID.fullmatch(directory.name):
            continue
        try:
            cards.append(session_card(directory.name))
        except OSError as exc:
            print(f"[sessions] skipped {directory}: {exc}", flush=True)
    cards.sort(key=lambda card: card["updated_at"], reverse=True)
    return JSONResponse(
        {"sessions": cards[:limit], "backend": startup_backend.name},
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/sessions")
def create_session() -> JSONResponse:
    now = datetime.now(timezone.utc)
    for _attempt in range(10):
        session_id = f"room-{now:%Y%m%d}-{secrets.token_hex(3)}"
        session_dir = SESSIONS / session_id
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise HTTPException(503, "Could not allocate a unique room code")
    atomic_write_json(
        session_dir / "room.json",
        {
            "session_id": session_id,
            "title": f"Room {session_id.rsplit('-', 1)[-1].upper()}",
            "created_at": now.isoformat(),
        },
    )
    return JSONResponse(session_card(session_id), status_code=201)


def process_video_upload(session_id: str, client_id: str, video_path: Path) -> bool:
    started = time.perf_counter()
    print(f"[stage:video_ingest] started session={session_id} client={client_id}", flush=True)
    with state_lock:
        video_jobs[session_id][client_id] = {"status": "processing", "frame_count": 0}
    try:
        result = extract_sharp_frames(
            video_path, SESSIONS / session_id / "frames", client_id, config
        )
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        with state_lock:
            video_jobs[session_id][client_id] = {
                "status": "failed",
                "frame_count": 0,
                "seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(f"[stage:video_ingest] failed in {elapsed:.3f}s: {exc}", flush=True)
        return False
    else:
        elapsed = time.perf_counter() - started
        with state_lock:
            video_jobs[session_id][client_id] = {
                "status": "ready",
                "frame_count": len(result.frames),
                "decoded_frames": result.decoded_frames,
                "sampled_frames": result.sampled_frames,
                "passing_frames": result.passing_frames,
                "truncated": result.truncated,
                "seconds": elapsed,
            }
        print(
            f"[stage:video_ingest] ok in {elapsed:.3f}s: "
            f"{result.decoded_frames} decoded, {len(result.frames)} selected",
            flush=True,
        )
        return True


def mark_session_dirty(session_id: str) -> None:
    """Debounce completed captures while guaranteeing a bounded rebuild delay."""
    if not bool(config["live_updates"]):
        return
    now = time.monotonic()
    with state_lock:
        state = live_update_state.setdefault(session_id, {})
        if not state.get("dirty"):
            state["first_dirty"] = now
        state["dirty"] = True
        state["last_dirty"] = now
        state["generation"] = int(state.get("generation", 0)) + 1
    task = live_update_tasks.get(session_id)
    if task is None or task.done():
        task = asyncio.create_task(live_update_loop(session_id))
        live_update_tasks[session_id] = task
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)


async def live_update_loop(session_id: str) -> None:
    debounce = float(config["live_update_debounce_seconds"])
    max_wait = float(config["live_update_max_wait_seconds"])
    while True:
        with state_lock:
            state = live_update_state.setdefault(session_id, {})
            if not state.get("dirty"):
                return
            deadline = min(
                float(state["last_dirty"]) + debounce,
                float(state["first_dirty"]) + max_wait,
            )
            reconstruction_active = jobs.get(session_id, {}).get("status") in {
                "queued", "running"
            }
            queue_full = sum(
                job.get("status") in {"queued", "running"}
                for job in jobs.values()
            ) >= inference_queue_limit
            snapshot_generation = int(state.get("generation", 0))
        if reconstruction_active or queue_full:
            await asyncio.sleep(0.5)
            continue
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(min(remaining, 0.5))
            continue
        frames = list((SESSIONS / session_id / "frames").glob("*.jpg"))
        if not frames:
            with state_lock:
                state = live_update_state.setdefault(session_id, {})
                if int(state.get("generation", 0)) == snapshot_generation:
                    state["dirty"] = False
                    return
            continue
        with state_lock:
            state = live_update_state.setdefault(session_id, {})
            if not state.get("dirty"):
                return
            if jobs.get(session_id, {}).get("status") in {"queued", "running"}:
                continue
            active_jobs = sum(
                job.get("status") in {"queued", "running"}
                for job in jobs.values()
            )
            if active_jobs >= inference_queue_limit:
                continue
            # New clips arriving after this point remain dirty and schedule one
            # follow-up pass while the current reconstruction is processed.
            state["dirty"] = int(state.get("generation", 0)) != snapshot_generation
            jobs[session_id] = {
                "status": "queued",
                "trigger": "live_update",
                "queued_at": datetime.now(timezone.utc).isoformat(),
            }
        print(
            f"[live-update] rebuilding session={session_id} from {len(frames)} frames",
            flush=True,
        )
        await asyncio.to_thread(
            reconstruct_job, session_id, bool(config["live_update_train_splat"])
        )


async def process_queued_video(session_id: str, client_id: str, video_path: Path) -> None:
    async with video_decode_semaphore:
        if await asyncio.to_thread(process_video_upload, session_id, client_id, video_path):
            mark_session_dirty(session_id)


@app.get("/api/session/{session_id}/status")
def session_status(session_id: str) -> JSONResponse:
    try:
        checked_id(session_id, "session_id")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(session_snapshot(session_id), headers={"Cache-Control": "no-store"})


def _prune_model_revisions(session_dir: Path, active_version: str) -> None:
    retention = int(config["model_revision_retention"])
    if retention < 2:
        raise ValueError("model_revision_retention must be at least 2")
    models_dir = session_dir / "models"
    revisions = sorted(
        (
            path for path in models_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
        reverse=True,
    ) if models_dir.is_dir() else []
    keep = {active_version}
    for revision in revisions:
        if len(keep) >= retention:
            break
        keep.add(revision.name)
    for revision in revisions:
        if revision.name not in keep:
            try:
                shutil.rmtree(revision)
            except OSError as exc:
                print(f"[revision-prune] could not remove {revision}: {exc}", flush=True)


def reconstruct_job(session_id: str, train_splat: bool = True) -> None:
    with state_lock:
        current = jobs.get(session_id, {})
        trigger = current.get("trigger", "direct")
        queued_at = current.get("queued_at", datetime.now(timezone.utc).isoformat())
        jobs[session_id] = {
            "status": "queued",
            "trigger": trigger,
            "queued_at": queued_at,
        }
    with inference_lock:
        with state_lock:
            jobs[session_id] = {
                "status": "running",
                "trigger": trigger,
                "queued_at": queued_at,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        _reconstruct_job_locked(session_id, train_splat, trigger)


def _reconstruct_job_locked(session_id: str, train_splat: bool, trigger: str) -> None:
    session_dir = SESSIONS / session_id
    published_geometry = False

    def publish_revision(build_dir: Path, include_splat: bool) -> None:
        nonlocal published_geometry
        build_meta = json.loads((build_dir / "meta.json").read_text(encoding="utf-8"))
        build_quality = build_meta.get("quality", {})
        build_warnings = build_quality.get(
            "blocking_warnings", build_quality.get("warnings", [])
        )
        current_path = session_dir / "current.json"
        previous_meta_path = session_dir / "meta.json"
        if (
            trigger == "live_update"
            and current_path.is_file()
            and previous_meta_path.is_file()
            and build_warnings
        ):
            try:
                previous_meta = json.loads(previous_meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous_meta = {}
            previous_quality = previous_meta.get("quality")
            previous_warnings = (previous_quality or {}).get(
                "blocking_warnings", (previous_quality or {}).get("warnings", [])
            )
            if previous_quality is not None and not previous_warnings:
                raise ReconstructionQualityHeld(
                    "Automatic update held; reconstruction quality checks failed. "
                    "The last good shared model remains active."
                )
        version = str(time.time_ns())
        relative_path = f"models/{version}/"
        revision_dir = session_dir / relative_path
        revision_dir.mkdir(parents=True, exist_ok=False)
        atomic_copy_file(build_dir / "cameras.json", revision_dir / "cameras.json")
        atomic_copy_file(build_dir / "points.ply", revision_dir / "points.ply")
        atomic_copy_file(build_dir / "meta.json", revision_dir / "meta.json")
        if include_splat:
            atomic_copy_file(build_dir / "splat.ply", revision_dir / "splat.ply")

        # Preserve the root-level file contract for offline tools.
        atomic_copy_file(build_dir / "cameras.json", session_dir / "cameras.json")
        atomic_copy_file(build_dir / "points.ply", session_dir / "points.ply")
        if include_splat:
            atomic_copy_file(build_dir / "splat.ply", session_dir / "splat.ply")
        else:
            (session_dir / "splat.ply").unlink(missing_ok=True)
        atomic_copy_file(build_dir / "meta.json", session_dir / "meta.json")
        atomic_write_json(
            session_dir / "current.json",
            {"version": version, "path": relative_path},
        )
        _prune_model_revisions(session_dir, version)
        published_geometry = True

    try:
        with tempfile.TemporaryDirectory(prefix=".build-", dir=session_dir) as build_name:
            build_dir = Path(build_name)
            meta = run_pipeline(
                session_dir / "frames",
                build_dir,
                geometry_ready=(
                    (lambda ready_dir: publish_revision(ready_dir, False))
                    if train_splat else None
                ),
                train_splat=train_splat,
                backend=startup_backend,
            )
            built_splat = build_dir / "splat.ply"
            if built_splat.is_file():
                publish_revision(build_dir, True)
            elif not published_geometry:
                publish_revision(build_dir, False)
    except ReconstructionQualityHeld as exc:
        with state_lock:
            jobs[session_id] = {
                "status": "held",
                "trigger": trigger,
                "error": str(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        print(f"[quality-gate] session={session_id}: {exc}", flush=True)
    except BaseException as exc:
        with state_lock:
            jobs[session_id] = {
                "status": "failed",
                "trigger": trigger,
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
    else:
        with state_lock:
            jobs[session_id] = {
                "status": "complete",
                "trigger": trigger,
                "backend": meta["backend"],
                "seconds": meta["total_seconds"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }


@app.post("/api/session/{session_id}/reconstruct")
async def start_reconstruction(session_id: str) -> JSONResponse:
    try:
        checked_id(session_id, "session_id")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    frames = list((SESSIONS / session_id / "frames").glob("*.jpg"))
    if not frames:
        raise HTTPException(400, "No uploaded frames in this session")
    with state_lock:
        current = jobs.get(session_id, {})
        if current.get("status") in {"queued", "running"}:
            raise HTTPException(409, "Reconstruction is already running")
        active_jobs = sum(
            job.get("status") in {"queued", "running"}
            for job in jobs.values()
        )
        if active_jobs >= inference_queue_limit:
            raise HTTPException(429, "Inference queue is full; try again shortly")
        active_videos = video_jobs.get(session_id, {}).values()
        if any(
            item.get("status") in {"uploading", "queued", "processing"}
            for item in active_videos
        ):
            raise HTTPException(409, "Video uploads are still being processed")
        jobs[session_id] = {
            "status": "queued",
            "trigger": "manual",
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        live_update_state.setdefault(session_id, {})["dirty"] = False
    asyncio.create_task(asyncio.to_thread(reconstruct_job, session_id))
    return JSONResponse({"accepted": True, "session_id": session_id}, status_code=202)


@app.websocket("/ws/upload")
async def upload(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = ""
    client_id = ""
    frame_index = 0
    upload_kind = "frames"
    video_path: Path | None = None
    declared_frame_count: int | None = None
    upload_timeout = float(config["upload_timeout_seconds"])
    try:
        if upload_timeout <= 0:
            raise ValueError("upload_timeout_seconds must be positive")
        hello = json.loads(
            await asyncio.wait_for(websocket.receive_text(), timeout=upload_timeout)
        )
        if hello.get("type") != "start":
            raise ValueError("first WebSocket message must be type=start")
        session_id = checked_id(str(hello.get("session_id", "")), "session_id")
        client_id = checked_id(str(hello.get("client_id", "")), "client_id")
        upload_kind = str(hello.get("upload_kind", "frames"))
        if upload_kind not in {"frames", "video"}:
            raise ValueError("upload_kind must be frames or video")
        if upload_kind == "video" and not bool(config["video_upload"]):
            raise ValueError("Video uploads are disabled in config")
        if upload_kind == "frames" and hello.get("frame_count") is not None:
            declared_frame_count = int(hello["frame_count"])
            if not 1 <= declared_frame_count <= int(config["frames_per_client"]):
                raise ValueError("frame_count exceeds frames_per_client")
        frames_dir = SESSIONS / session_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if upload_kind == "video":
            suffix = video_extension(str(hello.get("mime_type", "")))
            video_path = SESSIONS / session_id / "uploads" / f"{client_id}{suffix}"
        with state_lock:
            existing_video = video_jobs.get(session_id, {}).get(client_id, {})
            if upload_kind == "video" and existing_video.get("status") in {
                "uploading", "queued", "processing"
            }:
                raise ValueError("This client's previous video is still being processed")
            clients.setdefault(session_id, {})[client_id] = {
                "connected_at": datetime.now(timezone.utc).isoformat(),
                "frame_count": 0,
                "upload_kind": upload_kind,
            }
            if upload_kind == "video":
                video_jobs.setdefault(session_id, {})[client_id] = {
                    "status": "uploading", "frame_count": 0
                }
        await websocket.send_json({"type": "ready", "client_id": client_id})
        while True:
            packet = await asyncio.wait_for(
                websocket.receive(), timeout=upload_timeout
            )
            if packet.get("bytes") is not None:
                payload = packet["bytes"]
                if upload_kind == "video":
                    if video_path is None:
                        raise RuntimeError("Video upload path was not initialized")
                    if frame_index:
                        raise ValueError("Video upload accepts exactly one binary message")
                    if len(payload) > int(config["max_video_upload_bytes"]):
                        raise ValueError("Video exceeds max_video_upload_bytes")
                    atomic_write_bytes(video_path, payload)
                    frame_index = 1
                    await websocket.send_json({"type": "ack", "bytes": len(payload)})
                else:
                    if frame_index >= int(config["frames_per_client"]):
                        raise ValueError("JPEG upload exceeds frames_per_client")
                    if len(payload) > int(config["max_upload_bytes"]):
                        raise ValueError("JPEG exceeds max_upload_bytes")
                    if not payload.startswith(b"\xff\xd8"):
                        raise ValueError("binary upload is not a JPEG")
                    target = frames_dir / f"{client_id}_{frame_index:03d}.jpg"
                    atomic_write_bytes(target, payload)
                    frame_index += 1
                    with state_lock:
                        clients[session_id][client_id]["frame_count"] = frame_index
                    await websocket.send_json({"type": "ack", "frame": frame_index})
            elif packet.get("text") is not None:
                command = json.loads(packet["text"])
                if command.get("type") == "complete":
                    if upload_kind == "video":
                        if video_path is None or not video_path.is_file():
                            raise ValueError("Video upload completed without video bytes")
                        with state_lock:
                            video_jobs.setdefault(session_id, {})[client_id] = {
                                "status": "queued", "frame_count": 0
                            }
                        task = asyncio.create_task(
                            process_queued_video(session_id, client_id, video_path)
                        )
                        background_tasks.add(task)
                        task.add_done_callback(background_tasks.discard)
                        await websocket.send_json(
                            {"type": "complete", "queued": True, "frames": 0}
                        )
                    else:
                        if (
                            declared_frame_count is not None
                            and frame_index != declared_frame_count
                        ):
                            raise ValueError(
                                f"Expected {declared_frame_count} JPEGs, received {frame_index}"
                            )
                        mark_session_dirty(session_id)
                        await websocket.send_json({"type": "complete", "frames": frame_index})
                    await websocket.close(code=1000)
                    break
            elif packet.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        with state_lock:
            current_video = video_jobs.get(session_id, {}).get(client_id, {})
            if current_video.get("status") == "uploading":
                current_video["status"] = "failed"
                current_video["error"] = "Upload connection closed before completion"
    except Exception as exc:
        with state_lock:
            current_video = video_jobs.get(session_id, {}).get(client_id, {})
            if current_video.get("status") == "uploading":
                current_video["status"] = "failed"
                current_video["error"] = str(exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1008)
        except Exception:
            pass
    finally:
        if session_id and client_id:
            with state_lock:
                current_video = video_jobs.get(session_id, {}).get(client_id, {})
                if current_video.get("status") == "uploading":
                    current_video["status"] = "failed"
                    current_video["error"] = "Upload connection closed before completion"
                clients.get(session_id, {}).pop(client_id, None)


def development_certificate() -> tuple[Path, Path]:
    directory = ROOT / ".certs"
    certificate, key = directory / "dev-cert.pem", directory / "dev-key.pem"
    if certificate.is_file() and key.is_file():
        return certificate, key
    directory.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(certificate), "-days", "3",
                "-subj", "/CN=roomscan.local",
                "-addext", "subjectAltName=DNS:roomscan.local,DNS:localhost,IP:127.0.0.1",
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not create a self-signed cert; install openssl or set ROOMSCAN_CERT_FILE and ROOMSCAN_KEY_FILE") from exc
    return certificate, key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--self-signed", action="store_true")
    arguments = parser.parse_args()
    cert_value = os.getenv("ROOMSCAN_CERT_FILE")
    key_value = os.getenv("ROOMSCAN_KEY_FILE")
    if arguments.self_signed:
        certificate, key = development_certificate()
    elif cert_value and key_value:
        certificate, key = Path(cert_value), Path(key_value)
    else:
        parser.error("HTTPS is required: set ROOMSCAN_CERT_FILE and ROOMSCAN_KEY_FILE or pass --self-signed")
    import uvicorn

    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        ssl_certfile=str(certificate),
        ssl_keyfile=str(key),
        ws_max_size=int(config["max_video_upload_bytes"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
