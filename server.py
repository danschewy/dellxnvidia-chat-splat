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
import subprocess
import threading
from typing import Any

from backends import select_backend
from reconstruct import run_pipeline
from roomscan_io import atomic_write_bytes, load_config


ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "session"
STATIC = ROOT / "static"
SAMPLE = ROOT / "sample_data"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
config = load_config()
startup_backend = select_backend(config, ROOT)
clients: dict[str, dict[str, dict[str, Any]]] = {}
jobs: dict[str, dict[str, Any]] = {}
state_lock = threading.Lock()


try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
def capture_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


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
            "jpeg_quality", "capture_width", "motion_rotation_threshold",
            "motion_translation_threshold", "point_size",
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
    all_client_ids = sorted(set(per_client) | set(live))
    return {
        "session_id": session_id,
        "backend": startup_backend.name,
        "frame_count": len(frame_names),
        "clients": [
            {
                "client_id": client_id,
                "frame_count": count,
                "connected": client_id in live,
            }
            for client_id in all_client_ids
            for count in [per_client.get(client_id, int(live.get(client_id, {}).get("frame_count", 0)))]
        ],
        "connected_clients": len(live),
        "job": job,
        "viewer_ready": (SESSIONS / session_id / "points.ply").is_file(),
    }


@app.get("/api/session/{session_id}/status")
def session_status(session_id: str) -> JSONResponse:
    try:
        checked_id(session_id, "session_id")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(session_snapshot(session_id), headers={"Cache-Control": "no-store"})


def reconstruct_job(session_id: str) -> None:
    session_dir = SESSIONS / session_id
    try:
        meta = run_pipeline(session_dir / "frames", session_dir)
    except BaseException as exc:
        with state_lock:
            jobs[session_id] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
    else:
        with state_lock:
            jobs[session_id] = {
                "status": "complete",
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
        if current.get("status") == "running":
            raise HTTPException(409, "Reconstruction is already running")
        jobs[session_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    asyncio.create_task(asyncio.to_thread(reconstruct_job, session_id))
    return JSONResponse({"accepted": True, "session_id": session_id}, status_code=202)


@app.websocket("/ws/upload")
async def upload(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = ""
    client_id = ""
    frame_index = 0
    try:
        hello = json.loads(await websocket.receive_text())
        if hello.get("type") != "start":
            raise ValueError("first WebSocket message must be type=start")
        session_id = checked_id(str(hello.get("session_id", "")), "session_id")
        client_id = checked_id(str(hello.get("client_id", "")), "client_id")
        frames_dir = SESSIONS / session_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        with state_lock:
            clients.setdefault(session_id, {})[client_id] = {
                "connected_at": datetime.now(timezone.utc).isoformat(), "frame_count": 0
            }
        await websocket.send_json({"type": "ready", "client_id": client_id})
        while True:
            packet = await websocket.receive()
            if packet.get("bytes") is not None:
                payload = packet["bytes"]
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
                    await websocket.send_json({"type": "complete", "frames": frame_index})
                    await websocket.close(code=1000)
                    break
            elif packet.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1008)
        except Exception:
            pass
    finally:
        if session_id and client_id:
            with state_lock:
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

    uvicorn.run(app, host=arguments.host, port=arguments.port, ssl_certfile=str(certificate), ssl_keyfile=str(key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
