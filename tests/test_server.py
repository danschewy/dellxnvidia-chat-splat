from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
import unittest
import uuid
from unittest import mock

from video_ingest import ExtractionResult


os.environ["ROOMSCAN_BACKEND"] = "stub"

try:
    from fastapi.testclient import TestClient
    import server
except ImportError:
    TestClient = None
    server = None


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = f"test_{uuid.uuid4().hex[:10]}"
        self.session_dir = server.SESSIONS / self.session_id
        self.client = TestClient(server.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)

    def test_pages_and_browser_config_are_served(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/viewer").status_code, 200)
        self.assertEqual(self.client.get("/status").status_code, 200)
        config = self.client.get("/api/config").json()
        self.assertEqual(config["capture_seconds"], 15)
        self.assertEqual(config["frames_per_client"], 20)
        self.assertEqual(config["splat_max_screen_size"], 12.0)
        self.assertEqual(config["splat_exposure"], 1.8)
        self.assertTrue(config["video_upload"])
        self.assertEqual(config["video_bits_per_second"], 3_000_000)
        self.assertEqual(config["upload_timeout_seconds"], 30)
        self.assertTrue(config["live_updates"])
        self.assertEqual(config["viewer_refresh_seconds"], 2)
        self.assertEqual(server.config["video_worker_count"], 2)
        self.assertEqual(server.config["inference_queue_limit"], 4)
        self.assertEqual(server.config["model_revision_retention"], 3)

    def test_binary_websocket_upload_lands_in_contract(self) -> None:
        jpeg = (server.SAMPLE / "frames" / "frame_001.jpg").read_bytes()
        with self.client.websocket_connect("/ws/upload") as socket:
            socket.send_text(json.dumps({"type": "start", "session_id": self.session_id, "client_id": "phone_a"}))
            self.assertEqual(socket.receive_json()["type"], "ready")
            socket.send_bytes(jpeg)
            self.assertEqual(socket.receive_json(), {"type": "ack", "frame": 1})
            socket.send_text(json.dumps({"type": "complete"}))
            self.assertEqual(socket.receive_json(), {"type": "complete", "frames": 1})
        frame = self.session_dir / "frames" / "phone_a_000.jpg"
        self.assertEqual(frame.read_bytes(), jpeg)
        status = self.client.get(f"/api/session/{self.session_id}/status").json()
        self.assertEqual(status["frame_count"], 1)
        self.assertEqual(status["clients"][0]["client_id"], "phone_a")
        self.assertEqual(status["job"]["status"], "idle")

    def test_jpeg_fallback_rejects_declared_count_over_hard_limit(self) -> None:
        with self.client.websocket_connect("/ws/upload") as socket:
            socket.send_text(json.dumps({
                "type": "start",
                "session_id": self.session_id,
                "client_id": "phone_overflow",
                "upload_kind": "frames",
                "frame_count": server.config["frames_per_client"] + 1,
            }))
            error = socket.receive_json()
        self.assertEqual(error["type"], "error")
        self.assertIn("frames_per_client", error["message"])

    def test_video_upload_is_persisted_and_queued_for_frame_extraction(self) -> None:
        payload = b"fake-phone-video"

        def fake_extract(video_path, frames_dir, client_id, config):
            self.assertEqual(video_path.read_bytes(), payload)
            frames_dir.mkdir(parents=True, exist_ok=True)
            target = frames_dir / f"{client_id}_000.jpg"
            shutil.copy2(server.SAMPLE / "frames" / "frame_001.jpg", target)
            return ExtractionResult([target], 30, 4, 3)

        live_timing = {
            "live_update_debounce_seconds": 0.01,
            "live_update_max_wait_seconds": 0.03,
        }
        with mock.patch.dict(server.config, live_timing), mock.patch.object(
            server, "extract_sharp_frames", side_effect=fake_extract
        ):
            with self.client.websocket_connect("/ws/upload") as socket:
                socket.send_text(json.dumps({
                    "type": "start",
                    "session_id": self.session_id,
                    "client_id": "phone_video",
                    "upload_kind": "video",
                    "mime_type": "video/mp4;codecs=h264",
                }))
                self.assertEqual(socket.receive_json()["type"], "ready")
                shutil.copy2(
                    server.SAMPLE / "frames" / "frame_002.jpg",
                    self.session_dir / "frames" / "existing_000.jpg",
                )
                status = self.client.get(
                    f"/api/session/{self.session_id}/status"
                ).json()
                self.assertEqual(status["processing_videos"], 1)
                blocked = self.client.post(
                    f"/api/session/{self.session_id}/reconstruct"
                )
                self.assertEqual(blocked.status_code, 409)
                socket.send_bytes(payload)
                self.assertEqual(socket.receive_json(), {"type": "ack", "bytes": len(payload)})
                socket.send_text(json.dumps({"type": "complete"}))
                self.assertEqual(
                    socket.receive_json(), {"type": "complete", "queued": True, "frames": 0}
                )

            status = {}
            for _ in range(100):
                status = self.client.get(f"/api/session/{self.session_id}/status").json()
                phone = next(
                    client for client in status["clients"]
                    if client["client_id"] == "phone_video"
                )
                if phone["state"] == "ready":
                    break
                time.sleep(0.01)
            self.assertEqual(phone["state"], "ready")
            self.assertEqual(status["frame_count"], 2)
            self.assertEqual(status["processing_videos"], 0)
            self.assertEqual(
                (self.session_dir / "uploads" / "phone_video.mp4").read_bytes(), payload
            )
            for _ in range(200):
                status = self.client.get(f"/api/session/{self.session_id}/status").json()
                if status["viewer_ready"] and status["job"]["status"] == "complete":
                    break
                time.sleep(0.01)
            self.assertTrue(status["viewer_ready"])
            self.assertEqual(status["job"]["trigger"], "live_update")
            self.assertTrue(status["model_version"])

    def test_reconstruction_is_manual_and_produces_viewer_artifacts(self) -> None:
        frames = self.session_dir / "frames"
        frames.mkdir(parents=True)
        source = server.SAMPLE / "frames" / "frame_001.jpg"
        shutil.copy2(source, frames / "phone_b_000.jpg")
        before = self.client.get(f"/api/session/{self.session_id}/status").json()
        self.assertFalse(before["viewer_ready"])
        server.reconstruct_job(self.session_id)
        after = self.client.get(f"/api/session/{self.session_id}/status").json()
        self.assertTrue(after["viewer_ready"])
        self.assertTrue(after["model_version"])
        self.assertEqual(after["job"]["status"], "complete")
        self.assertTrue((self.session_dir / "points.ply").is_file())
        self.assertTrue((self.session_dir / "cameras.json").is_file())
        self.assertTrue((self.session_dir / "meta.json").is_file())
        current = json.loads((self.session_dir / "current.json").read_text())
        revision = self.session_dir / current["path"]
        self.assertTrue((revision / "points.ply").is_file())
        self.assertTrue((revision / "cameras.json").is_file())

    def test_failed_rebuild_preserves_last_published_model(self) -> None:
        self.session_dir.mkdir(parents=True)
        previous = {
            "points.ply": b"last-good-points",
            "cameras.json": b"last-good-cameras",
            "meta.json": b"last-good-meta",
        }
        for name, payload in previous.items():
            (self.session_dir / name).write_bytes(payload)
        with mock.patch.object(server, "run_pipeline", side_effect=RuntimeError("boom")):
            server.reconstruct_job(self.session_id)
        for name, payload in previous.items():
            self.assertEqual((self.session_dir / name).read_bytes(), payload)
        self.assertEqual(server.jobs[self.session_id]["status"], "failed")

    def test_bad_live_rebuild_is_held_and_preserves_last_good_model(self) -> None:
        self.session_dir.mkdir(parents=True)
        previous = {
            "points.ply": b"last-good-points",
            "cameras.json": b"last-good-cameras",
        }
        for name, payload in previous.items():
            (self.session_dir / name).write_bytes(payload)
        (self.session_dir / "meta.json").write_text(json.dumps({
            "quality": {"warnings": []},
        }))
        (self.session_dir / "current.json").write_text(json.dumps({
            "version": "good", "path": "models/good/",
        }))
        server.jobs[self.session_id] = {"status": "running", "trigger": "live_update"}

        def bad_pipeline(image_dir, out_dir, **kwargs):
            (out_dir / "points.ply").write_bytes(b"bad-points")
            (out_dir / "cameras.json").write_bytes(b"bad-cameras")
            (out_dir / "meta.json").write_text(json.dumps({
                "backend": "stub",
                "total_seconds": 1.0,
                "quality": {"warnings": ["camera_path_discontinuity"]},
            }))
            return {"backend": "stub", "total_seconds": 1.0}

        with mock.patch.object(server, "run_pipeline", side_effect=bad_pipeline):
            server.reconstruct_job(self.session_id, train_splat=False)

        for name, payload in previous.items():
            self.assertEqual((self.session_dir / name).read_bytes(), payload)
        self.assertEqual(server.jobs[self.session_id]["status"], "held")
        self.assertEqual(json.loads((self.session_dir / "current.json").read_text())["version"], "good")

    def test_revision_pruning_keeps_active_and_rollback_window(self) -> None:
        models = self.session_dir / "models"
        for version in ("1", "2", "3", "4", "5"):
            (models / version).mkdir(parents=True)
        with mock.patch.dict(server.config, {"model_revision_retention": 3}):
            server._prune_model_revisions(self.session_dir, "5")
        self.assertEqual(
            sorted(path.name for path in models.iterdir()),
            ["3", "4", "5"],
        )

    def test_manual_rebuild_rejects_when_global_inference_queue_is_full(self) -> None:
        frames = self.session_dir / "frames"
        frames.mkdir(parents=True)
        shutil.copy2(
            server.SAMPLE / "frames" / "frame_001.jpg",
            frames / "phone_000.jpg",
        )
        queued_ids = [f"queued_{index}" for index in range(server.inference_queue_limit)]
        try:
            with server.state_lock:
                for queued_id in queued_ids:
                    server.jobs[queued_id] = {"status": "queued"}
            response = self.client.post(f"/api/session/{self.session_id}/reconstruct")
        finally:
            with server.state_lock:
                for queued_id in queued_ids:
                    server.jobs.pop(queued_id, None)
        self.assertEqual(response.status_code, 429)


if __name__ == "__main__":
    unittest.main()
