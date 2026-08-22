from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import unittest
import uuid


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

    def tearDown(self) -> None:
        self.client.close()
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)

    def test_pages_and_browser_config_are_served(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/viewer").status_code, 200)
        self.assertEqual(self.client.get("/status").status_code, 200)
        config = self.client.get("/api/config").json()
        self.assertEqual(config["capture_seconds"], 15)
        self.assertEqual(config["frames_per_client"], 20)

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
        self.assertEqual(after["job"]["status"], "complete")
        self.assertTrue((self.session_dir / "points.ply").is_file())
        self.assertTrue((self.session_dir / "cameras.json").is_file())
        self.assertTrue((self.session_dir / "meta.json").is_file())


if __name__ == "__main__":
    unittest.main()
