from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StaticTests(unittest.TestCase):
    def test_capture_has_required_quality_and_motion_logic(self) -> None:
        source = (ROOT / "static" / "app.js").read_text()
        self.assertIn("DeviceOrientationEvent", source)
        self.assertIn("requestPermission", source)
        self.assertIn("devicemotion", source)
        self.assertIn("TURN PHONE SIDEWAYS", source)
        self.assertIn("Hold the shared landmark", source)
        self.assertIn("Finish on the shared landmark", source)
        self.assertIn("WALK SIDEWAYS", (ROOT / "static" / "index.html").read_text())
        self.assertIn("laplacian", source.lower())
        self.assertIn("image/jpeg", source)
        self.assertIn("MediaRecorder", source)
        self.assertIn("upload_kind: 'video'", source)
        self.assertIn("uploadVideo(recordedVideo, captureId)", source)
        self.assertIn("function sendAndWait", source)
        self.assertIn("waitForSocketOpen", source)
        self.assertIn("Upload timed out", source)
        self.assertIn("selectTemporalFrames", source)
        self.assertIn("resetForAnotherCapture", source)
        self.assertIn("SCAN ANOTHER AREA", (ROOT / "static" / "index.html").read_text())

    def test_viewer_uses_only_local_runtime_modules(self) -> None:
        html = (ROOT / "static" / "viewer.html").read_text()
        source = (ROOT / "static" / "viewer.js").read_text()
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertIn("/static/vendor/three.module.js", html)
        self.assertIn("/static/vendor/spark.module.js", html)
        self.assertTrue((ROOT / "static" / "vendor" / "spark.module.js").is_file())
        self.assertIn("SparkRenderer", source)
        self.assertIn("SplatMesh", source)
        self.assertIn("maxPixelRadius", source)
        self.assertIn("nonLod: true", source)
        self.assertIn("roomscanBoundingSphere", source)
        self.assertIn("splat.ply", source)
        self.assertIn("points.ply", source)
        self.assertIn("cameras.json", source)
        self.assertIn("lerpVectors", source)
        self.assertIn("runtimeConfig.point_size", source)
        self.assertIn("Math.max(0, (now - flightStart)", source)
        self.assertIn("Math.atan(principalY / focalY)", source)
        self.assertIn("estimateRecoveredUp", source)
        self.assertIn("leveledCameraPose", source)
        self.assertIn("camera.up.copy(recoveredUp)", source)
        self.assertIn("runtimeConfig.splat_max_screen_size", source)
        self.assertIn("runtimeConfig.splat_exposure", source)
        self.assertIn("pollForModelUpdate", source)
        self.assertIn("status.model_version", source)
        self.assertIn("status.model_path", source)
        self.assertIn("Updating shared reconstruction", source)

    def test_viewer_maps_spark_three_addon_imports_offline(self) -> None:
        html = (ROOT / "static" / "viewer.html").read_text()
        match = re.search(r'<script type="importmap">(.*?)</script>', html, re.DOTALL)
        self.assertIsNotNone(match)
        imports = json.loads(match.group(1))["imports"]
        self.assertEqual(imports["three/addons/"], "/static/vendor/three-addons/")
        self.assertTrue(
            (ROOT / "static" / "vendor" / "three-addons" / "postprocessing" / "Pass.js").is_file()
        )

    def test_status_requires_manual_reconstruction(self) -> None:
        source = (ROOT / "static" / "status.html").read_text()
        self.assertIn("Rebuild now", source)
        self.assertIn("method:'POST'", source)
        self.assertIn("Video queue", source)
        self.assertIn("processing_videos", source)
        self.assertIn("Live updates", source)

    def test_lobby_can_create_join_and_preview_rooms(self) -> None:
        html = (ROOT / "static" / "lobby.html").read_text()
        source = (ROOT / "static" / "lobby.js").read_text()
        self.assertIn("New room scan", html)
        self.assertIn("Existing rooms", html)
        self.assertIn("room.thumbnail_url", source)
        self.assertIn("fetch('/api/sessions'", source)
        self.assertIn("method: 'POST'", source)
        self.assertIn("searchParams.get('session')", source)


if __name__ == "__main__":
    unittest.main()
