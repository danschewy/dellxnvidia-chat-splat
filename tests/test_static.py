from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StaticTests(unittest.TestCase):
    def test_capture_has_required_quality_and_motion_logic(self) -> None:
        source = (ROOT / "static" / "app.js").read_text()
        self.assertIn("DeviceOrientationEvent", source)
        self.assertIn("requestPermission", source)
        self.assertIn("devicemotion", source)
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
        self.assertIn("splat.ply", source)
        self.assertIn("points.ply", source)
        self.assertIn("cameras.json", source)
        self.assertIn("lerpVectors", source)
        self.assertIn("splatScale", source)
        self.assertIn("splatOpacity", source)
        self.assertIn("runtimeConfig.point_size", source)
        self.assertIn("vertexColors: false", source)
        self.assertIn("attribute vec3 splatColor", source)
        self.assertIn("Math.max(0, (now - flightStart)", source)
        self.assertIn("Math.atan(principalY / focalY)", source)
        self.assertIn("function projectedSplatScale()", source)
        self.assertIn("renderer.domElement.height", source)
        self.assertIn("runtimeConfig.splat_max_screen_size", source)
        self.assertIn("runtimeConfig.splat_exposure", source)
        self.assertIn("pollForModelUpdate", source)
        self.assertIn("status.model_version", source)
        self.assertIn("status.model_path", source)
        self.assertIn("Updating shared reconstruction", source)
        self.assertRegex(source, r"fragmentShader: `[\s\S]*uniform float exposure;")

    def test_status_requires_manual_reconstruction(self) -> None:
        source = (ROOT / "static" / "status.html").read_text()
        self.assertIn("Rebuild now", source)
        self.assertIn("method:'POST'", source)
        self.assertIn("Video queue", source)
        self.assertIn("processing_videos", source)
        self.assertIn("Live updates", source)


if __name__ == "__main__":
    unittest.main()
