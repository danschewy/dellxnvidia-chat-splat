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

    def test_status_requires_manual_reconstruction(self) -> None:
        source = (ROOT / "static" / "status.html").read_text()
        self.assertIn("Reconstruct", source)
        self.assertIn("method:'POST'", source)


if __name__ == "__main__":
    unittest.main()
