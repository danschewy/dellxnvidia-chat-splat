from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from roomscan_io import load_config
from video_ingest import extract_sharp_frames, select_frame_candidates, video_extension


ROOT = Path(__file__).resolve().parents[1]


class VideoIngestTests(unittest.TestCase):
    def test_temporal_selection_keeps_every_part_of_capture(self) -> None:
        candidates = [
            (1000.0 if index < 4 else float(index), index, bytes([index]))
            for index in range(12)
        ]
        selected = select_frame_candidates(candidates, 4, "temporal_sharpness")
        self.assertEqual([item[1] for item in selected], [0, 3, 8, 11])

    def test_video_mime_type_maps_to_safe_extension(self) -> None:
        self.assertEqual(video_extension("video/mp4;codecs=h264"), ".mp4")
        self.assertEqual(video_extension("video/webm; codecs=vp8"), ".webm")
        with self.assertRaises(ValueError):
            video_extension("application/octet-stream")

    def test_opencv_video_is_decoded_ranked_and_written_as_jpegs(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy are only required by video ingestion")

        config = load_config(ROOT / "config.json")
        config.update({"capture_width": 160, "capture_fps": 2, "frames_per_client": 4})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            video_path = directory / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(16):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :] = (index * 11, index * 7, index * 3)
                cv2.line(frame, (0, index * 3 % 48), (63, (index * 5 + 9) % 48), (255, 255, 255), 2)
                writer.write(frame)
            writer.release()

            result = extract_sharp_frames(
                video_path, directory / "frames", "phone_video", config
            )

            self.assertEqual(len(result.frames), 4)
            self.assertEqual(result.decoded_frames, 16)
            self.assertEqual(result.sampled_frames, 8)
            self.assertEqual([path.name for path in result.frames], [
                "phone_video_000.jpg", "phone_video_001.jpg",
                "phone_video_002.jpg", "phone_video_003.jpg",
            ])
            for path in result.frames:
                image = cv2.imread(str(path))
                self.assertIsNotNone(image)
                self.assertEqual(image.shape[1], 160)


if __name__ == "__main__":
    unittest.main()
