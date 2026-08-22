from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from backends import ReconstructionResult, select_backend
from backends.stub import StubBackend
from backends.vggt_backend import _invert_batched_se3
import reconstruct
from roomscan_io import evenly_subsample, load_config, read_points_ply
from splat_trainer import _background_shape


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config.json")

    def test_required_config_knobs_exist(self) -> None:
        required = {
            "max_frames", "vggt_resolution", "confidence_threshold", "blur_threshold",
            "frames_per_client", "mask_people", "gsplat_iterations", "point_size",
            "splat_max_screen_size", "splat_exposure", "weights_dir", "backend_override",
        }
        self.assertFalse(required - self.config.keys())
        self.assertEqual(self.config["vggt_resolution"], 518)
        self.assertEqual(self.config["max_frames"], 32)
        self.assertIn(self.config["geometry_model"], {"vggt", "pi3"})

    def test_explicit_stub_selection_requires_no_ml_dependencies(self) -> None:
        with mock.patch.dict(os.environ, {"ROOMSCAN_BACKEND": "stub"}):
            backend = select_backend(self.config, ROOT)
        self.assertIsInstance(backend, StubBackend)

    def test_backend_segmentation_contract_returns_masks_from_images_only(self) -> None:
        parameters = list(inspect.signature(StubBackend.segment_people).parameters)
        self.assertEqual(parameters, ["self", "images"])
        backend = StubBackend(self.config, ROOT)
        self.assertEqual(backend.segment_people([Path("a.jpg"), Path("b.jpg")]), [None, None])

    def test_fixture_contract_is_complete(self) -> None:
        frames = sorted((ROOT / "sample_data" / "frames").glob("*.jpg"))
        points, colors = read_points_ply(ROOT / "sample_data" / "points.ply")
        cameras = json.loads((ROOT / "sample_data" / "cameras.json").read_text())
        self.assertEqual(len(frames), 25)
        self.assertGreater(len(points), 20_000)
        self.assertEqual(len(points), len(colors))
        self.assertEqual(len(cameras), 25)
        for camera in cameras:
            self.assertEqual((len(camera["T_wc"]), len(camera["T_wc"][0])), (4, 4))
            self.assertEqual((len(camera["K"]), len(camera["K"][0])), (3, 3))
            self.assertIn("confidence", camera)

    def test_even_subsample_keeps_endpoints(self) -> None:
        self.assertEqual(evenly_subsample(list(range(10)), 4), [0, 3, 6, 9])

    def test_vggt_pose_inverse_flattens_and_restores_batch_axes(self) -> None:
        class ShapeOnlyTensor:
            def __init__(self, shape):
                self.shape = tuple(shape)

            def reshape(self, shape):
                shape = tuple(shape)
                if shape[0] == -1:
                    flattened = 1
                    for size in self.shape[:-2]:
                        flattened *= size
                    shape = (flattened,) + shape[1:]
                return ShapeOnlyTensor(shape)

        extrinsic = ShapeOnlyTensor((1, 2, 3, 4))
        received_shapes = []

        def inverse(flattened):
            received_shapes.append(flattened.shape)
            return ShapeOnlyTensor((flattened.shape[0], 4, 4))

        inverted = _invert_batched_se3(extrinsic, inverse)

        self.assertEqual(received_shapes, [(2, 3, 4)])
        self.assertEqual(inverted.shape, (1, 2, 4, 4))

    def test_gsplat_background_shape_matches_packing_mode(self) -> None:
        self.assertEqual(_background_shape(True, camera_count=1), (3,))
        self.assertEqual(_background_shape(False, camera_count=2), (2, 3))

    def test_person_masks_remove_corresponding_dense_geometry(self) -> None:
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("NumPy/OpenCV are only required by the real ML backends")

        result = ReconstructionResult(
            points=np.arange(24, dtype=np.float32).reshape(8, 3),
            colors=np.arange(24, dtype=np.uint8).reshape(8, 3),
            confidences=np.ones(8, dtype=np.float32),
            cameras=[],
            point_layout=(2, 2, 2),
        )
        masks = [
            np.array([[False, True], [False, False]]),
            np.array([[True, False], [False, False]]),
        ]

        filtered, removed = reconstruct._exclude_masked_points(result, masks)

        self.assertEqual(removed, 2)
        self.assertEqual(len(filtered.points), 6)
        self.assertIsNone(filtered.point_layout)

    def test_stub_pipeline_writes_full_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "scene"
            with mock.patch.dict(os.environ, {"ROOMSCAN_BACKEND": "stub"}):
                meta = reconstruct.run_pipeline(ROOT / "sample_data" / "frames", out)
            self.assertEqual(meta["backend"], "stub")
            for relative in ("frames", "points.ply", "cameras.json", "meta.json"):
                self.assertTrue((out / relative).exists(), relative)
            self.assertFalse((out / "splat.ply").exists())

    def test_late_splat_failure_preserves_reconstruction(self) -> None:
        class FailingSplatStub(StubBackend):
            name = "stub"

            def train_splat(self, poses, points, images):
                raise RuntimeError("simulated CUDA extension failure")

        backend = FailingSplatStub(self.config, ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "scene"
            with mock.patch.object(reconstruct, "select_backend", return_value=backend):
                meta = reconstruct.run_pipeline(ROOT / "sample_data" / "frames", out)
            self.assertTrue((out / "points.ply").is_file())
            self.assertTrue((out / "cameras.json").is_file())
            self.assertEqual(meta["stages"]["train_splat"]["status"], "failed")

    def test_oom_retries_with_fewer_frames(self) -> None:
        class OomThenSuccess:
            name = "fake"

            def __init__(self):
                self.counts = []

            def reconstruct(self, images):
                self.counts.append(len(images))
                if len(images) > 4:
                    raise RuntimeError("CUDA out of memory")
                return ReconstructionResult([], [], [], [])

        backend = OomThenSuccess()
        result, used = reconstruct._reconstruct_with_backoff(backend, [Path(str(i)) for i in range(16)])
        self.assertIsInstance(result, ReconstructionResult)
        self.assertEqual(backend.counts, [16, 8, 4])
        self.assertEqual(len(used), 4)


if __name__ == "__main__":
    unittest.main()
