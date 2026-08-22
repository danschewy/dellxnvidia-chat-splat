from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from backends import ReconstructionResult, select_backend
from backends.stub import StubBackend
from backends.vggt_backend import _invert_batched_se3
import reconstruct
import models
from roomscan_io import evenly_subsample, load_config, read_points_ply
from multiphone import (
    SimilarityTransform,
    apply_similarity,
    estimate_similarity_ransac,
    frame_identity,
    group_images_by_device,
    select_capture_aware,
    transform_reconstruction,
)
from splat_trainer import _background_shape


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config.json")

    def test_required_config_knobs_exist(self) -> None:
        required = {
            "max_frames", "vggt_resolution", "confidence_threshold", "blur_threshold",
            "multi_device_submaps", "submap_max_frames", "submaps_per_device",
            "submap_frames_per_capture",
            "alignment_max_images_per_submap", "alignment_feature_count",
            "alignment_ratio_test", "alignment_min_pair_matches",
            "alignment_max_image_pairs", "alignment_min_image_pairs",
            "alignment_ransac_threshold_ratio", "alignment_ransac_iterations",
            "alignment_min_inliers", "alignment_min_inlier_ratio",
            "alignment_scale_min", "alignment_scale_max",
            "vggt_preprocess_mode", "vggt_portrait_height",
            "frames_per_client", "mask_people", "gsplat_iterations", "point_size",
            "frame_selection",
            "video_upload", "video_bits_per_second", "max_video_upload_bytes",
            "max_video_decode_seconds", "max_video_source_fps", "upload_timeout_seconds",
            "video_worker_count", "live_update_train_splat",
            "inference_queue_limit",
            "quality_max_camera_step_ratio", "quality_min_camera_steps",
            "model_revision_retention",
            "session_list_limit",
            "live_updates", "live_update_debounce_seconds",
            "live_update_max_wait_seconds", "viewer_refresh_seconds",
            "max_point_cloud_points", "splat_max_screen_size", "splat_exposure",
            "weights_dir", "backend_override",
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

    def test_capture_identity_survives_pipeline_copy_prefixes(self) -> None:
        identity = frame_identity(Path("012_9df8632037c9-mt4wp3mb_007.jpg"))
        self.assertEqual(identity.device_id, "9df8632037c9")
        self.assertEqual(identity.capture_id, "9df8632037c9-mt4wp3mb")
        self.assertEqual(identity.frame_index, 7)

    def test_capture_aware_sampling_keeps_local_motion_sequences(self) -> None:
        images = [
            Path(f"phone_a-cap{capture}_{frame:03d}.jpg")
            for capture in range(8)
            for frame in range(20)
        ]
        selected = select_capture_aware(images, limit=16, frames_per_capture=4)
        captures = {}
        for image in selected:
            identity = frame_identity(image)
            captures.setdefault(identity.capture_id, []).append(identity.frame_index)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(captures), 4)
        self.assertTrue(all(len(frames) == 4 for frames in captures.values()))
        self.assertEqual(set(group_images_by_device(selected)), {"phone_a"})

    def test_similarity_ransac_recovers_scale_rotation_and_translation(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required for alignment tests")
        generator = np.random.default_rng(7)
        source = generator.normal(size=(80, 3))
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected = SimilarityTransform(1.7, rotation, np.array([3.0, -2.0, 0.5]))
        target = apply_similarity(source, expected)
        target[:12] = generator.normal(size=(12, 3)) * 20

        estimated = estimate_similarity_ransac(
            source,
            target,
            threshold=0.02,
            iterations=512,
            minimum_inliers=20,
            minimum_inlier_ratio=0.5,
            scale_min=0.2,
            scale_max=5.0,
        )
        self.assertIsNotNone(estimated)
        transform, inliers, _median_error = estimated
        self.assertGreaterEqual(int(inliers.sum()), 68)
        self.assertAlmostEqual(transform.scale, expected.scale, places=5)
        self.assertTrue(np.allclose(transform.rotation, rotation, atol=1e-5))
        self.assertTrue(np.allclose(transform.translation, expected.translation, atol=1e-5))

    def test_similarity_transforms_points_and_camera_poses(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required for alignment tests")
        result = ReconstructionResult(
            points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            colors=np.array([[1, 2, 3]], dtype=np.uint8),
            confidences=np.array([1.0], dtype=np.float32),
            cameras=[{
                "frame": "phone-cap_000.jpg",
                "T_wc": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "confidence": 1.0,
            }],
        )
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transformed = transform_reconstruction(
            result, SimilarityTransform(2.0, rotation, np.array([4.0, 5.0, 6.0]))
        )
        self.assertTrue(np.allclose(transformed.points, [[4.0, 7.0, 6.0]]))
        camera = np.asarray(transformed.cameras[0]["T_wc"])
        self.assertTrue(np.allclose(camera[:3, :3], rotation))
        self.assertTrue(np.allclose(camera[:3, 3], [4.0, 7.0, 6.0]))

    def test_vggt_preprocessing_handles_portrait_once_at_native_resolution(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is only required by the real ML backends")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "portrait.jpg"
            portrait_height = int(self.config["vggt_portrait_height"])
            Image.new("RGB", (720, 1280), (40, 80, 120)).save(source)
            crop = reconstruct._resize_for_vggt(
                [source], root / "crop", 518, "crop", portrait_height=portrait_height
            )[0]
            pad = reconstruct._resize_for_vggt(
                [source], root / "pad", 518, "pad", portrait_height=portrait_height
            )[0]
            with Image.open(crop) as crop_image, Image.open(pad) as pad_image:
                self.assertEqual(crop_image.size, (518, portrait_height))
                self.assertEqual(pad_image.size, (518, 518))

    def test_prepared_batch_does_not_crop_tall_portrait_again(self) -> None:
        try:
            import torch  # noqa: F401
            from PIL import Image
        except ImportError:
            self.skipTest("PyTorch/Pillow are only required by the real ML backends")
        from image_batch import load_prepared_image_batch

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "portrait.jpg"
            Image.new("RGB", (518, 728), (40, 80, 120)).save(source)
            batch = load_prepared_image_batch([source])
        self.assertEqual(tuple(batch.shape), (1, 3, 728, 518))

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

    def test_camera_quality_is_grouped_by_capture_and_detects_jumps(self) -> None:
        def camera(frame, x):
            return {
                "frame": frame,
                "T_wc": [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }

        cameras = [camera(f"phone_a_{index:03d}.jpg", index * 0.1) for index in range(7)]
        cameras += [camera(f"phone_b_{index:03d}.jpg", 50 + index * 0.1) for index in range(7)]
        grouped = reconstruct._camera_path_quality(cameras, 8.0, 6)
        self.assertTrue(grouped["continuous"])
        self.assertEqual(grouped["evaluated_capture_count"], 2)

        cameras[-1] = camera("phone_b_006.jpg", 70)
        jumped = reconstruct._camera_path_quality(cameras, 8.0, 6)
        self.assertFalse(jumped["continuous"])
        self.assertEqual(jumped["worst_capture"], "phone_b")

    def test_gsplat_background_shape_matches_packing_mode(self) -> None:
        self.assertEqual(_background_shape(True, camera_count=1), (3,))
        self.assertEqual(_background_shape(False, camera_count=2), (2, 3))

    def test_splat_ply_is_binary_little_endian_for_browser_runtimes(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is only required by the real ML backends")
        from splat_trainer import _encode_splat_ply

        payload = _encode_splat_ply(
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            np.array([0.4], dtype=np.float32),
            np.array([[-1.0, -2.0, -3.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        header, body = payload.split(b"end_header\n", 1)
        self.assertIn(b"format binary_little_endian 1.0", header)
        self.assertEqual(len(body), 17 * 4)
        values = struct.unpack("<17f", body)
        self.assertEqual(values[:6], (1.0, 2.0, 3.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(values[9], 0.4, places=6)

    def test_virtualenv_bin_is_added_without_resolving_python_symlink(self) -> None:
        with mock.patch.object(models.sys, "executable", "/opt/roomscan/.venv/bin/python"), mock.patch.dict(
            os.environ, {"PATH": "/usr/local/bin:/usr/bin"}
        ):
            models._ensure_interpreter_bin_on_path()
            self.assertEqual(
                os.environ["PATH"].split(os.pathsep)[0],
                "/opt/roomscan/.venv/bin",
            )

    def test_confidence_filter_spreads_selection_across_a_tied_vggt_floor(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is only required by the real ML backends")
        result = ReconstructionResult(
            points=np.arange(18, dtype=np.float32).reshape(6, 3),
            colors=np.zeros((6, 3), dtype=np.uint8),
            confidences=np.array([1, 1, 1, 1, 1.1, 1.2], dtype=np.float32),
            cameras=[],
        )

        filtered = reconstruct._filter_points(result, 0.5, max_points=3)
        all_equal = reconstruct._filter_points(
            ReconstructionResult(result.points, result.colors, np.ones(6), []),
            0.5,
        )
        capped_equal = reconstruct._filter_points(
            ReconstructionResult(result.points, result.colors, np.ones(6), []),
            0.5,
            max_points=4,
        )

        self.assertEqual(len(filtered.points), 3)
        self.assertTrue(np.allclose(filtered.confidences[-2:], [1.1, 1.2]))
        self.assertEqual(len(all_equal.points), 6)
        self.assertEqual(len(capped_equal.points), 4)

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
