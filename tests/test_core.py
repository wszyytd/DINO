import unittest

import numpy as np
from PIL import Image

from dinoexp.core import (
    camera_pose,
    crop_bounds,
    project_box,
    pose_matches,
    select_frame,
    cosine_scores,
    prepare_image,
    local_box,
    checked_path,
)


class CoreTests(unittest.TestCase):
    def test_orbit_rotates_with_vehicle_heading(self):
        pose = camera_pose((10, 20, 2), 90, 0, 0, 10)
        np.testing.assert_allclose(pose[:3], [10, 30, 2], atol=1e-8)
        self.assertAlmostEqual(abs(pose[4]), 90)

    def test_elevated_camera_looks_down_at_target(self):
        pose = camera_pose((0, 0, 0), 0, 0, 30, 10)
        self.assertAlmostEqual(pose[2], 5)
        self.assertAlmostEqual(pose[3], -30)

    def test_project_uses_ue_camera_axes(self):
        vertices = [[10, -1, -1], [10, 1, 1]]
        np.testing.assert_allclose(
            project_box(vertices, np.eye(4), 100, 100, 90), [45, 45, 55, 55], atol=1e-8
        )

    def test_projection_rejects_points_behind_camera(self):
        with self.assertRaises(ValueError):
            project_box([[-1, 0, 0]], np.eye(4), 100, 100, 90)

    def test_clipped_vehicle_is_not_silently_cropped(self):
        with self.assertRaises(ValueError):
            crop_bounds((-5, 10, 30, 40), 100, 100)
        self.assertEqual(crop_bounds((10, 20, 30, 40), 100, 100, padding=0), (10, 20, 30, 40))

    def test_pose_match_handles_angle_wrap(self):
        self.assertTrue(pose_matches((0, 0, 0, 0, 179.9, 0), (0, 0, 0, 0, -180.1, 0)))
        self.assertFalse(pose_matches((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)))

    def test_frame_selection_rejects_stale_or_wrong_pose(self):
        p = (0, 0, 0, 0, 0, 0)
        frames = [(8, p, "stale"), (10, (1, 0, 0, 0, 0, 0), "wrong"), (10, p, "ok")]
        self.assertEqual(select_frame(frames, 10, p), "ok")
        self.assertIsNone(select_frame(frames[:2], 10, p))

    def test_cosine_values_and_zero_feature_rejection(self):
        np.testing.assert_allclose(cosine_scores([1, 0], [[2, 0], [0, 3], [-1, 0]]), [1, 0, -1])
        with self.assertRaises(ValueError):
            cosine_scores([0, 0], [[1, 0]])

    def test_letterbox_keeps_both_ends_of_vehicle(self):
        im = Image.new("RGB", (200, 50), "red")
        prepared = prepare_image(im, 224)
        self.assertEqual(prepared.size, (224, 224))
        self.assertEqual(prepared.getpixel((0, 112)), (255, 0, 0))
        self.assertEqual(prepared.getpixel((223, 112)), (255, 0, 0))

    def test_local_region_requires_visible_same_part(self):
        ref = {"part": "side", "visible": True, "box": [0, 0, 1, 1]}
        query = {"part": "side", "visible": False, "box": [0, 0, 1, 1]}
        self.assertIsNone(local_box(query, ref, (100, 50)))
        query["visible"] = True
        query["part"] = "plate"
        self.assertIsNone(local_box(query, ref, (100, 50)))
        query["part"] = "side"
        self.assertEqual(local_box(query, ref, (100, 50)), (0, 0, 100, 50))

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            checked_path("data", "../outside.png")


if __name__ == "__main__":
    unittest.main()
