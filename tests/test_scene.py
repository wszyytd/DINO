import unittest

from dinoexp.scene import candidate_poses, validate_scene


class SceneTests(unittest.TestCase):
    def test_two_candidates_share_heading_but_have_distinct_positions(self):
        poses = candidate_poses((10, 20, 1, 0, 90, 0), 8)
        self.assertAlmostEqual(poses[0][0], 10)
        self.assertAlmostEqual(poses[0][1], 16)
        self.assertAlmostEqual(poses[1][1], 24)
        self.assertEqual(poses[0][4], poses[1][4])

    def test_identical_appearance_does_not_get_fake_identity_labels(self):
        config = dict(
            target_identity="A",
            candidates=[
                dict(identity="A", blueprint="vehicle.dodge.charger", color="255,255,255"),
                dict(identity="B", blueprint="vehicle.dodge.charger", color="255,255,255"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_scene(config)


if __name__ == "__main__":
    unittest.main()
