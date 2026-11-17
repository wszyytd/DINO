import unittest

from dinoexp.memory_retention import choose_view_probe, score_pair, summarize


class MemoryRetentionTests(unittest.TestCase):
    def test_probe_uses_independent_photo_and_camera_position_across_angle_wrap(self):
        def sample(point, angle, digest):
            return dict(point=point, azimuth=angle, rgb_sha256=digest)
        anchor = sample("p0", 179, "observed")
        pool = {
            "same_point": sample("p0", 179, "other"),
            "same_rgb": sample("p1", 179, "observed"),
            "route_rgb": sample("p2", 179, "future"),
            "far": sample("p3", 140, "far"),
            "valid": sample("p4", -178, "valid"),
        }
        self.assertEqual(choose_view_probe(anchor, pool, {"observed", "future"}, 25), "valid")

    def test_unseen_car_masks_margin_and_does_not_look_like_an_update_win(self):
        before = score_pair([-1e4, 1, .3], [False, True], 0)
        after = score_pair([3, 1, .3], [True, True], 0)
        result = summarize([dict(split="val", kind="new_view", step=0, anchor_step=0,
                                 before=before, after=after, probe="photo", identity="A", route="r")])
        self.assertIsNone(before["margin"])
        self.assertEqual(after["margin"], 2)
        self.assertEqual(result["val"]["new_view"]["paired"], 0)
        self.assertEqual(result["val"]["new_view"]["help"], 0)

    def test_margin_catches_rising_confuser_and_same_identity_different_null(self):
        before = score_pair([2, 1, 1.5], [True, True], 0)
        after = score_pair([3, 4, 5], [True, True], 0)
        row = dict(split="val", kind="old_exact", step=2, anchor_step=0,
                   before=before, after=after, probe="photo", identity="A", route="r")
        outcome = summarize([row])["val"]["old_exact"]
        self.assertEqual(after["decision"], "null")
        self.assertEqual(outcome["harm"], 1)
        self.assertEqual(outcome["mean_margin_delta"], -2)
