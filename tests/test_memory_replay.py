import unittest

from dinoexp.memory_replay import build_schedule, partition_routes


def route():
    samples = {}
    for name, uid in (("a1","A"), ("a2","A"), ("a3","A"), ("b1","B"),
                      ("qa","A"), ("qb","B")):
        samples[name] = dict(identity=uid, rgb_sha256=name)
    return dict(id="example", group="bg1", split="train", identities=["A","B"], samples=samples,
        steps=[
            dict(step=0, writes={"A": dict(sample="a1", independent="qa")}),
            dict(step=1, writes={"B": dict(sample="b1", independent="qb"),
                                "A": dict(sample="a2", independent="qa")}),
            dict(step=2, writes={}),
            dict(step=3, writes={"A": dict(sample="a3", independent=None)}),
        ])


class MemoryReplayTests(unittest.TestCase):
    def test_replay_contains_every_written_crop_at_every_real_write(self):
        first, second, third = build_schedule(route())
        self.assertEqual([first["step"], second["step"], third["step"]], [0, 1, 3])
        self.assertEqual(first["replay"], [("a1", 0)])
        self.assertCountEqual(second["replay"], [("a1", 0), ("a2", 0), ("b1", 1)])
        self.assertCountEqual(third["replay"], [("a1", 0), ("a2", 0), ("a3", 0), ("b1", 1)])
        self.assertEqual(first["prewrite"], [])
        self.assertEqual(second["prewrite"], [("qa", 0)])
        self.assertEqual(second["fixed_labels"], [0, 1])
        self.assertEqual(first["fixed_labels"], [0, 2])

    def test_reject_independent_query_with_future_observation_rgb(self):
        example = route()
        example["samples"]["qa"]["rgb_sha256"] = "a3"
        with self.assertRaisesRegex(ValueError, "independent query overlaps"):
            build_schedule(example)

    def test_background_and_rgb_cannot_cross_split(self):
        first = route()
        second = dict(first, id="other", split="val")
        with self.assertRaisesRegex(ValueError, "background"):
            partition_routes([first, second])
        second["group"] = "bg2"
        with self.assertRaisesRegex(ValueError, "RGB"):
            partition_routes([first, second])
        second["samples"] = {k: dict(v, rgb_sha256=f"val-{k}")
                             for k, v in second["samples"].items()}
        train, val = partition_routes([first, second])
        self.assertEqual([r["id"] for r in train], ["example"])
        self.assertEqual([r["id"] for r in val], ["other"])
