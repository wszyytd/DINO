import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dinoexp.memory_replay import build_schedule, partition_routes
from dinoexp.memory_retention import open_crop
from dinoexp.region_ab_data import sha


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

    def test_postwrite_includes_first_query_and_never_becomes_a_write(self):
        first, second, third = build_schedule(route())
        self.assertEqual(first["postwrite"], [("qa", 0)])
        self.assertCountEqual(second["postwrite"], [("qa", 0), ("qb", 1)])
        self.assertEqual(third["postwrite"], [])
        self.assertNotIn(("qa", 0), third["replay"])

    def test_same_question_comparison_requires_both_candidates_before_the_frame(self):
        from dinoexp.memory_replay import paired_independent_scores

        def row(uid, pre, post, both=True):
            return dict(kind="new_view", identity="scene:" + uid,
                        before_frame=dict(decision=pre, margin=1.0 if both else None),
                        after=dict(decision=post, margin=1.0))

        rows = [row("A", "other", "target"), row("B", "target", "null"),
                row("B", "target", "target"), row("A", "target", "target", False)]
        metrics = paired_independent_scores(rows)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual((metrics["pre_correct"], metrics["post_correct"]), (2, 2))
        self.assertEqual((metrics["help"], metrics["harm"]), (1, 1))
        self.assertEqual(metrics["by_candidate"]["B"]["total"], 2)
        self.assertEqual(metrics["balanced_post_accuracy"], .75)

    def test_foreground_crop_preserves_only_same_instance_pixels_and_checks_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.new("RGB", (5, 3), (21, 45, 72))
            image.putpixel((2, 1), (234, 201, 17))
            image.save(root / "rgb.png")
            mask = Image.new("L", image.size)
            mask.putpixel((2, 1), 255)
            mask.save(root / "mask.png")
            sample = dict(image="rgb.png", rgb_sha256=sha(root / "rgb.png"),
                          size=list(image.size), bbox=[1, 1, 4, 3],
                          mask=dict(path="mask.png", sha256=sha(root / "mask.png")))
            bbox = open_crop(root, sample)
            cutout = open_crop(root, sample, foreground=True)
            self.assertEqual(bbox.size, cutout.size)
            self.assertEqual(cutout.getpixel((1, 0)), (234, 201, 17))
            self.assertEqual(bbox.getpixel((0, 0)), (21, 45, 72))
            self.assertEqual(cutout.getpixel((0, 0)), (0, 0, 0))
            Image.new("L", mask.size).save(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "mask"):
                open_crop(root, sample, foreground=True)
            sample["mask"]["sha256"] = sha(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "empty"):
                open_crop(root, sample, foreground=True)

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
