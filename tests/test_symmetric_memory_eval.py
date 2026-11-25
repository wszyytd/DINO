import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class SymmetricMemoryTests(unittest.TestCase):
    def test_original_protocol_and_symmetric_histories(self):
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.memory_retention_eval import evaluate_route as original_route
        from dinoexp.symmetric_memory_eval import (HISTORIES, check_angles, evaluate_route, natural_summary,
                                                 paired_contrasts, symmetric_summary)

        torch.manual_seed(7)
        torch.set_num_threads(1)
        route = dict(id="test", split="val", group="spawn-64", identities=["test:A", "test:B"],
                     samples={}, steps=[])
        cameras = [(0, 1), (2, None), (11, 12), (13, 14), (15, 16), (17, None), (20, 21), (22, 23)]
        x, q, flipped = {}, {}, {}
        def sample(n):
            return torch.randn(n, 256), torch.ones(n, dtype=torch.bool)
        for step, (w, probe) in enumerate(cameras):
            entries = {}
            for uid in route["identities"]:
                for cam in (w, probe):
                    if cam is None:
                        continue
                    key = f"{uid}/{cam:02d}"
                    route["samples"][key] = dict(camera_index=cam, azimuth=15 * cam,
                                                 identity=uid, rgb_sha256=key)
                    x[key], q[key], flipped[key] = sample(16), sample(64), sample(64)
                entries[uid] = dict(sample=f"{uid}/{w:02d}", independent=None if probe is None
                                    else f"{uid}/{probe:02d}", previous_view_gap=None,
                                    probe_previous_view_gap=None)
            route["steps"].append(dict(step=step, writes=entries, frame=str(step)))
        model = EvidenceMatcher().eval()
        with torch.inference_mode():
            original, _ = original_route(model, "memory", route, (x, q), 45, repeat=False, prewrite=True)
            natural, symmetric = evaluate_route(model, route, x, q, flipped)
        check_angles(route, 25, 45)
        self.assertEqual((len(natural), len(symmetric)), (14, 8 * 3 * len(HISTORIES)))
        reference = {(r["probe"], r["kind"]): r["after"] for r in original}
        for r in natural:
            self.assertEqual(r["scores"]["original"]["decision"], reference[r["probe"], r["kind"]]["decision"])
            for key in ("target", "other", "null"):
                self.assertAlmostEqual(r["scores"]["original"][key], reference[r["probe"], r["kind"]][key], places=4)
                self.assertAlmostEqual(r["scores"]["mean"][key],
                                       (r["scores"]["original"][key]+r["scores"]["flip"][key])/2, places=4)
        for r in symmetric:
            self.assertNotIn(r["camera"], HISTORIES[r["history"]])
            if r["history"] in ("F", "R"):
                others = [s for s in symmetric if (s["probe"], s["history"]) == (r["probe"], r["history"])]
                self.assertTrue(all(s["scores"] == r["scores"] for s in others))
        self.assertEqual(natural_summary(natural)["all_independent"]["n"], 12)
        scores = symmetric_summary(symmetric)
        self.assertEqual(scores["FR"]["memory"]["overall"]["n"], 8)
        self.assertEqual(scores["FR"]["memory"]["overall"]["paired_views_n"], 4)
        self.assertEqual(scores["FR"]["last"], scores["R"]["last"])
        self.assertEqual(scores["RF"]["last"], scores["F"]["last"])
        contrasts = paired_contrasts(symmetric)
        last_front = next(c for c in contrasts if c["baseline"] == "R/last" and c["questions"] == "front/retention")
        self.assertEqual((last_front["n"], last_front["helped"], last_front["harmed"]), (4, 0, 0))
        route["samples"]["test:A/23"]["azimuth"] = 60
        with self.assertRaises(ValueError):
            check_angles(route, 25, 45)

    def test_gallery_concatenates_within_each_candidate(self):
        import torch
        from dinoexp.symmetric_memory_eval import gallery_state
        a = dict(slots=torch.randn(1, 2, 8, 4), seen=torch.ones(1, 2, dtype=torch.bool),
                 count=torch.ones(1, 2))
        b = dict(slots=a["slots"] + 10, seen=a["seen"], count=a["count"])
        gallery = gallery_state([a, b])
        torch.testing.assert_close(gallery["slots"][:, :, :8], a["slots"])
        torch.testing.assert_close(gallery["slots"][:, :, 8:], b["slots"])
