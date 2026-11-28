import importlib.util
import unittest

if not importlib.util.find_spec("torch"):
    raise unittest.SkipTest("requires PyTorch")

import torch

from dinoexp.experiment8_view_routing import CAMERAS, evaluate_routes, pool_view_key, report, route_logits, summarize


class ViewRoutingTests(unittest.TestCase):
    def test_key_means_the_last_layer_without_padding(self):
        raw = torch.tensor([[[99., 99.], [99., 99.], [99., 99.]],
                            [[3., 0.], [0., 4.], [100., 100.]]])
        valid = torch.tensor([True, True, False])
        torch.testing.assert_close(pool_view_key(raw, valid), torch.tensor([.6, .8]))

    def test_both_candidates_share_one_view_and_dropped_view_has_no_weight(self):
        cos = torch.tensor([[[.99, .01], [.70, .70], [.05, .05], [.3, .3]]])
        scores = torch.tensor([[[20., 0., -.1], [3., 7., -.1], [1., 2., -.1], [2., 1., -.1]]])
        states = route_logits(cos, scores, temperature=.1)
        full = states["full"]
        self.assertEqual(full["picked"].tolist(), [1])
        torch.testing.assert_close(full["hard"], scores[:, 1])
        scaled = (cos.mean(-1)/.1).softmax(dim=1)
        torch.testing.assert_close(full["soft"], (scaled[..., None]*scores).sum(1))
        removed = states["drop_20"]
        self.assertEqual(removed["picked"].tolist(), [0])
        self.assertEqual(removed["weights"][0, 1].item(), 0)
        torch.testing.assert_close(removed["hard"], scores[:, 0])
        torch.testing.assert_close(removed["soft"][:, 2], scores[:, 0, 2])
        with self.assertRaisesRegex(ValueError, "temperature"):
            route_logits(cos, scores, temperature=0.)

    def test_held_out_report_keeps_selected_views_and_identity_results_separate(self):
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.gallery_reader import cache_route
        from test_gallery_reader import fixture

        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = EvidenceMatcher(dim=12, heads=3, slots=8).eval()
        route, pairs, obs, probes = fixture("val")
        cache = cache_route(model, route, pairs, obs, probes)
        one_hot = torch.eye(len(CAMERAS))
        writes = {p["write"]: one_hot[p["position"]-1] for p in pairs}
        queries = {p["query"]: one_hot[p["position"] % len(CAMERAS)] for p in pairs}
        rows = evaluate_routes(model, {route["id"]: (cache, writes, queries)})
        counts = summarize(rows)[route["id"]]
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(counts["view_hits"].values()), 0)
        for row, p in zip(rows, pairs, strict=True):
            selected = CAMERAS[p["position"] % len(CAMERAS)]
            self.assertEqual(row["routing"]["full"]["picked_camera"], selected)
            self.assertEqual(row["routing"]["full"]["hard"]["logits"], row["conditions"][f"single_{selected:02}"]["logits"])
            self.assertEqual(row["routing"][f"drop_{selected:02}"]["weights"][f"{selected:02}"], 0)
            self.assertEqual(row["oracle"], row["conditions"][f"single_{p['write_camera']:02}"])
        self.assertIn("drop_12 soft", report({route["id"]: counts}, .05))
