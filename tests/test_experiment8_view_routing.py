import importlib.util
import unittest

if not importlib.util.find_spec("torch"):
    raise unittest.SkipTest("requires PyTorch")

import torch
from PIL import Image

from dinoexp.experiment8_view_routing import (
    CAMERAS, POOLING, evaluate_routes, pool_view_key, pooling_report, report, route_logits, summarize,
)
from dinoexp.routing_foreground import foreground_keys, foreground_patch_weights, spatial_cosines


class ViewRoutingTests(unittest.TestCase):
    def test_foreground_weights_follow_letterbox_and_do_not_use_black_pixels_as_mask(self):
        mask = Image.new("L", (28, 56))
        mask.paste(255, (0, 0, 14, 28))
        weights = foreground_patch_weights(mask, 56)
        active = torch.tensor(weights.sum(0)).nonzero().flatten().tolist()
        self.assertEqual(active, [1, 5])  # Four patch columns, one-column padding on the left.
        raw = torch.full((16, 2), 1000.)
        raw[1], raw[5] = torch.tensor([1., 0.]), torch.tensor([0., 1.])
        keys = foreground_keys(raw, torch.tensor(weights))
        self.assertEqual(keys["foreground_grid"].shape, (16, 2))
        torch.testing.assert_close(keys["foreground_mean"], torch.tensor([2.**-.5, 2.**-.5]))
        torch.testing.assert_close(keys["foreground_grid"][0], torch.tensor([1., 0.]))
        torch.testing.assert_close(keys["foreground_grid"][12], torch.tensor([0., 1.]))

    def test_spatial_cosine_uses_only_corresponding_occupied_cells(self):
        probe = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]])
        bank = torch.tensor([[[1., 0.], [0., 0.], [1., 0.]],
                             [[0., 1.], [1., 0.], [0., 0.]],
                             [[0., 0.], [0., 0.], [1., 0.]]])
        result = spatial_cosines(probe, bank, torch.tensor([[.1, .2, .3]]))
        torch.testing.assert_close(result, torch.tensor([[1., 0., .3]]))

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

    def test_pooling_comparison_changes_selection_and_preserves_single_view_scores(self):
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.gallery_reader import cache_route
        from test_gallery_reader import fixture

        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = EvidenceMatcher(dim=12, heads=3, slots=8).eval()
        route, pairs, obs, probes = fixture("val")
        cache = cache_route(model, route, pairs, obs, probes)
        axes = torch.eye(len(CAMERAS))

        def keys(index, foreground_index):
            foreground = axes[foreground_index]
            return dict(global_mean=axes[index], foreground_mean=foreground, foreground_grid=foreground.repeat(16, 1))

        writes = {p["write"]: keys(p["position"] - 1, p["position"] - 1) for p in pairs}
        queries = {p["query"]: keys(p["position"] % len(CAMERAS), p["position"] - 1) for p in pairs}
        results = evaluate_routes(model, {route["id"]: (cache, writes, queries)}, compare_pooling=True)
        summaries = {method: summarize(rows) for method, rows in results.items()}
        self.assertEqual(tuple(results), POOLING)
        self.assertEqual([sum(s[route["id"]]["view_hits"].values()) for s in summaries.values()], [0, 8, 8])
        first = results["global_mean"]
        for method in POOLING[1:]:
            self.assertEqual([r["conditions"] for r in first], [r["conditions"] for r in results[method]])
        self.assertIn("foreground_grid/report.md", pooling_report(summaries, results))
