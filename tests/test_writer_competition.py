import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class WriterCompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_two_normalizations_and_padded_inputs(self):
        import torch
        from dinoexp.competitive_writer import CompetitiveWriter

        torch.manual_seed(7)
        writer = CompetitiveWriter(8, 2).to(self.device)
        q = torch.randn(2, 3, 8, device=self.device)
        x = torch.randn(2, 5, 8, device=self.device)
        valid = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0]], dtype=torch.bool, device=self.device)
        assignment, weights, _ = writer.distribute(q, x, valid)
        self.assertEqual(weights.shape, (2, 2, 3, 5))
        torch.testing.assert_close(assignment.sum(-2), valid[:, None].expand(2, 2, 5).float())
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        self.assertFalse(weights.masked_select(~valid[:, None, None]).any())
        self.assertTrue((weights[0, 0] - weights[0, 1]).abs().max().item() > 0)
        out, _ = writer(q, x.masked_fill(~valid[..., None], float("nan")), valid)
        for i in range(2):
            cropped, _ = writer(q[i:i+1], x[i:i+1, valid[i]], valid[i:i+1, valid[i]])
            torch.testing.assert_close(out[i], cropped[0])

    def test_initialization_checkpoint_and_training_objective(self):
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.memory_replay_training import route_objective
        from dinoexp.replay_matcher import build_replay_matcher

        torch.manual_seed(7)
        base = EvidenceMatcher().to(self.device)
        comp = build_replay_matcher(dict(base.config, writer_attention="competitive")).to(self.device)
        comp.load_state_dict(base.state_dict(), strict=True)
        for name, value in comp.state_dict().items():
            torch.testing.assert_close(value, base.state_dict()[name], rtol=0, atol=0)
        def sample(n):
            return torch.randn(n, 256, device=self.device), torch.ones(n, dtype=torch.bool, device=self.device)
        obs = {k: sample(16) for k in "abc"}
        queries = {k: sample(64) for k in "abcq"}
        schedule = [dict(writes=[("a", 0), ("b", 1)], replay=[("a", 0)], prewrite=[], postwrite=[]),
                    dict(writes=[("c", 0)], replay=[("a", 0)], prewrite=[("q", 0)], postwrite=[("q", 0)])]
        losses = route_objective(comp, schedule, (obs, queries), None, "competitive_prepost", .4, .7)
        reference = route_objective(comp, schedule, (obs, queries), None, "replay_prepost", .4, .7)
        for expected, actual in zip(reference, losses, strict=True):
            torch.testing.assert_close(expected, actual)
        self.assertTrue(all(x.item() > 0 for x in losses))
        losses[0].backward()
        for layer in (comp.writer.attention, comp.updater.attention, comp.reader.attention):
            self.assertGreater(layer.in_proj_weight.grad.norm().item(), 0)
        reloaded = build_replay_matcher(comp.config).to(self.device)
        reloaded.load_state_dict(comp.state_dict(), strict=True)
        x, mask = obs["a"]
        x, mask = x[None, None], mask[None, None]
        state = comp.update(x, mask)
        torch.testing.assert_close(reloaded.update(x, mask)["slots"], state["slots"])
        torch.testing.assert_close(comp.update(x, ~mask, state)["slots"], state["slots"])
        self.assertFalse(torch.allclose(base.update(x, mask)["slots"], state["slots"]))

    def test_controlled_histories_masks_counts_and_reports(self):
        import json
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.memory_replay_training import _validate
        from dinoexp.replay_matcher import build_replay_matcher
        from dinoexp.writer_experiment import evaluate_writer, select_view_pair, writer_report

        route = dict(id="route", split="val", group="heldout", identities=["car0", "car1"], steps=[], samples={})
        writes, queries = {}, {}
        for step, angle in enumerate([0, 4, 90, 92, 94, 96]):
            entries = {}
            for uid in route["identities"]:
                key = f"{uid}-{step}"
                # Second candidate's single angle cannot form a qualifying pair.
                az = angle if uid == "car0" else 0
                entries[uid] = dict(sample=key, independent=key+"q", previous_view_gap=None,
                                    probe_previous_view_gap=None)
                for k, a in ((key, az), (key+"q", az+1)):
                    route["samples"][k] = dict(azimuth=a, rgb_sha256=k)
                writes[key] = (torch.randn(16, 16, device=self.device),
                               torch.ones(16, dtype=torch.bool, device=self.device))
                queries[key+"q"] = (torch.randn(64, 16, device=self.device),
                                    torch.ones(64, dtype=torch.bool, device=self.device))
                queries[key] = queries[key+"q"]
            route["steps"].append(dict(step=step, frame=str(step), writes=entries))
        pair = select_view_pair(route, "car0", 25, 45)
        self.assertEqual(len(set([pair["a"], pair["b"], *pair["extra_a"], *pair["extra_b"]])), 6)
        self.assertIsNone(select_view_pair(route, "car1", 25, 45))
        base = EvidenceMatcher(dim=16).to(self.device)
        for config in (base.config, dict(base.config, writer_attention="competitive")):
            model = build_replay_matcher(config).to(self.device)
            model.load_state_dict(base.state_dict(), strict=True)
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                result = evaluate_writer(model, [route], {"route": (writes, queries)},
                                         dict(max_query_angle=25, novelty_angle=45), folder)
                self.assertEqual((result["eligible"], result["candidates"]), (1, 2))
                self.assertEqual(result["probes"]["matched-AB"]["total"], 1)
                self.assertIn("ABBBB", result["probes"])
                self.assertTrue((folder / "writer_heatmaps.png").is_file())
                maps = json.loads((folder / "writer_maps.json").read_text())
                self.assertEqual(len(maps), 12)
                self.assertIsNone(maps[0]["update_norm_by_slot"])
                self.assertEqual(len(maps[1]["update_norm_by_slot"]), 8)
                self.assertEqual(maps[0]["assignment"] is None, "writer_attention" not in config)
                self.assertIn("matched-AB", writer_report({"m": result}, {}))
        # The route evaluator works at the production 256-wide feature contract.
        def full_width(cache):
            return {key: (x.repeat(1, 16), mask) for key, (x, mask) in cache.items()}
        full_model = build_replay_matcher(dict(EvidenceMatcher().config, writer_attention="competitive"))
        result = _validate(full_model.to(self.device), [route], {"route": (full_width(writes), full_width(queries))}, 45)
        self.assertEqual(result["by_route"]["route"]["paired_independent"]["total"], 10)
        self.assertIn("heldout", {r["group"] for r in result["rows"]})


if __name__ == "__main__":
    unittest.main()
