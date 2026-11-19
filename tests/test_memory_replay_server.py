import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_MEMORY_REPLAY_CUDA_TESTS") == "1",
                     "run explicitly on the CUDA training server")
class MemoryReplayServerTests(unittest.TestCase):
    def test_slot_ema_first_missing_and_reloaded_state(self):
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.replay_matcher import build_replay_matcher, initialize_slot_ema

        device = torch.device("cuda:0")
        learned = EvidenceMatcher(dim=16, slots=3).to(device)
        ema = initialize_slot_ema(learned.config, learned.state_dict(), .8).to(device)
        self.assertFalse(hasattr(ema, "updater"))
        x = torch.randn(1, 2, 4, 16, device=device)
        keep = torch.ones(1, 2, 4, dtype=torch.bool, device=device)
        first = ema.update(x, keep)
        torch.testing.assert_close(first["slots"], learned.update(x, keep)["slots"])
        x2 = torch.randn_like(x)
        keep[:, 1] = False
        fresh = ema.update(x2, keep)
        second = ema.update(x2, keep, first)
        torch.testing.assert_close(second["slots"][:, 0], .8 * first["slots"][:, 0] + .2 * fresh["slots"][:, 0])
        torch.testing.assert_close(second["slots"][:, 1], first["slots"][:, 1])
        self.assertEqual(second["count"].tolist(), [[2, 1]])
        loaded = build_replay_matcher(ema.config).to(device)
        loaded.load_state_dict(ema.state_dict(), strict=True)
        torch.testing.assert_close(loaded.update(x2, keep, first)["slots"], second["slots"])

    def test_independent_postwrite_alone_trains_the_current_update(self):
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.memory_replay_training import route_objective

        torch.manual_seed(7)
        model = EvidenceMatcher().cuda()
        def encode(n):
            return torch.randn(n, 256, device="cuda"), torch.ones(n, dtype=torch.bool, device="cuda")
        writes = {key: encode(16) for key in ("a", "b", "c")}
        queries = {key: encode(64) for key in (*writes, "q")}
        schedule = [
            dict(writes=[("a", 0), ("b", 1)], prewrite=[], postwrite=[], replay=[("a", 0), ("b", 1)]),
            dict(writes=[("c", 0)], prewrite=[("q", 0)], postwrite=[("q", 0)], replay=[("c", 0)])]
        total, replay, pre, post = route_objective(model, schedule, (writes, queries), None,
                                                 "replay_prepost", .4, .7)
        torch.testing.assert_close(total, replay + .4 * pre + .7 * post)
        post.backward()
        self.assertGreater(model.gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.updater.attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_prewrite_has_no_current_rgb_and_training_reaches_the_gate(self):
        import torch

        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.memory_replay_training import route_loss
        from dinoexp.region_ab_prepare import cuda_device

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = cuda_device("cuda:0")
        torch.manual_seed(7)
        model = EvidenceMatcher().to(device)

        def encoded(count):
            return (torch.randn(count, 256, device=device),
                    torch.ones(count, dtype=torch.bool, device=device))

        writes = {key: encoded(16) for key in ("a1", "b1", "a2")}
        queries = {key: encoded(64) for key in ("a1", "b1", "a2", "q")}
        schedule = [
            dict(writes=[("a1", 0), ("b1", 1)], prewrite=[], replay=[("a1", 0), ("b1", 1)]),
            dict(writes=[("a2", 0)], prewrite=[("q", 0)], replay=[("a1", 0), ("b1", 1), ("a2", 0)]),
        ]
        total, _, pre = route_loss(model, schedule, (writes, queries), None, "replay_independent", 1.)
        with torch.no_grad():
            amended = dict(writes, a2=encoded(16))
            _, _, amended_pre = route_loss(model, schedule, (amended, queries), None,
                                           "replay_independent", 1.)
        torch.testing.assert_close(pre.detach(), amended_pre, atol=0, rtol=0)
        total.backward()
        self.assertGreater(float(model.gate.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(model.gate.weight.grad).all())
        self.assertGreater(float(model.updater.attention.in_proj_weight.grad.abs().sum()), 0.)
