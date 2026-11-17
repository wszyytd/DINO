import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_MEMORY_REPLAY_CUDA_TESTS") == "1",
                     "run explicitly on the CUDA training server")
class MemoryReplayServerTests(unittest.TestCase):
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
