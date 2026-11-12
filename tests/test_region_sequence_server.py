"""CUDA-only tests. Opt in on the authorized server, never on the local PC."""
import copy
import os
import unittest

ENABLED = os.environ.get('RUN_REGION_SEQUENCE_CUDA_TESTS') == '1'
if ENABLED:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from dinoexp.region_ab_prepare import cuda_device
    from dinoexp.region_sequence_data import DEFAULT_CONFIG
    from dinoexp.region_sequence_training import (advance, make_model, rollout, score,
                                                  server_checks, swap_content)


@unittest.skipUnless(ENABLED, 'Authorized CUDA server only: RUN_REGION_SEQUENCE_CUDA_TESTS=1')
class SequenceServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = cuda_device('cuda')
        torch.set_num_threads(4)
        cls.cfg = copy.deepcopy(DEFAULT_CONFIG['training'])

    def test_mandatory_training_preflight(self):
        self.assertEqual(server_checks(self.device, self.cfg)['status'], 'passed')

    def test_all_methods_are_causal_and_keep_the_last_absent_car(self):
        x = torch.randn(1, 4, 2, 16, 256, device=self.device)
        valid = torch.zeros(1, 4, 2, 16, dtype=torch.bool, device=self.device)
        valid[:, 0, 0] = True; valid[:, 1, 1] = True; valid[:, 2, 0] = True
        inputs = dict(observation=x, valid=valid, current_valid=valid,
                      reference=torch.randn(1, 64, 256, device=self.device),
                      reference_valid=torch.ones(1, 64, dtype=torch.bool, device=self.device))
        with torch.no_grad():
            for method in ('last', 'mean', 'memory'):
                model = make_model(self.cfg, 27, self.device, method).eval()
                logits, final = rollout(model, method, inputs)
                changed = dict(inputs, observation=x.clone())
                changed['observation'][:, 2:] += 100
                altered, _ = rollout(model, method, changed)
                torch.testing.assert_close(logits[:, :2], altered[:, :2], atol=0, rtol=0)
                self.assertTrue(final['seen'].all())
                torch.testing.assert_close(final['count'], torch.tensor([[2., 1.]], device=self.device))
                stopped = advance(model, method, x[:, 3], valid[:, 3], final)
                snapshot = {k: v.clone() for k, v in stopped.items()}
                score(model, inputs, stopped)
                for k in stopped:
                    torch.testing.assert_close(stopped[k], final[k], atol=0, rtol=0)
                    torch.testing.assert_close(stopped[k], snapshot[k], atol=0, rtol=0)

    def test_interventions_do_not_mutate_the_normal_prefix(self):
        model = make_model(self.cfg, 7, self.device, 'memory').eval()
        x = torch.randn(1, 2, 16, 256, device=self.device)
        valid = torch.ones(1, 2, 16, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            state = advance(model, 'memory', x, valid)
            snapshot = {k: v.clone() for k, v in state.items()}
            alternative = advance(model, 'memory', x+1, valid, swap_content(state))
            for k in state:
                torch.testing.assert_close(state[k], snapshot[k], atol=0, rtol=0)
            self.assertTrue(torch.isfinite(alternative['slots']).all())


if __name__ == '__main__':
    unittest.main()
