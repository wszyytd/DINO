import unittest

import torch

from dinoexp.evidence_memory import EvidenceMatcher


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.x = torch.randn(2, 3, 4, 16)
        self.valid = torch.ones(2, 3, 4, dtype=torch.bool)
        self.ref = torch.randn(2, 4, 16)
        self.rv = torch.ones(2, 4, dtype=torch.bool)

    def model(self, mode='memory'):
        return EvidenceMatcher(dim=16, heads=2, slots=3, mode=mode)

    def test_missing_observation_preserves_memory_even_with_invalid_values(self):
        m = self.model()
        state = m.update(self.x, self.valid)
        absent = m.update(torch.full_like(self.x, float('nan')), self.valid & False, state)
        for key in state:
            torch.testing.assert_close(state[key], absent[key])
        self.assertTrue(torch.isfinite(m.score(self.ref, self.rv, absent)).all())

    def test_candidate_permutation_and_growth(self):
        m = self.model()
        first = m.update(self.x[:, :2], self.valid[:, :2])
        grown = m.update(self.x, self.valid, first)
        self.assertEqual(grown['slots'].shape, (2, 3, 3, 16))
        p = torch.tensor([2, 0, 1])
        state = m.update(self.x, self.valid)
        permuted = m.update(self.x[:, p], self.valid[:, p])
        a, b = m.score(self.ref, self.rv, state), m.score(self.ref, self.rv, permuted)
        torch.testing.assert_close(a[:, p], b[:, :3])
        torch.testing.assert_close(a[:, -1], b[:, -1])

    def test_loss_reaches_earlier_observation(self):
        m = self.model()
        early = self.x.clone().requires_grad_()
        state = m.update(early, self.valid)
        state = m.update(self.x * .3, self.valid, state)
        m.score(self.ref, self.rv, state)[:, 0].sum().backward()
        self.assertGreater(float(early.grad.abs().sum()), 0)

    def test_current_resets_and_mean_duplicate_is_invariant(self):
        current = self.model('current')
        a = current.update(self.x, self.valid)
        a = current.update(self.x, self.valid & False, a)
        self.assertFalse(a['seen'].any())
        mean = self.model('mean')
        a = mean.update(self.x, self.valid)
        b = mean.update(self.x, self.valid, a)
        torch.testing.assert_close(a['slots'], b['slots'])

    def test_empty_start_has_only_null_probability(self):
        m = self.model()
        state = m.update(self.x, self.valid & False)
        scores = m.score(self.ref, self.rv, state).softmax(-1)
        torch.testing.assert_close(scores[:, -1], torch.ones(2))


if __name__ == '__main__':
    unittest.main()
