import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional training dependencies absent')
class LocalAttentionTests(unittest.TestCase):
    def test_padding_cannot_change_match_and_gradients_reach_head(self):
        import torch
        from dinoexp.local_attention import LocalMatcher
        torch.manual_seed(4)
        model = LocalMatcher(input_dim=12, dim=16, layers=2, heads=4,
                             probes=3, reference_grid=2, blocks=1)
        x = torch.randn(2, 2, 16, 12)
        valid = torch.ones(2, 16, dtype=torch.bool)
        valid[:, :4] = False
        model.eval()
        a = model(x, valid, x.flip(-1), valid)
        changed = x.clone()
        changed[:, :, :4] = 1e5
        b = model(changed, valid, x.flip(-1), valid)
        torch.testing.assert_close(a['logit'], b['logit'])
        self.assertEqual(tuple(a['query_attention'].shape), (2, 3, 16))
        self.assertTrue(torch.equal(a['query_attention'][:, :, :4],
                                    torch.zeros_like(a['query_attention'][:, :, :4])))
        a['logit'].sum().backward()
        self.assertIsNotNone(model.reference_adapter.projections[0].weight.grad)
        self.assertIsNotNone(model.candidate_adapter.projections[0].weight.grad)
        self.assertIsNotNone(model.probes.grad)

    def test_all_padding_and_nonsquare_grid_fail_explicitly(self):
        import torch
        from dinoexp.local_attention import LocalMatcher
        model = LocalMatcher(input_dim=8, dim=16, layers=1, heads=4,
                             probes=2, reference_grid=2, blocks=1)
        x = torch.randn(1, 1, 16, 8)
        with self.assertRaises(ValueError):
            model(x, torch.zeros(1, 16, dtype=torch.bool), x,
                  torch.ones(1, 16, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(x[:, :, :15], torch.ones(1, 15, dtype=torch.bool), x,
                  torch.ones(1, 16, dtype=torch.bool))

    def test_pair_network_can_fit_a_tiny_synthetic_training_set(self):
        import torch
        from dinoexp.local_attention import LocalMatcher
        torch.set_num_threads(1)
        torch.manual_seed(9)
        model = LocalMatcher(input_dim=8, dim=16, layers=1, heads=4,
                             probes=2, reference_grid=2, blocks=1)
        r = torch.randn(4, 1, 16, 8)
        q = r.clone()
        q[2:] = -r[2:]
        mask = torch.ones(4, 16, dtype=torch.bool)
        labels = torch.tensor([1., 1., 0., 0.])
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = []
        for _ in range(35):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(r, mask, q, mask)['logit'], labels)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] / 3)


if __name__ == '__main__':
    unittest.main()
