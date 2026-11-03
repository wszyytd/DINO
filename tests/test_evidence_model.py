import unittest

import torch

from dinoexp.evidence_model import EvidenceModel, ActionValue, prediction_loss, teacher_targets


class EvidenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.model = EvidenceModel(input_dim=8, dim=16, layers=1, heads=4,
                                   memory_tokens=3, blocks=1).eval()

    def test_history_order_and_padding_do_not_change_memory(self):
        z = torch.randn(2, 3, 5, 16)
        poses = torch.randn(2, 3, 5)
        quality = torch.randn(2, 3, 3)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        a = self.model.aggregate(z, poses, quality, valid)
        order = [2, 0, 1]
        b = self.model.aggregate(z[:, order], poses[:, order], quality[:, order], valid[:, order])
        torch.testing.assert_close(a, b)
        z[~valid] = 1e20
        poses[~valid] = 1e20
        quality[~valid] = 1e20
        torch.testing.assert_close(a, self.model.aggregate(z, poses, quality, valid))
        self.assertTrue(torch.isfinite(a).all())

    def test_observation_encoding_and_both_losses_have_gradients(self):
        features = torch.randn(4, 1, 4, 8)
        valid = torch.ones(4, 4, dtype=torch.bool)
        z = self.model.encode_observations(features, valid)
        self.assertEqual(z.shape, (4, 5, 16))
        q = self.model.encode_reference(features[:1], valid[:1])
        memory = self.model.aggregate(z.reshape(2, 2, 5, 16), torch.randn(2, 2, 5),
                                      torch.zeros(2, 2, 3), torch.ones(2, 2, dtype=torch.bool))
        logits = self.model.match(q, memory.unsqueeze(0))
        prediction = self.model.predict(memory, torch.randn(2, 5))
        target = torch.randn(2, 5, 8)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
        loss = loss + prediction_loss(prediction, target, torch.ones(2, 5, dtype=torch.bool))
        loss.backward()
        self.assertIsNotNone(self.model.observation_adapter.projections[0].weight.grad)
        self.assertIsNotNone(self.model.memory_queries.grad)
        self.assertEqual(prediction.shape, (2, 5, 8))

    def test_value_is_equivariant_to_candidate_order(self):
        value = ActionValue(dim=16, heads=4).eval()
        q = torch.randn(1, 4, 16)
        memories = torch.randn(1, 3, 3, 16)
        scores = torch.tensor([[.1, .2, .3]])
        counts = torch.tensor([[1., 3., 2.]])
        subjects = torch.tensor([[0, 2]])
        poses = torch.randn(1, 2, 5)
        before = value(q, memories, scores, counts, subjects, poses)
        perm = [2, 0, 1]
        after = value(q, memories[:, perm], scores[:, perm], counts[:, perm],
                      torch.tensor([[1, 0]]), poses)
        torch.testing.assert_close(before, after)
        self.assertLessEqual(before.abs().max().item(), 4.)

    def test_teacher_masks_padding_and_prediction_loss_skips_invalid_targets(self):
        features = torch.arange(32.).reshape(1, 1, 4, 8)
        mask = torch.tensor([[True, False, False, False]])
        target, valid = teacher_targets(features, mask, torch.ones(1, 8))
        self.assertEqual(valid.tolist(), [[True, True, False, False, False]])
        changed = features.clone()
        changed[:, :, 1:] = 1e20
        other, _ = teacher_targets(changed, mask, torch.ones(1, 8))
        torch.testing.assert_close(target, other)
        prediction = target.clone().requires_grad_()
        loss = prediction_loss(prediction, target, torch.zeros_like(valid))
        self.assertEqual(loss.item(), 0.)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
