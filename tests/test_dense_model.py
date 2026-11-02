import unittest
import torch
from dinoexp.dense_model import DenseSearchHead, fuse_spatial


def tiny_batch(batch=2):
    torch.manual_seed(12)
    indices = torch.arange(4).reshape(1, 1, 4, 1).expand(batch, 2, 4, 4).clone()
    weights = torch.zeros(batch, 2, 4, 4); weights[..., 0] = 1
    return dict(reference=torch.randn(batch, 4, 4, 8), reference_valid=torch.ones(batch, 4, dtype=torch.bool),
        views=torch.randn(batch, 2, 4, 4, 8), view_valid=torch.ones(batch, 2, 4, dtype=torch.bool),
        geometry=torch.tensor([[[0.,0.,1.,1.,1.,1.],[0.,0.,.5,.5,1.,1.]]]).expand(batch,-1,-1).clone(),
        indices=indices, weights=weights, priors=torch.ones(batch,2), view_weights=torch.ones(batch,2),
        camera_state=torch.zeros(batch,8), camera_valid=torch.tensor([True,False][:batch]))


class DenseModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_overlap_is_averaged_and_duplicate_multiplicity_does_not_add_votes(self):
        p = torch.tensor([[[.8,.2],[.2,.8]]])
        idx = torch.tensor([[[[0,0,0,0],[1,1,1,1]],[[0,0,0,0],[1,1,1,1]]]])
        w = torch.zeros(1,2,2,4); w[...,0]=1
        original, _ = fuse_spatial(p, idx, w, torch.ones(1,2), 2)
        duplicated, _ = fuse_spatial(p[:,[0,1,1]], idx[:,[0,1,1]], w[:,[0,1,1]],
                                      torch.tensor([[1.,.5,.5]]), 2)
        torch.testing.assert_close(original, duplicated)
        torch.testing.assert_close(original, torch.tensor([[.5,.5,0.,0.]]))

    def test_outputs_normalize_and_invalid_padding_cannot_change_predictions(self):
        model = DenseSearchHead(input_dim=8,dim=16,heads=2,grid_size=2)
        batch = tiny_batch()
        batch['view_valid'][:,:,3] = False; batch['reference_valid'][:,3] = False
        before = model(**batch)['probabilities']
        batch['views'][:,:,:,3] = 1e20; batch['reference'][:,:,3] = -1e20
        after = model(**batch)['probabilities']
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(before.sum(1), torch.ones(2))
        self.assertTrue((before>=0).all())

    def test_reference_changes_predictions_and_spatial_loss_reaches_trainable_branches(self):
        model = DenseSearchHead(input_dim=8,dim=16,heads=2,grid_size=2)
        batch = tiny_batch()
        first = model(**batch)['probabilities']
        loss = -first[0,0].log() - first[1,-1].log()
        loss.backward()
        for name in ('reference_adapter','view_adapter','correlation','patch_score',
                     'local_null','reliability','global_existence','view_state'):
            gradients = [p.grad for p in getattr(model,name).parameters() if p.grad is not None]
            self.assertTrue(any(float(g.abs().sum())>0 for g in gradients), name)
        batch['reference'] = batch['reference'] * -3
        self.assertFalse(torch.allclose(first, model(**batch)['probabilities']))

    def test_real_adaptation_freezes_state_and_null_but_not_matcher(self):
        model = DenseSearchHead(input_dim=8,dim=16,heads=2,grid_size=2)
        model.freeze_real_branches()
        for name in ('view_state','local_null','global_existence'):
            self.assertTrue(all(not p.requires_grad for p in getattr(model,name).parameters()))
        self.assertTrue(model.patch_score.weight.requires_grad)
        self.assertFalse(model.missing_state.requires_grad)
