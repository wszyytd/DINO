import unittest
import numpy as np
import torch
from dinoexp.dense_loss import dense_loss


class DenseLossTests(unittest.TestCase):
    def test_simulation_focal_loss_matches_hand_calculation(self):
        p = torch.tensor([[.25, .75]], requires_grad=True)
        labels = [dict(mode='simulation', target=np.array([1., 0.]), known=np.array([True]))]
        loss = dense_loss(p, labels)
        self.assertAlmostEqual(float(loss.detach()), -.75**2 * np.log(.25), places=6)
        loss.backward()
        self.assertNotEqual(float(p.grad[0, 0]), 0.)

    def test_real_unknown_regions_and_global_null_receive_no_localization_gradient(self):
        p = torch.tensor([[.2, .3, .1, .4]], requires_grad=True)
        labels = [dict(mode='confirmed_regions', target=np.array([1., 0., 0., 0.]),
                       known=np.array([True, True, False]))]
        loss = dense_loss(p, labels, gamma=0.)
        self.assertAlmostEqual(float(loss.detach()), -np.log(.4), places=6)
        loss.backward()
        self.assertEqual(float(p.grad[0, 2]), 0.)
        self.assertEqual(float(p.grad[0, 3]), 0.)
