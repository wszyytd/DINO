import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from dinoexp.evidence_model import ActionValue
from dinoexp.evidence_update_diagnostics import (
    fit_updates, gradient_geometry, pack_records, phase_weights, state_losses,
)
from tests.test_evidence_diagnostics import record


class UpdateDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.rows = [record(), record('other', 'h2')]
        self.rows[1]['targets'] = self.rows[1]['targets'][:, :2] + .7
        self.rows[1]['action_ids'] = self.rows[1]['action_ids'][:2]
        for key in ('subjects', 'poses'):
            self.rows[1]['inputs'][key] = self.rows[1]['inputs'][key][:, :2]

    def test_packed_loss_and_gradients_match_weighted_individual_states(self):
        model = ActionValue(dim=8, heads=2)
        x, y, mask = pack_records(self.rows)
        weights = torch.tensor([.25, .75])
        actual = (state_losses(model(**x), y, mask)*weights).sum()
        params = tuple(model.parameters())
        packed_grad = torch.autograd.grad(actual, params)
        expected = sum(w*F.mse_loss(model(**r['inputs'])/4, r['targets']/4)
                       for w, r in zip(weights, self.rows, strict=True))
        reference_grad = torch.autograd.grad(expected, params)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(packed_grad, reference_grad, strict=True):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        dirty = y.clone()
        dirty[~mask] = float('nan')
        torch.testing.assert_close(state_losses(model(**x), dirty, mask),
                                   state_losses(model(**x), y, mask))

    def test_phase_weights_preserve_original_hard_all_sampling_objective(self):
        odd = phase_weights(4, [0, 2], 1)
        even = phase_weights(4, [0, 2], 2)
        torch.testing.assert_close(odd, torch.tensor([.5, 0., .5, 0.]))
        torch.testing.assert_close(even, torch.full((4,), .25))
        torch.testing.assert_close((odd+even)/2, torch.tensor([.375, .125, .375, .125]))

    def test_gradient_geometry_identifies_cancellation_without_nan(self):
        result = gradient_geometry(torch.tensor([[1., 0.], [-1., 0.], [0., 0.]]),
                                   torch.tensor([.5, .5, 0.]))
        self.assertEqual(result['negative_cosine_fraction'], 1.)
        self.assertEqual(result['nonzero_gradient_states'], 2)
        self.assertEqual(result['mean_gradient_norm'], 0.)
        self.assertEqual(result['mean_to_individual_squared_norm_ratio'], 0.)

    def test_fit_modes_share_initial_state_and_count_work_without_changing_data(self):
        before = copy.deepcopy(self.rows)
        with tempfile.TemporaryDirectory() as folder:
            single = fit_updates(self.rows, {'dim': 8, 'heads': 2}, Path(folder)/'single',
                                 'single', steps=2, probe_steps=(0, 2))
            joint = fit_updates(self.rows, {'dim': 8, 'heads': 2}, Path(folder)/'joint',
                                'joint', steps=2, probe_steps=(0, 2))
            self.assertEqual(single['initial_fingerprint'], joint['initial_fingerprint'])
            self.assertEqual(single['curve'][-1]['training_state_forwards'], 2)
            self.assertEqual(joint['curve'][-1]['training_state_forwards'], 4)
            self.assertEqual(len(joint['per_step']), 3)
            with self.assertRaises(FileExistsError):
                fit_updates(self.rows, {'dim': 8, 'heads': 2}, Path(folder)/'joint', 'joint', steps=2)
        for a, b in zip(self.rows, before, strict=True):
            self.assertTrue(torch.equal(a['targets'], b['targets']))
            for key in a['inputs']:
                self.assertTrue(torch.equal(a['inputs'][key], b['inputs'][key]))


if __name__ == '__main__':
    unittest.main()
