"""CUDA-only neural checks. Run this file ON THE SERVER, never as a local CPU test."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import copy
import unittest
import numpy as np
import torch

from dinoexp.region_ab_training import (server_checks, make_batch, forward_current,
                                       make_model, initialization_hash)
from dinoexp.region_ab_prepare import cuda_device
from dinoexp.region_geometry import DEFAULTS


@unittest.skipUnless(torch.cuda.is_available(), 'server CUDA required; CPU forward is prohibited')
class RegionServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = cuda_device('cuda')

    def test_synthetic_checks_and_equal_initial_weights(self):
        cfg = DEFAULTS['training']
        self.assertEqual(server_checks(self.device, cfg)['status'], 'passed')
        a, b = make_model(cfg, 7, self.device), make_model(cfg, 7, self.device)
        self.assertEqual(initialization_hash(a), initialization_hash(b))

    def test_teacher_scores_and_identity_never_enter_network(self):
        row = dict(id='example', candidates=[{'box': [0, 0, 30, 30]}, {'box': [2, 2, 20, 20]}],
                   reference_sha256='ref', teacher_null=.1, target_identity='A', spatial=[[.5]])
        labels = {'example': dict(positives=[0])}
        rng = np.random.default_rng(7)
        refs = {'ref': dict(tokens=rng.normal(size=(64, 256)).astype('float32'), valid=np.ones(64, bool))}
        features = {'example': dict(observation=rng.normal(size=(2, 16, 256)).astype('float32'),
                                    valid=np.ones((2, 16), bool), errors=[])}
        original, target = make_batch([row], labels, refs, features, self.device)
        poisoned = dict(row, teacher_null=1e20, spatial=[[-1e20]], target_identity='B')
        alternate, _ = make_batch([poisoned], labels, refs, features, self.device)
        self.assertEqual(set(original), {'observation', 'valid', 'reference', 'reference_valid'})
        for key in original:
            torch.testing.assert_close(original[key], alternate[key], atol=0, rtol=0)
        moved, moved_target = make_batch([row], labels, refs, features, self.device, [[1, 0]])
        self.assertEqual(target[0].nonzero().flatten().tolist(), [0])
        self.assertEqual(moved_target[0].nonzero().flatten().tolist(), [1])
        model = make_model(DEFAULTS['training'], 7, self.device)
        torch.testing.assert_close(forward_current(model, moved), forward_current(model, original)[:, [1, 0, 2]], atol=2e-5, rtol=1e-5)
        # Both empty proposals and an explicit invalid candidate must retain Null supervision.
        empty = dict(row, candidates=[])
        empty_features = {'example': dict(observation=np.zeros((0, 16, 256), 'float32'), valid=np.zeros((0, 16), bool), errors=[])}
        batch, target = make_batch([empty], {'example': {'positives': []}}, refs, empty_features, self.device)
        self.assertEqual(target[0].nonzero().flatten().tolist(), [1])
        self.assertEqual(forward_current(model, batch).argmax(1).item(), 1)
        invalid_features = copy.deepcopy(features); invalid_features['example']['valid'][:] = False
        batch, _ = make_batch([row], labels, refs, invalid_features, self.device)
        with torch.no_grad():
            model.null.fill_(-20000)
        self.assertEqual(forward_current(model, batch).argmax(1).item(), 2)


if __name__ == '__main__':
    unittest.main()
