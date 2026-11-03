import unittest

import numpy as np

from dinoexp.evidence_state import Action, Observation, ReplayState, apply_action, true_margin
from dinoexp.evidence_data import relative_pose, split_by_position


class ReplayStateTests(unittest.TestCase):
    def setUp(self):
        self.first = Observation('first', 'A', 'a' * 64, (0., 1., 0., 1., 0.))
        self.state = ReplayState(('A', 'B'), ((self.first,), ()))

    def test_repeat_is_idempotent_and_counterfactual_branches_are_isolated(self):
        action = Action('a1', 0, self.first.pose, 'side')
        repeated = apply_action(self.state, action, self.first)
        self.assertEqual(repeated.histories, self.state.histories)
        self.assertEqual(repeated.attempted, ('a1',))
        alias = Observation('copy', 'A', self.first.image_hash, self.first.pose)
        self.assertEqual(apply_action(self.state, action, alias).histories, self.state.histories)
        fresh = Observation('new', 'A', 'b' * 64, self.first.pose)
        changed = apply_action(self.state, action, fresh)
        self.assertEqual(len(changed.histories[0]), 2)
        self.assertEqual(len(self.state.histories[0]), 1)
        self.assertEqual(self.state.attempted, ())

    def test_failure_consumes_attempt_without_memory_and_bad_route_fails(self):
        action = Action('failed', 1, self.first.pose, 'side')
        rejected = Observation('bad', 'B', None, self.first.pose, valid=False)
        next_state = apply_action(self.state, action, rejected)
        self.assertEqual(next_state.histories, self.state.histories)
        self.assertEqual(len(next_state.attempted), 1)
        with self.assertRaisesRegex(ValueError, 'candidate'):
            apply_action(self.state, action, self.first)
        with self.assertRaisesRegex(ValueError, 'already'):
            apply_action(next_state, action, rejected)

    def test_true_target_margin_recomputes_strongest_distractor(self):
        self.assertAlmostEqual(true_margin([.4, .3, .2], 0), .1)
        self.assertAlmostEqual(true_margin([.4, -.2, .35], 0), .05)
        with self.assertRaises(ValueError):
            true_margin([float('nan'), .2], 0)

    def test_actual_candidate_orientation_changes_relative_pose(self):
        camera = [10., 0., 0., 0., 0., 0.]
        a = relative_pose(camera, [0., 0., 0., 0., 0., 0.])
        b = relative_pose(camera, [0., 0., 0., 0., 180., 0.])
        np.testing.assert_allclose(a, [0, 1, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(b, [0, -1, 0, 1, 0], atol=1e-6)
        with self.assertRaises(ValueError):
            relative_pose([], [0] * 6)

    def test_location_roles_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            split_by_position([], {'repr': ['42'], 'value': ['42']})


if __name__ == '__main__':
    unittest.main()
