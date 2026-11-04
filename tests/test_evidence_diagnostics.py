import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dinoexp.evidence_data import Episode
from dinoexp.evidence_diagnostics import (
    fixed_fit, input_audit, paired_selection, ranking_diagnostics, reconstruct_state,
    transition_result, unique_record_indices,
)
from dinoexp.evidence_state import Action, Observation, ReplayState


def record(episode='e', history='h', targets=(0., 1., -.5, .2)):
    generator = torch.Generator().manual_seed(3)
    return {'episode_id': episode, 'history_ids': [[history], ['b']],
            'action_ids': ['a', 'b', 'c', 'd'], 'upstream_fingerprint': 'f',
            'targets': torch.tensor([targets]),
            'inputs': {'q': torch.randn(1, 2, 8, generator=generator),
                       'memories': torch.randn(1, 2, 2, 8, generator=generator),
                       'scores': torch.tensor([[.2, -.2]]), 'counts': torch.ones(1, 2),
                       'subjects': torch.tensor([[0, 0, 1, 1]]),
                       'poses': torch.randn(1, 4, 5, generator=generator)}}


class EvidenceDiagnosticTests(unittest.TestCase):
    def test_input_conflicts_use_the_means_actually_seen_by_value_head(self):
        first = record()
        second = copy.deepcopy(first)
        second['inputs']['q'] = second['inputs']['q'].flip(1)
        second['inputs']['memories'] = second['inputs']['memories'].flip(2)
        second['targets'][0, 0] += 1.
        self.assertTrue(input_audit([first, second])['exact_input_label_conflicts'])

    def test_cli_rejects_existing_output_before_loading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, 'run_evidence_diagnostics.py', '--data',
                                     'missing', '--cache', 'missing', '--source-run', 'missing',
                                     '--output', folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('output exists', result.stderr)

    def test_unique_states_reject_conflicting_labels_and_pair_selection_is_nested(self):
        rows = [record('e'+str(i % 4), 'h'+str(i)) for i in range(20)]
        rows.append(copy.deepcopy(rows[0]))
        unique, duplicates = unique_record_indices(rows)
        self.assertEqual(len(unique), 20)
        self.assertEqual(duplicates, 1)
        selection = paired_selection([rows, copy.deepcopy(rows)], (1, 4, 16), seed=7)
        self.assertEqual(selection['subsets']['1'], selection['subsets']['16'][:1])
        self.assertEqual(selection['subsets']['4'], selection['subsets']['16'][:4])
        self.assertEqual(len({rows[i]['episode_id'] for i in selection['subsets']['4']}), 4)
        rows[-1]['targets'][0, 0] = 3.
        with self.assertRaisesRegex(ValueError, 'conflict'):
            unique_record_indices(rows)

    def test_selection_rejects_misaligned_states_and_ignores_near_constant_targets(self):
        rows = [record('e', str(i)) for i in range(4)]
        other = copy.deepcopy(rows)
        other[0]['targets'].zero_()
        result = paired_selection([rows, other], (1,), seed=7)
        self.assertNotIn(0, result['subsets']['1'])
        other[1]['history_ids'] = [['wrong'], ['b']]
        with self.assertRaisesRegex(ValueError, 'aligned'):
            paired_selection([rows, other], (1,), seed=7)

    def test_reconstruct_preserves_attempted_failure_and_rejects_future_history(self):
        pose = (0., 1., 0., 1., 0.)
        first = Observation('first', 'A', 'a', pose)
        fresh = Observation('new', 'A', 'c', pose)
        failed = Observation('bad', 'B', None, pose, valid=False)
        actions = (Action('new', 0, pose, 'side'), Action('bad', 1, pose, 'front'))
        initial = ReplayState(('A', 'B'), ((first,), ()))
        episode = Episode('e', 'g', '64', 'ref', ('A', 'B'), 0, initial,
                          actions, {'new': fresh, 'bad': failed})
        row = {'episode_id': 'e', 'history_ids': [['first'], []], 'action_ids': ['new']}
        state = reconstruct_state(episode, row)
        self.assertEqual(state.histories, initial.histories)
        self.assertEqual(state.attempted, ('bad',))
        row['history_ids'] = [['first', 'new'], []]
        with self.assertRaises(ValueError):
            reconstruct_state(episode, row)

    def test_transition_uses_true_identity_and_ties_are_not_correct(self):
        corrected = transition_result([.1, .4], [.7, .2], 0)
        self.assertEqual(corrected['transition'], 'corrected')
        self.assertAlmostEqual(corrected['gain'], .8)
        self.assertEqual(transition_result([.7, .2], [.1, .4], 0)['transition'],
                         'introduced_error')
        tied = transition_result([.5, .5], [.5, .5], 1)
        self.assertFalse(tied['before_correct'])
        self.assertIsNone(tied['before_prediction'])

    def test_ranking_ignores_near_ties_but_regret_uses_actual_values(self):
        metrics = ranking_diagnostics(torch.tensor([.1, 0., 2.]),
                                      torch.tensor([0., .000001, 1.]), gap=.01)
        self.assertEqual(metrics['meaningful_pair_count'], 2)
        self.assertEqual(metrics['meaningful_pair_accuracy'], 1.)
        self.assertEqual(metrics['regret'], 0.)

    def test_fixed_fit_records_gradients_and_does_not_modify_inputs(self):
        torch.set_num_threads(1)
        rows = [record()]
        before = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as folder:
            result = fixed_fit(rows, {'dim': 8, 'heads': 2}, Path(folder)/'fit',
                               steps=2, evaluate_every=1, seed=7)
            self.assertEqual([r['step'] for r in result['curve']], [0, 1, 2])
            self.assertGreater(result['updates'][0]['gradient_norm'], 0.)
            self.assertGreater(result['parameter_delta_l2'], 0.)
            self.assertTrue((Path(folder)/'fit/last.pt').exists())
            with self.assertRaises(FileExistsError):
                fixed_fit(rows, {'dim': 8, 'heads': 2}, Path(folder)/'fit', steps=2)
        for key in rows[0]['inputs']:
            self.assertTrue(torch.equal(rows[0]['inputs'][key], before[0]['inputs'][key]))
        self.assertTrue(torch.equal(rows[0]['targets'], before[0]['targets']))


if __name__ == '__main__':
    unittest.main()
