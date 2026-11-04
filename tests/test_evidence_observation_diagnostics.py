import copy
from dataclasses import replace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dinoexp.evidence_model import ActionValue
from dinoexp.evidence_training import load_value
from tests.test_evidence_diagnostics import record


class ObservationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_observation_branch_starts_at_baseline_and_masks_invalid_features(self):
        from dinoexp.evidence_observation_diagnostics import ObservationValue
        x = record()['inputs']
        future = torch.randn(1, 4, 5, 8)
        valid = torch.tensor([[True, False, True, True]])
        torch.manual_seed(7)
        baseline = ActionValue(dim=8, heads=2)
        torch.manual_seed(7)
        model = ObservationValue(dim=8, heads=2)
        torch.testing.assert_close(model(**x, future=future, future_valid=valid), baseline(**x))
        with torch.no_grad():
            model.observation_output[-1].weight.fill_(.1)
        dirty = future.clone()
        dirty[:, 1] = float('nan')
        torch.testing.assert_close(model(**x, future=dirty, future_valid=valid),
                                   model(**x, future=future, future_valid=valid))
        future.requires_grad_()
        model(**x, future=future, future_valid=valid).sum().backward()
        self.assertGreater(float(future.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(future.grad[:, 1].abs().sum()), 0)

    def test_production_loader_explicitly_rejects_diagnostic_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'diagnostic.pt'
            torch.save({'diagnostic_only': True}, path)
            with self.assertRaisesRegex(ValueError, 'diagnostic'):
                load_value(path, 'upstream')

    def test_real_records_align_actions_validate_labels_and_leave_upstream_frozen(self):
        from dinoexp.evidence_labels import FrozenRuntime, fingerprint, generate_labels
        from dinoexp.evidence_model import EvidenceModel
        from dinoexp.evidence_observation_data import attach_observations
        from tests.test_evidence_experiment import SmallBank, small_episode
        bank, episode = SmallBank(), small_episode()
        model = EvidenceModel(input_dim=8, dim=8, layers=1, heads=2,
                              memory_tokens=2, blocks=1)
        runtime = FrozenRuntime(model, bank)
        rows = generate_labels(runtime, [episode], states_per_episode=1)
        initial = fingerprint(model)
        enriched, audit = attach_observations(runtime, [episode], rows)
        row = enriched[0]
        self.assertEqual(row['inputs']['future_valid'].tolist(), [[True, True, False]])
        self.assertEqual(float(row['inputs']['future'][0, 2].abs().sum()), 0)
        image = bank.get('3')
        expected = model.encode_observations(image['tokens'][None], image['valid'][None])[0]
        torch.testing.assert_close(row['inputs']['future'][0, 0], expected)
        self.assertEqual(set(row['inputs'])-set(rows[0]['inputs']), {'future', 'future_valid'})
        self.assertLess(audit['max_label_error'], 1e-5)
        self.assertEqual(fingerprint(model), initial)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertNotIn('future', rows[0]['inputs'])
        bad = copy.deepcopy(rows)
        bad[0]['targets'][0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'label'):
            attach_observations(runtime, [episode], bad)
        bad = copy.deepcopy(rows)
        bad[0]['action_ids'].reverse()
        with self.assertRaises(ValueError):
            attach_observations(runtime, [episode], bad)

    def test_chunked_gradient_matches_full_batch_with_global_state_weights(self):
        from dinoexp.evidence_observation_diagnostics import backward_chunks, prepare_records
        rows = [record(history=str(i)) for i in range(3)]
        rows[1]['targets'] = rows[1]['targets'][:, :2]
        rows[1]['action_ids'] = rows[1]['action_ids'][:2]
        for key in ('poses', 'subjects'):
            rows[1]['inputs'][key] = rows[1]['inputs'][key][:, :2]
        model = ActionValue(dim=8, heads=2)
        weights = torch.tensor([.2, 0., .8])
        fixed = prepare_records(rows, 'direct_joint')
        backward_chunks(model, fixed, weights, chunk_size=3)
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        backward_chunks(model, fixed, weights, chunk_size=1)
        for want, p in zip(expected, model.parameters(), strict=True):
            torch.testing.assert_close(p.grad, want, atol=2e-6, rtol=2e-5)

    def test_capacity_controls_keep_targets_but_zero_all_privileged_inputs(self):
        from dinoexp.evidence_observation_diagnostics import prepare_records
        row = record()
        row['inputs']['future'] = torch.randn(1, 4, 5, 8)
        row['inputs']['future_valid'] = torch.ones(1, 4, dtype=torch.bool)
        before = copy.deepcopy(row)
        zero = prepare_records([row], 'observation_zero_joint')[0]
        real = prepare_records([row], 'observation_real_joint')[0]
        direct = prepare_records([row], 'direct_joint')[0]
        self.assertEqual(float(zero['inputs']['future'].abs().sum()), 0)
        self.assertFalse(zero['inputs']['future_valid'].any())
        self.assertNotIn('future', direct['inputs'])
        torch.testing.assert_close(real['inputs']['future'], before['inputs']['future'])
        torch.testing.assert_close(zero['targets'], before['targets'])
        torch.testing.assert_close(row['inputs']['future'], before['inputs']['future'])

    def test_all_unique_selection_keeps_zero_gain_states_and_rejects_branch_misalignment(self):
        from dinoexp.evidence_observation_data import paired_unique
        rows = [record(history='first'), record(history='second', targets=(0., 0., 0., 0.))]
        rows.append(copy.deepcopy(rows[0]))
        selected, audit = paired_unique({'a': rows, 'b': copy.deepcopy(rows)})
        self.assertEqual(audit['unique_states'], 2)
        self.assertEqual(audit['duplicates'], 1)
        self.assertEqual(float(selected['a'][1]['targets'].abs().sum()), 0.)
        bad = copy.deepcopy(rows)
        bad[1]['action_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'aligned'):
            paired_unique({'a': rows, 'b': bad})

    def test_development_requires_opt_in_and_rejects_role_overlap(self):
        from run_evidence_observation_diagnostics import select_roles
        from tests.test_evidence_experiment import small_episode
        episodes = []
        for name in ('repr', 'value', 'eval'):
            episode = small_episode(name)
            def renamed(observation, prefix=name):
                return replace(observation, image_hash=prefix+observation.image_hash) if observation.valid else observation
            histories = tuple(tuple(renamed(o) for o in h) for h in episode.initial.histories)
            episodes.append(replace(episode, reference_hash=name+'ref',
                                    initial=replace(episode.initial, histories=histories),
                                    outcomes={k: renamed(o) for k, o in episode.outcomes.items()}))
        source = {'audit': {'scope': 'pilot_location_split_identity_overlap_same_background_reference',
                            'roles': {name: {'episodes': [name]} for name in ('repr', 'value', 'eval')}}}
        training, development, _ = select_roles(source, episodes, False)
        self.assertEqual([e.id for e in training], ['value'])
        self.assertEqual(development, [])
        self.assertEqual([e.id for e in select_roles(source, episodes, True)[1]], ['eval'])
        episodes[2] = replace(episodes[2], reference_hash=episodes[1].reference_hash)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            select_roles(source, episodes, True)

    def test_final_development_evaluation_cannot_change_fitted_parameters(self):
        from dinoexp.evidence_observation_diagnostics import fit_condition
        rows = [record()]
        dev = [record('development', targets=(3., -2., 1., 0.))]
        with tempfile.TemporaryDirectory() as folder:
            first = fit_condition(rows, [], {'dim': 8, 'heads': 2}, Path(folder)/'without',
                                  'direct_joint', steps=2)
            second = fit_condition(rows, dev, {'dim': 8, 'heads': 2}, Path(folder)/'with',
                                   'direct_joint', steps=2)
            self.assertEqual(first['final_fingerprint'], second['final_fingerprint'])
            self.assertEqual(second['development']['states'], 1)
            self.assertEqual(second['development']['rows'][0]['target'], [3., -2., 1., 0.])

    def test_fit_uses_all_states_counts_work_and_pairs_control_initialization(self):
        from dinoexp.evidence_observation_diagnostics import fit_condition
        rows = [record(history=str(i)) for i in range(3)]
        for row in rows:
            row['inputs'].update(future=torch.randn(1, 4, 5, 8),
                                 future_valid=torch.ones(1, 4, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as folder:
            modes = {}
            for condition in ('direct_single', 'direct_joint', 'observation_zero_joint',
                              'observation_real_joint'):
                modes[condition] = fit_condition(rows, [], {'dim': 8, 'heads': 2},
                                                Path(folder)/condition, condition,
                                                steps=6, chunk_size=2, evaluate_every=2)
            self.assertEqual(modes['direct_single']['curve'][-1]['training_state_forwards'], 6)
            self.assertEqual(modes['direct_joint']['curve'][-1]['training_state_forwards'], 18)
            self.assertEqual(modes['direct_joint']['matched_work']['training_state_forwards'], 6)
            self.assertEqual(modes['observation_zero_joint']['initial_fingerprint'],
                             modes['observation_real_joint']['initial_fingerprint'])
            self.assertIsNone(modes['direct_joint']['development'])
            self.assertEqual(modes['direct_joint']['states'], 3)
            with self.assertRaises(FileExistsError):
                fit_condition(rows, [], {'dim': 8, 'heads': 2}, Path(folder)/'direct_joint',
                              'direct_joint', steps=6)

    def test_cli_rejects_output_before_loading_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, 'run_evidence_observation_diagnostics.py',
                                     '--data', 'missing', '--cache', 'missing', '--source-run',
                                     'missing', '--output', folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('output exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
