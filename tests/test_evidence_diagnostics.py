"""Run on the server: includes model forward passes; never run locally by policy."""
import tempfile
import unittest
from pathlib import Path

import torch

from dinoexp.evidence_memory import EvidenceMatcher
from dinoexp.evidence_training import evaluate
from dinoexp.evidence_diagnostic_reports import score_description, metric_counts, write_reports
from dinoexp.evidence_diagnostic_data import enrich
from dinoexp.evidence_diagnostics import (
    UpdateObserver, clone_state, swap_slots, replay_batches, protect_output,
    repeat_trial,
)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_wrong_identity_and_null_are_different(self):
        wrong = score_description([1., 3., 0.], 0, [True, True])
        null = score_description([2., 1., 4.], 0, [True, True])
        self.assertEqual(wrong['error_type'], 'wrong_candidate')
        self.assertEqual(null['error_type'], 'null_instead_of_target')
        self.assertGreater(null['identity_margin'], 0)
        self.assertLess(null['candidate_null_margin'], 0)
        absent = score_description([2., 1., 4.], 0, [False, True])
        self.assertEqual(absent['truth_index'], 2)
        self.assertTrue(absent['correct'])

    def test_unavailable_competitor_does_not_enter_margin(self):
        result = score_description([2., 99., 1.], 0, [True, False])
        self.assertEqual(result['candidate_null_margin'], 1.)
        self.assertIsNone(result['identity_margin'])

    def test_slot_swap_keeps_original_flags_counts_and_tensor(self):
        state = dict(slots=torch.arange(12.).reshape(2, 3, 2, 1),
                     seen=torch.tensor([[True, True, False], [True, False, False]]),
                     count=torch.tensor([[2., 9., 0.], [1., 0., 0.]]))
        original = clone_state(state)
        swapped, eligible = swap_slots(state)
        self.assertEqual(eligible.tolist(), [True, False])
        torch.testing.assert_close(swapped['slots'][0, 0], state['slots'][0, 1])
        torch.testing.assert_close(swapped['slots'][0, 2], state['slots'][0, 2])
        for key in ('seen', 'count'):
            torch.testing.assert_close(swapped[key], state[key])
        for key in state:
            torch.testing.assert_close(original[key], state[key])

    def scene(self):
        return dict(scene_id='s', objects=['s:A', 's:B'],
                    observations=torch.randn(4, 2, 4, 16),
                    valid=torch.tensor([[[True]*4]*2, [[False]*4, [True]*4],
                                        [[False]*4]*2, [[True]*4]*2]),
                    references=torch.randn(2, 4, 16),
                    reference_valid=torch.ones(2, 4, dtype=torch.bool),
                    state_ids=['f0', 'f1', 'f2', 'f3'],
                    image_hashes=['h0', 'h1', 'h2', 'h3'])

    def test_observer_does_not_change_model_and_reports_active_updates(self):
        s = self.scene(); model = EvidenceMatcher(dim=16, heads=2, slots=3).eval()
        x, v = s['observations'][:1], s['valid'][:1]
        expected = model.update(x, v)
        with UpdateObserver(model) as observer:
            actual, details = observer.update(x, v, None)
            self.assertFalse(details[0][0]['gate_applied'])
            _, later = observer.update(x, v, actual)
            self.assertTrue(later[0][0]['gate_applied'])
            self.assertIn('gate_mean', later[0][0])
        torch.testing.assert_close(expected['slots'], actual['slots'])
        self.assertEqual(len(model.updater._forward_hooks), 0)

    def test_replay_matches_existing_evaluator_and_reset_truth_stays_historical(self):
        s = self.scene()
        routes = [dict(scene=0, target=i, frames=[0, 1, 2, 3], actions=[])
                  for i in (0, 1)]
        model = EvidenceMatcher(dim=16, heads=2, slots=3).eval()
        expected = evaluate(model, routes, [s], torch.device('cpu'), 2)['metrics']
        records = list(replay_batches(model, routes, [s], torch.device('cpu'), 2))
        normal = [r for batch in records for r in batch['normal']]
        self.assertEqual(metric_counts(normal), expected)
        hidden = [r for r in normal if r['target_index'] == 0 and r['step'] == 1][0]
        self.assertEqual(hidden['truth_index'], 0)
        reset = [r for batch in records for r in batch['controls']
                 if r['condition'] == 'reset' and r['step'] == 1 and r['target_index'] == 0][0]
        self.assertEqual(reset['intervention']['truth_index'], 0)
        self.assertLessEqual(reset['intervention']['logits'][0], -1e4)

    def test_physical_visibility_does_not_relabel_legacy_predictions(self):
        candidates = [dict(candidate_index=i, legacy_visible=True, visible_gt=False,
                           quality='unknown', bbox_center_normalized=None) for i in range(2)]
        contexts = [dict(frames=[dict(image_path='f.png', candidates=candidates)],
                         references={'s:A': 'ref.png'})]
        row = dict(scene_index=0, frame_index=0, target_id='s:A', target_index=0,
                   candidate_ids=['s:A', 's:B'], prefix_frames=[0], target_seen=True,
                   both_visible=True, truth_index=0)
        result = enrich(row, contexts)
        self.assertEqual(result['truth_index'], 0)
        self.assertEqual(result['physical_truth_index'], 2)
        self.assertTrue(result['truth_disagrees'])

    def test_report_handles_empty_controls_and_includes_prefix(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'data'; root.mkdir()
            output = Path(temp)/'diagnosis'; output.mkdir()
            Image.new('RGB', (4, 4)).save(root/'frame.png')
            Image.new('RGB', (4, 4)).save(root/'ref.png')
            row = dict(split='probe', mode='current', seed=7, task_id=0, step=0,
                       scene_id='s', frame_id='f', path_id='p', prefix_id='prefix',
                       target_id='A', candidate_ids=['A', 'B'], target_index=0,
                       target_visible=True, both_visible=True, target_seen=True, is_final=True,
                       quality_group='both_usable', target_horizontal='left',
                       predicted_id='A', image_path='frame.png', reference_path='ref.png',
                       candidate_metadata=[dict(candidate_id='A', quality='usable', mask_path=None)],
                       **score_description([2., 0., -1.], 0, [True, True]))
            write_reports(output, root, [row], [], [], [], [], 1)
            self.assertTrue((output/'steps.jsonl').is_file())
            self.assertIn('prefixes', (output/'error_types.csv').read_text(encoding='utf-8-sig'))
            self.assertIn('完整观察前缀', (output/'examples/index.html').read_text(encoding='utf-8'))

    def test_repeat_noop_and_new_frame_do_not_mutate_source(self):
        s = self.scene(); model = EvidenceMatcher(dim=16, heads=2, slots=3).eval()
        x, v = s['observations'][:1], s['valid'][:1]
        state = model.update(x, v); original = clone_state(state)
        result = repeat_trial(model, state, x, v, s['references'][:1],
                              s['reference_valid'][:1], 0, [True, True], 4,
                              (s['observations'][3:4], s['valid'][3:4]))
        self.assertLessEqual(result['dedup_max_probability_change'], 1e-6)
        self.assertIn('after_next', result)
        for key in state:
            torch.testing.assert_close(state[key], original[key])

    def test_output_cannot_modify_inputs_or_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'run'; source.mkdir()
            for bad in (source, source/'new', root):
                with self.assertRaises(ValueError):
                    protect_output(bad, [source])
            protect_output(root/'new-diagnosis', [source])


if __name__ == '__main__':
    unittest.main()
