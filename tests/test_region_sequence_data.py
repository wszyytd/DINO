"""Pure numpy/geometry tests. This module never loads a neural model or torch."""
import ast
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

from dinoexp.region_ab_data import read_json
from dinoexp.region_geometry import DEFAULTS, propose
from dinoexp.region_sequence_data import (DEFAULT_CONFIG, sequence_arrays,
                                          training_weights, validate_config)
from dinoexp.region_sequence_geometry import DEFAULT_HINTS, group_regions
from dinoexp.region_sequence_metrics import aggregate, control_summary, summarize
from dinoexp.region_sequence_routes import build_routes, validate_route
from dinoexp.search_grid import build_graph

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    records = {}
    features = {key: dict(observation=np.full((16, 256), n, np.float32), valid=np.ones(16, bool))
                for key, n in [('a', 1), ('b', 2), ('future', 3)]}
    refs = {'ref': dict(tokens=np.zeros((64, 256), np.float32), valid=np.ones(64, bool))}
    for t in range(5):
        obs = []
        if t in (1, 2):
            obs = [dict(identity='A', key='a', write_allowed=True)]
        elif t == 3:
            obs = [dict(identity='B', key='b', write_allowed=True)]
        records[str(t)] = dict(observations=obs, physical_visible={'A': t < 3, 'B': t == 3})
    route = dict(id='route/A', records=list(records), identities=['A', 'B'],
                 target_identity='A', reference_sha256='ref', split='train')
    return records, [route], refs, features


class GeometryTests(unittest.TestCase):
    def record(self, heat, **cfg):
        return dict(size=[100, 100], spatial=heat.tolist(),
                    **propose(heat, (100, 100), dict(DEFAULTS['proposal'], min_side=2, expansion=1, **cfg)))

    def test_fragments_group_once_and_expand_to_full_visible_extent(self):
        heat = np.zeros((10, 10)); heat[2, 2] = 1; heat[3, 5] = 1
        record = self.record(heat)
        self.assertEqual(len(record['candidates']), 2)
        mask = np.zeros((100, 100), bool); mask[20:40, 20:60] = True
        grouped = group_regions(record, {'A': mask})
        self.assertEqual(len(grouped['observations']), 1)
        obs = grouped['observations'][0]
        self.assertEqual(obs['source_candidate_ids'], [0, 1])
        self.assertEqual(obs['visible_pixels'], 800)
        self.assertEqual(obs['seed_pixels'], 200)
        self.assertLessEqual(obs['box'][0], 20); self.assertGreaterEqual(obs['box'][2], 60)
        self.assertTrue(obs['write_allowed'])

    def test_expanded_box_overlap_is_not_teacher_active_support(self):
        heat = np.zeros((10, 10)); heat[0, 0] = 1
        record = self.record(heat)
        record['candidates'][0]['box'] = [0, 0, 100, 100]
        mask = np.zeros((100, 100), bool); mask[50:70, 50:70] = True
        grouped = group_regions(record, {'A': mask})
        self.assertEqual(grouped['observations'], [])
        self.assertEqual(grouped['unassigned_candidate_ids'], [0])

    def test_shared_component_can_be_split_by_explicit_mask_assistance(self):
        heat = np.zeros((10, 10)); heat[2, 2] = 1
        a = np.zeros((100, 100), bool); a[20:30, 20:25] = True
        b = np.zeros_like(a); b[20:30, 25:30] = True
        grouped = group_regions(self.record(heat), {'A': a, 'B': b})
        self.assertEqual([o['identity'] for o in grouped['observations']], ['A', 'B'])
        self.assertEqual(grouped['mask_split_candidate_ids'], [0])

    def test_tiny_region_is_logged_but_cannot_write(self):
        heat = np.zeros((10, 10)); heat[2, 2] = 1
        mask = np.zeros((100, 100), bool); mask[20:22, 20:22] = True
        cfg = dict(DEFAULT_HINTS, min_seed_pixels=1)
        obs = group_regions(self.record(heat), {'A': mask}, cfg)['observations'][0]
        self.assertFalse(obs['write_allowed']); self.assertIsNotNone(obs['rejection'])

    def test_absent_and_uniform_heat_add_no_vehicle(self):
        mask = np.ones((100, 100), bool)
        self.assertEqual(group_regions(self.record(np.ones((10, 10))), {'A': mask})['observations'], [])
        self.assertEqual(group_regions(self.record(np.eye(10)), {'A': np.zeros_like(mask)})['observations'], [])

    def test_invalid_masks_and_unknown_components_fail(self):
        record = self.record(np.eye(10))
        with self.assertRaises(ValueError):
            group_regions(record, {'A': np.zeros((10, 10), bool)})
        record['candidates'][0]['merged_from'] = [999]
        with self.assertRaises(ValueError):
            group_regions(record, {'A': np.zeros((100, 100), bool)})


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph(3, 4, [0, 0, 8], -45, move_speed=2, turn_speed=45, hold_s=1)
        self.frames = [dict(scene='scene', state=sid, split='train' if state['pose'][1] <= 0 else 'probe')
                       for sid, state in self.graph['states'].items()]

    def test_deterministic_target_independent_patrols_keep_every_graph_edge(self):
        routes = build_routes('scene', 'train', self.graph, self.frames)
        self.assertEqual(routes, build_routes('scene', 'train', self.graph, list(reversed(self.frames))))
        self.assertTrue(routes)
        allowed = {f['state'] for f in self.frames if f['split'] == 'train'}
        covered, actions = set(), set()
        for r in routes:
            self.assertEqual(len(r['states']), 24)
            self.assertTrue(validate_route(r, self.graph, allowed))
            covered.update(r['states']); actions.update(r['actions'])
        self.assertEqual(covered, allowed)
        self.assertIn('yaw_right', actions)
        self.assertTrue(actions & {'row_plus', 'row_minus', 'col_plus', 'col_minus'})
        self.assertNotIn('hold_capture', actions)

    def test_invalid_edge_pose_cost_and_split_are_rejected(self):
        route = build_routes('scene', 'train', self.graph, self.frames)[0]
        allowed = set(route['states'])
        for field, value in [('actions', ['stop']+route['actions'][1:]),
                             ('costs_s', [999]+route['costs_s'][1:]),
                             ('poses', [[0]*6]+route['poses'][1:])]:
            with self.assertRaises(ValueError):
                validate_route(dict(route, **{field: value}), self.graph, allowed)
        with self.assertRaises(ValueError):
            validate_route(route, self.graph, set())

    def test_isolated_capture_does_not_create_a_fake_moving_route(self):
        self.assertEqual(build_routes('scene', 'train', self.graph, self.frames[:1]), [])
        with self.assertRaises(ValueError):
            build_routes('scene', 'train', self.graph, self.frames, length=1)


class SequenceDataTests(unittest.TestCase):
    def test_complete_config_and_saved_default_match(self):
        validate_config(DEFAULT_CONFIG)
        self.assertEqual(read_json(ROOT/'configs/region_sequence_mask_assisted.json'), DEFAULT_CONFIG)
        for section, key, value in [('routes', 'length', 1), ('routes', 'stride', 0),
                                     ('hints', 'padding_fraction', 1), ('training', 'seeds', [7, 7]),
                                     ('training', 'balance_classes', 1), ('training', 'heads', 3)]:
            cfg = copy.deepcopy(DEFAULT_CONFIG); cfg[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_truth_keeps_physical_miss_and_history_separate(self):
        records, routes, refs, features = fixture()
        arrays = sequence_arrays(routes, records, refs, features)
        self.assertEqual(arrays['truth'].tolist(), [[2, 0, 0, 0, 0]])
        self.assertEqual(arrays['physical_truth'].tolist(), [[0, 0, 0, 0, 0]])
        self.assertEqual(arrays['current_truth'].tolist(), [[2, 0, 0, 2, 2]])
        self.assertEqual(arrays['valid'][0].any(-1).tolist(),
                         [[False, False], [True, False], [False, False], [False, True], [False, False]])
        self.assertTrue(arrays['current_valid'][0, 2, 0].all())
        self.assertFalse(arrays['physical_now'][0, 4].any())
        self.assertEqual(arrays['observation'].shape, (1, 5, 2, 16, 256))

    def test_dedup_is_per_car_and_can_be_disabled(self):
        records, routes, refs, features = fixture()
        normal = sequence_arrays(routes, records, refs, features)
        repeated = sequence_arrays(routes, records, refs, features, dedup=False)
        self.assertFalse(normal['valid'][0, 2].any())
        self.assertTrue(repeated['valid'][0, 2, 0].all())
        np.testing.assert_array_equal(normal['truth'], repeated['truth'])
        np.testing.assert_array_equal(normal['observation'], repeated['observation'])

    def test_candidate_order_is_fixed_for_the_whole_route(self):
        records, routes, refs, features = fixture()
        for seed in range(5):
            a = sequence_arrays(routes, records, refs, features, rng=np.random.default_rng(seed))
            ai, bi = a['identities'][0].index('A'), a['identities'][0].index('B')
            self.assertEqual(a['truth'][0, 1:].tolist(), [ai]*4)
            self.assertTrue((a['observation'][0, 1:3, ai] == 1).all())
            self.assertTrue((a['observation'][0, 3, bi] == 2).all())

    def test_future_changes_do_not_change_earlier_inputs_or_truth(self):
        records, routes, refs, features = fixture()
        before = sequence_arrays(routes, records, refs, features)
        records['4']['observations'] = [dict(identity='A', key='future', write_allowed=True)]
        after = sequence_arrays(routes, records, refs, features)
        for key in ('observation', 'valid', 'current_valid', 'truth', 'physical_truth'):
            np.testing.assert_array_equal(before[key][:, :4], after[key][:, :4])
        self.assertTrue(after['valid'][0, 4, 0].all())

    def test_rejected_observation_never_silently_becomes_accepted(self):
        records, routes, refs, features = fixture()
        records['0']['observations'] = [dict(identity='A', key='missing', write_allowed=False)]
        self.assertEqual(sequence_arrays(routes, records, refs, features)['truth'][0, 0], 2)
        features['a']['valid'][:] = False
        with self.assertRaisesRegex(ValueError, 'never silently relabel'):
            sequence_arrays(routes, records, refs, features)

    def test_duplicate_car_fragments_fail_instead_of_writing_twice(self):
        records, routes, refs, features = fixture()
        records['1']['observations'] *= 2
        with self.assertRaisesRegex(ValueError, 'grouped before batching'):
            sequence_arrays(routes, records, refs, features)

    def test_balancing_uses_only_train_prefixes(self):
        records, routes, _, _ = fixture()
        w = training_weights(routes, records)
        self.assertEqual(w['counts'], {'Null': 1, 'A': 4})
        self.assertEqual(w['weights'], {'Null': 2.5, 'A': .625})
        self.assertEqual(training_weights(routes, records, False)['weights'], {'Null': 1., 'A': 1.})
        with self.assertRaises(ValueError):
            training_weights([dict(routes[0], split='probe')], records)

    def test_network_whitelist_excludes_teacher_scores_truth_and_ids(self):
        tree = ast.parse((ROOT/'dinoexp/region_sequence_training.py').read_text(encoding='utf-8'))
        value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'NETWORK_FIELDS' for t in n.targets))
        self.assertEqual(value, ('observation', 'valid', 'current_valid', 'reference', 'reference_valid'))

    def test_this_test_suite_has_not_imported_torch(self):
        self.assertFalse(any(n == 'torch' or n.startswith('torch.') for n in sys.modules))


class MetricsTests(unittest.TestCase):
    def prediction(self, truth, physical, choice, *, history=False):
        return dict(accepted_truth=truth, physical_truth=physical, choice_identity=choice,
                    seen_identities=['A', 'B'] if truth else [], rank_identity='A',
                    current_identities=[] if history else ['A'], target_identity='A')

    def test_physical_miss_does_not_disappear_in_accepted_accuracy(self):
        metrics = summarize([self.prediction(None, 'A', None), self.prediction('A', 'A', 'A', history=True)])
        self.assertEqual(metrics['accepted_accuracy'], 1)
        self.assertEqual(metrics['physical_target_accuracy'], .5)
        self.assertEqual(metrics['counts']['physical_seen_without_accepted_target'], 1)
        self.assertEqual(metrics['history_only_target_accuracy'], 1)
        self.assertEqual(metrics['always_null_accepted_accuracy'], .5)
        self.assertEqual(metrics['always_null_physical_accuracy'], 0)

    def test_empty_denominators_stay_missing_not_zero(self):
        self.assertIsNone(summarize([])['accepted_accuracy'])
        self.assertIsNone(summarize([self.prediction(None, None, None)])['two_seen_target_accuracy'])

    def test_controls_are_compared_on_the_same_eligible_rows(self):
        row = dict(control='reset', eligible=True, normal_correct=False, control_correct=True,
                   normal_choice_identity='B', choice_identity='A', target_probability=.8,
                   normal_target_probability=.2)
        summary = control_summary([row, dict(row, eligible=False)])['reset']
        self.assertEqual(summary['pairs'], 1)
        self.assertEqual(summary['normal_wrong_control_right'], 1)
        self.assertEqual(summary['choice_flips'], 1)
        self.assertAlmostEqual(summary['max_target_probability_change'], .6)

    def test_aggregate_keeps_seeds_and_population_std(self):
        metric = summarize([self.prediction('A', 'A', 'A')])
        run = dict(results=[dict(seed=s, method=m, splits={'probe': {'metrics': metric}})
                            for s in (7, 17, 27) for m in ('last', 'mean', 'memory')])
        value = aggregate(run)['probe']['memory']['accepted_target_accuracy']
        self.assertEqual(value['mean'], 1); self.assertEqual(value['population_std'], 0)
        self.assertEqual(len(value['per_seed']), 3)


if __name__ == '__main__':
    unittest.main()
