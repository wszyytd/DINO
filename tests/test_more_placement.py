from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from dinoexp.more_placement import apply_placement, placement_options, probe_placement


class PlacementTests(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]
                          / 'configs/two_candidates.json').read_text())
        cfg['anchor_spawn_index'] = 9
        self.cases = []
        for target in ('A', 'B'):
            config = deepcopy(cfg)
            config['target_identity'] = target
            self.cases.append(dict(name='test_ref' + target, split='test',
                                   capture_group='test_group', config=config))
        other = deepcopy(cfg)
        other['anchor_spawn_index'] = 42
        train = dict(name='train_done', split='train', capture_group='train_group', config=other)
        self.manifest = dict(scope='reduced_pilot', plan=dict(cases=[train, *self.cases],
            split_anchors=dict(train=[42], val=[15], test=[9])),
            original_plan=dict(split_anchors=dict(train=[42], val=[15], test=[9, 82, 14])))
        self.completed = {'train_done'}

    def test_options_keep_test_reservations_and_try_larger_spacing_without_changing_saved_plan(self):
        before = deepcopy(self.manifest)
        options = placement_options(self.manifest, self.completed)
        self.assertEqual(options[0], dict(anchor_spawn_index=82, separation=8))
        self.assertIn(dict(anchor_spawn_index=82, separation=12), options)
        self.assertEqual({p['anchor_spawn_index'] for p in options}, {9, 82, 14})
        self.assertEqual(self.manifest, before)

    def test_new_placement_updates_both_references_and_records_previous_values_only_for_test(self):
        result = apply_placement(self.manifest, self.completed,
                                 dict(anchor_spawn_index=82, separation=12, attempts=[]))
        test_cases = [c for c in result['plan']['cases'] if c['split'] == 'test']
        self.assertTrue(all(c['config']['anchor_spawn_index'] == 82
                            and c['config']['separation'] == 12 for c in test_cases))
        self.assertEqual(result['plan']['cases'][0], self.manifest['plan']['cases'][0])
        self.assertEqual(result['original_plan'], self.manifest['original_plan'])
        self.assertEqual(result['plan']['split_anchors']['test'], [82])
        self.assertEqual(result['placement_revisions'][0]['previous']['anchor_spawn_index'], 9)

    def test_reference_group_cannot_move_after_either_reference_was_completed(self):
        with self.assertRaisesRegex(ValueError, 'completed'):
            placement_options(self.manifest, self.completed | {'test_refA'})
        with self.assertRaisesRegex(ValueError, 'reserved'):
            apply_placement(self.manifest, self.completed,
                            dict(anchor_spawn_index=42, separation=12, attempts=[]))

    def test_live_probe_tries_pair_spacing_and_cleans_every_owned_actor(self):
        active, destroyed = [], []
        class Actor:
            def __init__(self, x):
                self.x = x
            def set_simulate_physics(self, value):
                pass
            def destroy(self):
                active.remove(self)
                destroyed.append(self)
                return True
        class Blueprint:
            def __init__(self, name):
                self.id = name
            def set_attribute(self, key, value):
                pass
            def has_attribute(self, key):
                return key in ('color', 'role_name')
        def spawn(bp, transform):
            x = transform.location.x
            if active and abs(x - active[0].x) < 12:
                return None
            actor = Actor(x)
            active.append(actor)
            return actor
        world = SimpleNamespace(get_blueprint_library=lambda:
            SimpleNamespace(find=lambda name: Blueprint(name)), try_spawn_actor=spawn,
            get_settings=lambda: SimpleNamespace(synchronous_mode=False), wait_for_tick=lambda timeout: None)
        carla = SimpleNamespace(Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda location, rotation: SimpleNamespace(location=location, rotation=rotation))
        catalog = dict(spawn_points=[dict(index=i, pose=[i * 100, 0, 0, 0, 0, 0])
                                    for i in (9, 82, 14)])
        result = probe_placement(carla, world, self.manifest, self.completed, catalog)
        self.assertEqual(result['anchor_spawn_index'], 82)
        self.assertEqual(result['separation'], 12)
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(active, [])
        self.assertEqual(len(destroyed), 3)

    def test_cleanup_failure_stops_the_probe_instead_of_trying_more_positions(self):
        calls = []
        actor = SimpleNamespace(set_simulate_physics=lambda value: None, destroy=lambda: False)
        bp = SimpleNamespace(set_attribute=lambda key, value: None, has_attribute=lambda key: False)
        def spawn(blueprint, transform):
            calls.append(transform)
            return actor if len(calls) == 1 else None
        world = SimpleNamespace(get_blueprint_library=lambda: SimpleNamespace(find=lambda name: bp),
            get_settings=lambda: SimpleNamespace(synchronous_mode=False), try_spawn_actor=spawn)
        carla = SimpleNamespace(Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda location, rotation: (location, rotation))
        catalog = dict(spawn_points=[dict(index=i, pose=[i * 100, 0, 0, 0, 0, 0])
                                    for i in (9, 82, 14)])
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            probe_placement(carla, world, self.manifest, self.completed, catalog)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
