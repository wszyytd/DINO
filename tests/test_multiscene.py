"""Determinism, capture failure recovery and route export without CARLA or models."""
import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dinoexp import multiscene_capture as capture
from dinoexp import multiscene_routes as export
from dinoexp.multiscene_plan import case_for, choose_backgrounds, dry_plan, slots_for, validate_config
from test_livos_roundtrip import fixture

REPO = Path(__file__).resolve().parents[1]


def configuration():
    cfg = json.loads((REPO/'configs/multiscene_roundtrip_v1.json').read_text(encoding='utf-8'))
    cfg['sampling'].update(backgrounds=2, train_backgrounds=1, layouts_per_background=1,
                           anchor_spawn_indices=[0, 1], max_attempts=2)
    cfg['route'].update(min_view_pixels=4, min_outward=3, max_outward=5)
    return cfg


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.poses = [[0, 0, .6, 0, 0, 0], [120, 0, .6, 0, 90, 0]]

    def test_layout_reproducible_separated_and_angles_independent(self):
        backgrounds = choose_backgrounds(self.config, self.poses)
        slots = slots_for(self.config)
        first = case_for(self.config, backgrounds[0], slots[0], 1)
        self.assertEqual(first, case_for(self.config, backgrounds[0], slots[0], 1))
        self.assertNotEqual(first, case_for(self.config, backgrounds[0], slots[0], 2))
        second_before = case_for(self.config, backgrounds[1], slots[1], 1)
        for attempt in range(1, 10):
            case_for(self.config, backgrounds[0], slots[0], attempt)
        self.assertEqual(second_before, case_for(self.config, backgrounds[1], slots[1], 1))
        a, b = first['layout']['placements']
        self.assertTrue(9 <= math.dist(a[:2], b[:2]) <= 11)
        self.assertTrue(all(-180 <= p[2] <= 180 for p in (a, b)))
        self.assertNotEqual(a[2], b[2])
        self.assertEqual({b['split'] for b in backgrounds}, {'train', 'val'})
        self.assertEqual(len({b['group'] for b in backgrounds}), 2)

    def test_automatic_backgrounds_repeat_and_reject_overlap_or_steep_anchor(self):
        self.config['sampling'].update(anchor_spawn_indices=None)
        poses = [[x, 0, .6, 0, 0, 0] for x in (0, 100, 220, 350)]
        a = choose_backgrounds(self.config, poses)
        self.assertEqual(a, choose_backgrounds(self.config, poses))
        self.assertGreaterEqual(math.dist(a[0]['anchor_pose'][:2], a[1]['anchor_pose'][:2]), 60)
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            choose_backgrounds(self.config, [[x, 0, .6, 0, 0, 0] for x in (0, 20)])
        self.config['sampling']['anchor_spawn_indices'] = [0, 1]
        poses[1][3] = 20
        with self.assertRaisesRegex(ValueError, 'too steep'):
            choose_backgrounds(self.config, poses)

    def test_capture_provenance_portable_across_windows_linux_newlines(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source.py'
            with patch.object(capture, 'REPO', root), patch.object(capture, 'SOURCES', ('source.py',)):
                source.write_bytes(b'first = 1\nsecond = 2\n')
                original = capture.provenance()
                source.write_bytes(b'first = 1\r\nsecond = 2\r\n')
                self.assertEqual(original, capture.provenance())
                source.write_bytes(b'first = 9\r\nsecond = 2\r\n')
                self.assertNotEqual(original, capture.provenance())

    def test_invalid_config_and_dry_plan_counts(self):
        bad = copy.deepcopy(self.config)
        bad['capture']['candidates'] = []
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            validate_config(bad)
        bad = copy.deepcopy(self.config)
        bad['route']['min_outward'] = 6
        with self.assertRaisesRegex(ValueError, 'min_outward'):
            validate_config(bad)
        result = dry_plan(self.config)
        self.assertEqual(result['images_if_first_attempts_pass'], 400)
        self.assertEqual(result['maximum_grid_images'], 800)
        self.assertFalse(result['carla_connected'])
        self.assertFalse(result['captured'])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base/'capture'
        self.config = configuration()
        self.poses = [[0, 0, .6, 0, 0, 0], [120, 0, .6, 0, 90, 0]]
        fake_map = SimpleNamespace(get_spawn_points=lambda: self.poses)
        fake_library = SimpleNamespace(find=lambda name: name)
        self.world = SimpleNamespace(get_map=lambda: fake_map, get_blueprint_library=lambda: fake_library)
        self.graph, _ = fixture(self.base/'fixture')
        records = [json.loads(x) for x in (self.base/'fixture/scenes/s/frames.jsonl').read_text().splitlines()]
        self.raw = {r['state_id']: r for r in records}
        for row in self.raw.values():
            for obj in row['objects'].values():
                obj['quality']['touches_boundary'] = False
        self.scene = dict(anchor=[0, 0, .6, 0, 0, 0], objects={
            's:A': dict(pose=[0, 0, .9, 0, 0, 0]), 's:B': dict(pose=[10, 0, .9, 0, 0, 0])})
        self.seen = []
        for target, replacement in [('preflight', lambda *_: {}), ('pose_values', lambda x: x),
                                    ('read_layout', lambda *_: (self.scene, self.graph, self.raw))]:
            mock = patch.object(capture, target, side_effect=replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def simulate_capture(self, _carla, _world, _versions, _config, root, case):
        self.seen.append(copy.deepcopy(case))
        directory = root/case['path']
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'capture_marker.txt').write_text(json.dumps(case), encoding='utf-8')
        capture.atomic_json(directory/'scene.json', dict(cleanup_errors=[]))

    def collect(self, callback=None, resume=False):
        with patch.object(capture, 'capture_layout', side_effect=callback or self.simulate_capture):
            return capture.collect(self.config, self.root, None, self.world, ['0.9.15']*2, resume=resume)

    def test_rejected_attempt_preserved_and_not_published(self):
        def callback(*args):
            self.simulate_capture(*args)
            if args[-1]['randomization']['attempt'] == 1:
                raise ValueError('simulated blocked placement')
        manifest, report = self.collect(callback)
        self.assertEqual(report['accepted_scenes'], 2)
        self.assertEqual(report['by_split'], {'train': 1, 'val': 1})
        self.assertTrue(all(c['randomization']['attempt'] == 2 for c in manifest['cases']))
        for slot in manifest['slots']:
            self.assertEqual([a['status'] for a in slot['attempts']], ['rejected', 'accepted'])
            self.assertTrue((self.root/slot['attempts'][0]['case']['path']/'capture_marker.txt').is_file())
        self.assertEqual(capture.validate_collection(self.root), report)
        self.assertFalse((self.root/'.capture.lock').exists())
        with self.assertRaises(FileExistsError):
            self.collect()

    def test_keyboard_interrupt_resumes_new_attempt_without_replacing_old_data(self):
        def interrupt(*args):
            self.simulate_capture(*args)
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.collect(interrupt)
        old = capture.check_manifest(self.root)
        first = old['slots'][0]['attempts'][0]
        self.assertEqual(first['status'], 'interrupted')
        path = self.root/first['case']['path']/'capture_marker.txt'
        content = path.read_bytes()
        manifest, _ = self.collect(resume=True)
        self.assertEqual(manifest['slots'][0]['attempts'][1]['status'], 'accepted')
        self.assertEqual(content, path.read_bytes())
        self.assertEqual(manifest['cases'][0]['randomization']['attempt'], 2)

    def test_hard_interruption_running_journal_is_advanced_on_resume(self):
        backgrounds = choose_backgrounds(self.config, self.poses)
        manifest = capture.create_manifest(self.root, self.config, backgrounds, ['0.9.15']*2, self.poses)
        slot = manifest['slots'][0]
        case = case_for(self.config, backgrounds[0], slot, 1)
        self.simulate_capture(None, None, None, None, self.root, case)
        slot['attempts'].append(dict(case=case, status='running'))
        capture.atomic_json(self.root/'dataset_manifest.json', manifest)
        result, _ = self.collect(resume=True)
        self.assertEqual([a['status'] for a in result['slots'][0]['attempts']], ['interrupted', 'accepted'])

    def test_partial_collection_explicit_and_attempts_bounded(self):
        def callback(*args):
            self.simulate_capture(*args)
            if args[-1]['split'] == 'train':
                raise ValueError('simulated no route')
        manifest, report = self.collect(callback)
        self.assertEqual(manifest['collection_status'], 'partial')
        self.assertEqual((report['accepted_scenes'], report['exhausted_scenes']), (1, 1))
        self.assertEqual(len(self.seen), 3)
        self.assertEqual([c['split'] for c in manifest['cases']], ['val'])
        self.assertEqual(capture.validate_collection(self.root), report)
        self.collect(resume=True)
        self.assertEqual(len(self.seen), 3)

    def test_no_success_is_never_published_as_usable_dataset(self):
        def callback(*args):
            self.simulate_capture(*args)
            raise ValueError('all attempts fail')
        manifest, report = self.collect(callback)
        self.assertEqual(report['accepted_scenes'], 0)
        self.assertEqual(manifest['status'], 'incomplete')
        self.assertEqual(len(self.seen), 4)
        with self.assertRaisesRegex(ValueError, 'no published'):
            capture.validate_collection(self.root)

    def test_cleanup_failure_stops_instead_of_retrying(self):
        def callback(*args):
            self.simulate_capture(*args)
            capture.atomic_json(self.root/args[-1]['path']/'scene.json', dict(cleanup_errors=['not destroyed']))
            raise RuntimeError('simulated capture failure')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            self.collect(callback)
        self.assertEqual(len(self.seen), 1)
        self.assertFalse((self.root/'.capture.lock').exists())

    def test_changed_config_versions_and_tampered_success_are_rejected(self):
        self.collect()
        cfg = copy.deepcopy(self.config)
        cfg['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'original configuration'):
            capture.check_manifest(self.root, cfg)
        with self.assertRaisesRegex(ValueError, 'version/map spawn'):
            capture.collect(self.config, self.root, None, self.world, ['different']*2, resume=True)
        manifest = capture.check_manifest(self.root)
        route_path = self.root/manifest['cases'][0]['path']/'roundtrip_routes.json'
        capture.atomic_json(route_path, {})
        with self.assertRaisesRegex(ValueError, 'route index changed'):
            self.collect(resume=True)

    def test_journal_status_and_lock_exclusivity(self):
        self.collect()
        manifest = capture.check_manifest(self.root)
        manifest['slots'][0]['attempts'][0]['status'] = 'invented'
        capture.atomic_json(self.root/'dataset_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'attempt status'):
            capture.check_manifest(self.root)
        with capture.capture_lock(self.root):
            with self.assertRaises(FileExistsError):
                with capture.capture_lock(self.root):
                    self.fail('concurrent capture entered')
            self.assertTrue((self.root/'.capture.lock').exists())
        self.assertFalse((self.root/'.capture.lock').exists())

    def test_route_rejects_too_short_or_tilted_settled_car(self):
        case = case_for(self.config, choose_backgrounds(self.config, self.poses)[0], slots_for(self.config)[0], 1)
        cfg = copy.deepcopy(self.config)
        cfg['route'].update(min_outward=6, max_outward=6)
        with self.assertRaisesRegex(ValueError, 'no qualifying'):
            capture.routes_for(case, self.scene, self.graph, self.raw, cfg)
        self.scene['objects']['s:A']['pose'][5] = 20
        with self.assertRaisesRegex(ValueError, 'tilt'):
            capture.routes_for(case, self.scene, self.graph, self.raw, self.config)


class ExportTests(unittest.TestCase):
    def test_shorter_additional_route_filtered_by_same_acceptance_policy(self):
        with TemporaryDirectory() as temporary:
            root, output = Path(temporary)/'data', Path(temporary)/'output'
            fixture(root)
            manifest = capture.read_json(root/'dataset_manifest.json')
            cfg = configuration()
            cfg['route'].update(min_outward=5, max_routes=2)
            manifest.update(config=cfg, collection_status='complete', slots=[{}])
            manifest['cases'][0].update(group='background-test', split='train', randomization=dict(attempt=1))
            capture.atomic_json(root/'dataset_manifest.json', manifest)
            args = SimpleNamespace(data=str(root), output=str(output), size=32)
            validation = dict(accepted_scenes=1, accepted_backgrounds=1, by_split=dict(train=1))
            with patch.object(export, 'validate_collection', return_value=validation):
                plan = export.prepare(args)
            self.assertEqual(len(plan['routes']), 1)
            self.assertEqual(plan['counts']['outward_frame_counts'], [5])
            self.assertEqual(plan['counts']['selected_tasks'], 2)
            self.assertEqual(plan['counts']['total_steps'], 18)
            self.assertEqual(plan['config']['max_cases'], 2)
            self.assertEqual(plan, export.load_plan(args))

    def test_shared_loader_preview_metadata_and_code_changes(self):
        with TemporaryDirectory() as temporary:
            root, output = Path(temporary)/'data', Path(temporary)/'output'
            fixture(root)
            manifest = capture.read_json(root/'dataset_manifest.json')
            manifest.update(config=configuration(), collection_status='complete', slots=[{}])
            manifest['cases'][0].update(group='background-test', split='val', randomization=dict(attempt=2))
            capture.atomic_json(root/'dataset_manifest.json', manifest)
            args = SimpleNamespace(data=str(root), output=str(output), size=32)
            validation = dict(accepted_scenes=1, accepted_backgrounds=1, by_split=dict(val=1))
            # Raw capture calibration is covered by the original auditor; this fixture tests export only.
            with patch.object(export, 'validate_collection', return_value=validation):
                plan = export.prepare(args)
            self.assertEqual(plan, export.load_plan(args))
            self.assertTrue(all(c['group'] == 'background-test' and c['split'] == 'val' for c in plan['cases']))
            self.assertEqual(plan['cases'][0]['frames'], plan['cases'][1]['frames'])
            self.assertEqual(plan['counts']['selected_unique_rgb'], 5)
            self.assertEqual(plan['counts']['total_steps'], 18)
            export.render_report(args, preview=True)
            data = capture.read_json(output/'preview/data.json')
            self.assertEqual(data['cases'][0]['split'], 'val')
            self.assertEqual(data['cases'][0]['steps'][5]['phase'], 'returning')
            self.assertTrue(all(s['prediction'] is None for c in data['cases'] for s in c['steps']))
            with patch.object(export, 'provenance', return_value={}):
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    export.load_plan(args)


if __name__ == '__main__':
    unittest.main()
