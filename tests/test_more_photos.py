import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dinoexp.more_photos import build_plan, choose_models, select_anchors
from dinoexp.scene import candidate_poses


class MorePhotoTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((Path(__file__).resolve().parents[1]
                               / 'configs/two_candidates.json').read_text())
        self.models = [f'vehicle.test.car{i}' for i in range(8)]
        self.points = [{'index': i, 'pose': [i * 200, 0, 0, 0, 0, 0]} for i in range(12)]

    def test_plan_has_balanced_references_and_disjoint_models_and_locations(self):
        plan = build_plan(self.base, self.models, select_anchors(self.points))
        self.assertEqual(len(plan['cases']), 48)
        self.assertEqual(plan['expected_query_crops'], 960)
        self.assertEqual(plan['expected_reference_crops'], 48)
        ids, locations = {}, {}
        for split in ('train', 'val', 'test'):
            cases = [c for c in plan['cases'] if c['split'] == split]
            self.assertEqual({c['config']['target_identity'] for c in cases}, {'A', 'B'})
            ids[split] = {v['blueprint'] for c in cases for v in c['config']['candidates']}
            locations[split] = {c['config']['anchor_spawn_index'] for c in cases}
            self.assertTrue(all(len({v['color'] for v in c['config']['candidates']}) == 1
                                for c in cases))
        self.assertFalse(ids['train'] & (ids['val'] | ids['test']))
        self.assertFalse(ids['val'] & ids['test'])
        self.assertFalse(locations['train'] & (locations['val'] | locations['test']))
        # Each physical condition is observed with both reference identities in the same split.
        groups = {}
        for c in plan['cases']:
            groups.setdefault(c['capture_group'], []).append(c)
        self.assertTrue(all(len(c) == 2 for c in groups.values()))
        self.assertTrue(all(len({c['split'] for c in cases}) == 1 for cases in groups.values()))

    def test_model_selection_never_invents_unavailable_assets(self):
        available = self.models + ['vehicle.dodgecop.charger', 'vehicle.test.ambulance']
        chosen = choose_models(available)
        self.assertEqual(len(chosen), 8)
        self.assertTrue(set(chosen) <= set(available))
        self.assertFalse(any('cop' in m or 'ambulance' in m for m in chosen))
        with self.assertRaises(ValueError):
            choose_models(self.models[:5])
        with self.assertRaises(ValueError):
            choose_models(self.models, self.models[:7] + ['vehicle.missing.car'])

    def test_actual_seven_asset_inventory_keeps_splits_disjoint_and_all_models_used(self):
        available = ['vehicle.dodge.charger', 'vehicle.lincoln.mkz', 'vehicle.nissan.patrol',
                     'vehicle.carlacola.actors', 'vehicle.fuso.mitsubishi',
                     'vehicle.mini.cooper', 'vehicle.taxi.ford']
        chosen = choose_models(available)
        self.assertEqual(set(chosen), set(available))
        plan = build_plan(self.base, chosen, select_anchors(self.points))
        splits = plan['split_models']
        self.assertEqual([len(splits[s]) for s in ('train', 'val', 'test')], [3, 2, 2])
        self.assertFalse(set(splits['train']) & (set(splits['val']) | set(splits['test'])))
        self.assertFalse(set(splits['val']) & set(splits['test']))
        self.assertEqual({c['blueprint'] for case in plan['cases']
                          for c in case['config']['candidates']}, set(available))
        self.assertTrue(all(len(case['config']['candidates']) == 2 for case in plan['cases']))
        self.assertEqual(plan['expected_query_crops'] + plan['expected_reference_crops'], 1260)

    def test_six_models_can_still_form_three_disjoint_two_model_splits(self):
        chosen = choose_models(self.models[:6], self.models[:6])
        plan = build_plan(self.base, chosen, select_anchors(self.points))
        self.assertEqual([len(plan['split_models'][s]) for s in ('train', 'val', 'test')], [2, 2, 2])
        self.assertEqual(plan['expected_query_crops'] + plan['expected_reference_crops'], 756)

    def test_locations_must_be_spatially_separated(self):
        selected = select_anchors(self.points)
        self.assertEqual(len(selected), 9)
        close = [dict(index=i, pose=[i, 0, 0, 0, 0, 0]) for i in range(12)]
        with self.assertRaises(ValueError):
            select_anchors(close)

    def test_vehicle_heading_can_change_without_moving_the_slots(self):
        normal = candidate_poses([0, 0, 0, 0, 0, 0], 8)
        varied = candidate_poses([0, 0, 0, 0, 0, 0], 8, [0, 180])
        self.assertEqual(normal[0], varied[0])
        self.assertEqual(normal[1][:3], varied[1][:3])
        self.assertEqual(varied[1][4], 180)
        with self.assertRaises(ValueError):
            candidate_poses([0, 0, 0, 0, 0, 0], 8, [float('nan'), 0])

    def test_swapping_reference_does_not_change_query_camera_geometry(self):
        from dinoexp.scene import capture_view_pose
        plan = build_plan(self.base, self.models, select_anchors(self.points))
        first, second = plan['cases'][:2]
        for a, b in zip(first['config']['views'], second['config']['views'], strict=True):
            poses = []
            for case, view in ((first, a), (second, b)):
                cfg = case['config']
                target = next(c for c in cfg['candidates'] if c['identity'] == cfg['target_identity'])
                poses.append(capture_view_pose([10, 20, 2], 0, target['yaw_offset'], view,
                    is_reference=False, query_yaw_mode=cfg['query_yaw_mode']))
            self.assertEqual(poses[0], poses[1])

    def test_collection_stops_on_failure_and_preserves_incomplete_status(self):
        import subprocess
        from collect_match_photos import capture_plan
        plan = build_plan(self.base, self.models, select_anchors(self.points))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'new'
            runner = Mock(side_effect=subprocess.CalledProcessError(2, ['capture']))
            with self.assertRaises(subprocess.CalledProcessError):
                capture_plan(plan, root, runner=runner)
            self.assertEqual(runner.call_count, 1)
            saved = json.loads((root / 'collection.json').read_text())
            self.assertEqual(saved['status'], 'incomplete')
            self.assertEqual(saved['completed_scenes'], 0)
            self.assertFalse((root / 'pairs_train.json').exists())

    def test_complete_synthetic_flow_exports_portable_balanced_split_manifests(self):
        import csv
        from PIL import Image
        from collect_match_photos import capture_plan
        plan = build_plan(self.base, self.models, select_anchors(self.points))
        counter = 0

        def fake_capture(command, check):
            nonlocal counter
            cfg = json.loads(Path(command[command.index('--config') + 1]).read_text())
            destination = Path(command[command.index('--output') + 1])
            destination.mkdir()
            (destination / 'run.json').write_text(json.dumps(dict(status='complete')))
            records = [('reference', cfg['target_identity'], 'reference', 'reference')]
            records += [(f'{c["identity"]}_{v["name"]}', c['identity'], 'query', v['name'])
                        for v in cfg['views'] for c in cfg['candidates']]
            rows = []
            for sid, identity, role, view in records:
                counter += 1
                crop = sid + '.png'
                Image.new('RGB', (28, 14), (80 + counter % 120,
                          100 + counter // 120, 120)).save(destination / crop)
                rows.append(dict(sample_id=sid, identity=identity, role=role, view=view,
                                 status='captured', crop=crop))
            with (destination / 'metadata.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            root = Path(tmp) / 'new'
            capture_plan(plan, root, runner=fake_capture)
            self.assertEqual(json.loads((root / 'collection.json').read_text())['status'], 'complete')
            for split, count in [('train', 480), ('val', 240), ('test', 240)]:
                pairs = json.loads((root / f'pairs_{split}.json').read_text())['pairs']
                self.assertEqual(len(pairs), count)
                self.assertEqual(sum(p['label'] for p in pairs), count // 2)
                self.assertTrue(all(p['reference'].startswith(split + '/') for p in pairs))


if __name__ == '__main__':
    unittest.main()
