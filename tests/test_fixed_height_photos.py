import json
import math
import csv
import tempfile
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import unittest

from dinoexp.scene import capture_view_pose


class FixedHeightTests(unittest.TestCase):
    def test_live_inventory_dry_run_is_read_only_and_does_not_replace_missing_assets(self):
        from collect_fixed_height_photos import main
        models = ['vehicle.dodge.charger', 'vehicle.lincoln.mkz']
        blueprints = [SimpleNamespace(id=name, has_attribute=lambda key: True,
                                     get_attribute=lambda key: SimpleNamespace(as_int=lambda: 4)) for name in models]
        world = SimpleNamespace(get_blueprint_library=lambda: SimpleNamespace(filter=lambda query: blueprints))
        catalog = dict(map='test', existing_vehicles=[], spawn_points=[
            dict(index=i, pose=[i*100, 0, 0, 0, 0, 0]) for i in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'photos'
            with patch.dict('sys.modules', carla=SimpleNamespace()), \
                    patch('capture.connect', return_value=(None, world, ['x', 'x'])), \
                    patch('capture.inventory', return_value=catalog), \
                    patch('sys.argv', ['collect', '--output', str(output), '--dry-run']):
                with redirect_stdout(io.StringIO()) as text:
                    main()
                self.assertEqual(json.loads(text.getvalue())['expected_crops'], 588)
                self.assertFalse(output.exists())
                blueprints.pop()
                with self.assertRaisesRegex(ValueError, 'unavailable'):
                    main()
                self.assertFalse(output.exists())

    def test_capture_verification_rejects_camera_height_drift(self):
        from collect_fixed_height_photos import verify_case
        from PIL import Image
        cfg = dict(target_identity='A', fixed_camera_height_m=8,
                   candidates=[dict(identity='A'), dict(identity='B')], views=[dict(name='v')])
        case = dict(config=cfg, anchor_z=2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (16, 16), (100, 100, 100)).save(root/'crop.png')
            (root/'run.json').write_text(json.dumps(dict(status='complete', config=cfg, map='m')))
            for height in (10, 11):
                with (root/'metadata.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['sample_id', 'identity', 'role', 'status', 'crop', 'view', 'camera_pose'])
                    writer.writeheader()
                    for name, identity, role in [('ref', 'A', 'reference'), ('a', 'A', 'query'), ('b', 'B', 'query')]:
                        writer.writerow(dict(sample_id=name, identity=identity, role=role, status='captured',
                                             crop='crop.png', view='v', camera_pose=json.dumps([0, 0, height, 0, 0, 0])))
                if height == 10:
                    verify_case(root, case, 'm')
                else:
                    with self.assertRaisesRegex(ValueError, 'height'):
                        verify_case(root, case, 'm')

    def test_fixed_world_camera_height_for_reference_and_queries(self):
        for center in ([0, 0, 1], [4, 0, 2]):
            for angle in range(0, 360, 15):
                pose = capture_view_pose(center, 30, 80,
                                         dict(azimuth=angle, elevation=30, distance=20),
                                         is_reference=center[0] == 4, camera_z=8)
                self.assertAlmostEqual(pose[2], 8)
                self.assertAlmostEqual(math.hypot(pose[0]-center[0], pose[1]-center[1]),
                                       20*math.cos(math.radians(30)))

    def test_plan_counts_height_angles_and_nonidentical_pairs(self):
        from dinoexp.fixed_height_photos import build_plan
        base = json.loads(Path('configs/two_candidates.json').read_text())
        anchors = [dict(index=i, pose=[i*100, 0, 0, 0, 0, 0]) for i in range(2)]
        plan = build_plan(base, ['vehicle.dodge.charger', 'vehicle.lincoln.mkz'], anchors)
        self.assertEqual(len(plan['cases']), 12)
        self.assertEqual(plan['expected_crops'], 588)
        self.assertEqual(len({c['group'] for c in plan['cases']}), 6)
        for case in plan['cases']:
            cfg = case['config']
            self.assertEqual(cfg['fixed_camera_height_m'], 8)
            self.assertEqual([v['azimuth'] for v in cfg['views']], list(range(0, 360, 15)))
            self.assertEqual(len({(c['blueprint'], c['color']) for c in cfg['candidates']}), 2)
        with self.assertRaises(ValueError):
            build_plan(base, ['vehicle.dodge.charger'], anchors)
        with self.assertRaises(ValueError):
            build_plan(base, ['vehicle.dodge.charger', 'vehicle.lincoln.mkz'], anchors, height=float('nan'))


if __name__ == '__main__':
    unittest.main()
