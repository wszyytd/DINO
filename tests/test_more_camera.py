from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from dinoexp.more_placement import apply_camera_radius
from dinoexp.scene import capture_view_pose


class TestCameraRadius(unittest.TestCase):
    def setUp(self):
        config = json.loads((Path(__file__).resolve().parents[1]
                             / 'configs/two_candidates.json').read_text())
        config.update(anchor_spawn_index=82, separation=12, query_yaw_mode='anchor')
        config['views'] = [dict(name='side', azimuth=90, elevation=25, distance=26),
                           dict(name='descend', azimuth=90, elevation=55, distance=40),
                           dict(name='overview', azimuth=90, elevation=80, distance=70)]
        cases = [dict(name='test_' + target, split='test', capture_group='test_group',
                      config=dict(deepcopy(config), target_identity=target)) for target in ('A', 'B')]
        other = dict(name='train_done', split='train', config=deepcopy(config))
        self.manifest = dict(plan=dict(cases=[other, *cases]))

    def test_synced_side_pose_moves_out_of_far_lateral_position_without_changing_range(self):
        result = apply_camera_radius(self.manifest, {'train_done'}, 10)
        old_view = self.manifest['plan']['cases'][1]['config']['views'][0]
        new_view = result['plan']['cases'][1]['config']['views'][0]
        center, anchor_yaw = [-27.558, -61.238, 1.972], 0.596735
        old = capture_view_pose(center, anchor_yaw, 0, old_view, is_reference=False,
                                query_yaw_mode='anchor')
        new = capture_view_pose(center, anchor_yaw, 0, new_view, is_reference=False,
                                query_yaw_mode='anchor')
        self.assertGreater(old[1], -40)
        self.assertLess(new[1], -50)
        self.assertAlmostEqual(math.dist(new[:2], center[:2]), 10)
        self.assertAlmostEqual(math.dist(new[:3], center), 26)
        self.assertEqual(new_view['name'], 'side_raised')
        self.assertGreater(new_view['elevation'], 67)

    def test_changes_both_pending_queries_together_and_preserves_reference_and_completed_config(self):
        original = deepcopy(self.manifest)
        result = apply_camera_radius(self.manifest, {'train_done'}, 10)
        left, right = [c['config'] for c in result['plan']['cases'] if c['split'] == 'test']
        self.assertEqual(left['views'], right['views'])
        for view in left['views']:
            self.assertLessEqual(view['distance'] * math.cos(math.radians(view['elevation'])), 10.000001)
        self.assertEqual(left['reference'], original['plan']['cases'][1]['config']['reference'])
        self.assertEqual(left['candidates'], original['plan']['cases'][1]['config']['candidates'])
        self.assertEqual(left['anchor_spawn_index'], 82)
        self.assertEqual(left['separation'], 12)
        self.assertEqual(result['plan']['cases'][0], original['plan']['cases'][0])
        self.assertEqual(self.manifest, original)
        self.assertEqual(len(result['camera_revisions']), 1)

    def test_reapplying_same_radius_does_not_rename_views_or_add_revisions_again(self):
        once = apply_camera_radius(self.manifest, {'train_done'}, 10)
        twice = apply_camera_radius(once, {'train_done'}, 10)
        self.assertEqual(once, twice)

    def test_cannot_change_a_completed_test_reference_or_use_invalid_radius(self):
        with self.assertRaisesRegex(ValueError, 'completed'):
            apply_camera_radius(self.manifest, {'train_done', 'test_A'}, 10)
        for radius in (0, -1, float('nan'), float('inf')):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                apply_camera_radius(self.manifest, {'train_done'}, radius)


if __name__ == '__main__':
    unittest.main()
