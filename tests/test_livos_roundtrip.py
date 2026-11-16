"""Pure-data route constraints and unchanged shared LiVOS runner compatibility."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from dinoexp.livos_roundtrip_routes import (
    MOVES, REVERSE, YAWS, choose_routes, load_roundtrip_plan, prepare, render_report,
)
from dinoexp.livos_vehicle_data import code_hashes, load_plan
from dinoexp.region_ab_data import sha, write_json
from dinoexp.region_sequence_routes import validate_route


def fixture(root):
    directory = root/'scenes/s'
    for child in ['rgb', 'labels', 'audit']:
        (directory/child).mkdir(parents=True)
    poses = [[0, 0, 8, -45, 0, 0], [4, 0, 8, -45, 0, 0],
             [4, 0, 8, -45, 45, 0], [8, 0, 8, -45, 45, 0], [12, 0, 8, -45, 45, 0]]
    actions = ['row_plus', 'yaw_right', 'row_plus', 'row_plus']
    states = {f'f{i}': dict(pose=p, edges={}) for i, p in enumerate(poses)}
    for i, action in enumerate(actions):
        cost = .5 if action in YAWS else 2.
        states[f'f{i}']['edges'][action] = dict(to=f'f{i+1}', cost_s=cost)
        states[f'f{i+1}']['edges'][REVERSE[action]] = dict(to=f'f{i}', cost_s=cost)
    records, audits, frames = [], [], {}
    for i, state in enumerate(states):
        image = directory/'rgb'/f'{state}.png'
        Image.new('RGB', (16, 16), (40+i, 60, 80)).save(image)
        objects = {}
        for uid, lo in [('s:A', 2), ('s:B', 9)]:
            mask = np.zeros((16, 16), np.uint8)
            mask[lo:lo+4, lo:lo+4] = 255
            path = directory/'labels'/f'{state}_{uid[-1]}.png'
            Image.fromarray(mask).save(path)
            objects[uid] = dict(mask=dict(path=path.relative_to(root).as_posix(), sha256=sha(path)),
                                quality=dict(visible_pixels=16, recognition_quality='usable'))
        records.append(dict(state_id=state, annotation_validity='valid',
                            image_path=image.relative_to(root).as_posix(), objects=objects, ignore_mask=None))
        audits.append(dict(state_id=state, rgb=dict(file_sha256=sha(image))))
        frames[state] = dict(scene='s', state=state, objects={
            uid: dict(pixels=16, boundary=False) for uid in objects})
    (directory/'frames.jsonl').write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
    (directory/'audit/observations.jsonl').write_text('\n'.join(map(json.dumps, audits)), encoding='utf-8')
    write_json(directory/'graph.json', dict(states=states))
    write_json(directory/'scene.json', dict(objects={
        's:A': dict(center=[2, 1, 0]), 's:B': dict(center=[7, 1, 0])}))
    write_json(root/'dataset_manifest.json', dict(status='complete', mode='discrete_view_graph',
               cases=[dict(scene_id='s', path='scenes/s')]))
    return dict(states=states), frames


class RoundtripTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'data'
        self.graph, self.frames = fixture(self.root)
        self.args = SimpleNamespace(data=str(self.root), output=str(Path(self.temp.name)/'output'),
                                    min_view_pixels=4, max_routes=1, max_outward=17, beam_width=128, size=32)

    def routes(self):
        return choose_routes('s', self.graph, self.frames, min_pixels=4)

    def test_return_is_exact_observed_reverse_without_duplicate_endpoint(self):
        routes = self.routes()
        self.assertEqual(routes, self.routes())
        self.assertEqual(len(routes), 1)
        r = routes[0]
        self.assertEqual(r['outward_frames'], 5)
        self.assertEqual(r['states'], ['f0', 'f1', 'f2', 'f3', 'f4', 'f3', 'f2', 'f1', 'f0'])
        self.assertEqual(r['replay_of_step'], [None]*5+[3, 2, 1, 0])
        self.assertEqual(r['phase_by_step'], ['outbound']*4+['turnaround']+['returning']*4)
        self.assertEqual(r['distance_m'], 24)
        self.assertEqual(r['unique_positions'], 4)
        self.assertTrue(validate_route(r, self.graph, self.frames))

    def test_yaw_is_separated_and_cannot_change_height(self):
        for route in self.routes():
            actions = route['actions']
            self.assertFalse(any(a in YAWS and b in YAWS
                                 for a, b in zip(actions[:-1], actions[1:], strict=True)))
            self.assertLessEqual(route['outward_yaw_steps'], 4)
            self.assertTrue(all(p[2] == 8 for p in route['poses']))
        self.graph['states']['f2']['pose'][2] = 9
        self.assertEqual(self.routes(), [])

    def test_outward_has_no_revisit_or_translation_backtracking(self):
        for r in self.routes():
            n = r['outward_frames']
            self.assertEqual(len(set(r['states'][:n])), n)
            seen = {tuple(r['poses'][0][:3])}
            for i, action in enumerate(r['actions'][:n-1]):
                pos = tuple(r['poses'][i+1][:3])
                if action in MOVES:
                    self.assertNotIn(pos, seen)
                seen.add(pos)

    def test_no_missing_vehicle_or_unobserved_reverse_edge(self):
        self.frames['f2']['objects']['s:B']['pixels'] = 0
        self.assertEqual(self.routes(), [])
        self.frames['f2']['objects']['s:B']['pixels'] = 16
        del self.graph['states']['f2']['edges']['yaw_left']
        for route in self.routes():
            pairs = set(zip(route['states'][:-1], route['states'][1:], strict=True))
            self.assertNotIn(('f1', 'f2'), pairs)
            self.assertNotIn(('f2', 'f1'), pairs)

    def test_yaw_edge_must_be_45_degrees(self):
        self.graph['states']['f2']['pose'][4] = 180
        self.assertEqual(self.routes(), [])

    def test_limits_and_identity_change_rejected(self):
        with self.assertRaises(ValueError):
            choose_routes('s', self.graph, self.frames, max_outward=2)
        self.frames['f2']['objects']['s:C'] = self.frames['f2']['objects'].pop('s:B')
        with self.assertRaisesRegex(ValueError, 'consistent vehicle identities'):
            self.routes()

    def test_common_loader_same_frames_for_both_targets_and_preview(self):
        plan = prepare(self.args)
        self.assertEqual(load_roundtrip_plan(self.args), plan)
        self.assertEqual(load_plan(Path(self.args.output), self.root), plan)
        self.assertEqual(plan['code'], code_hashes())
        self.assertEqual(plan['counts']['total_steps'], 18)
        self.assertEqual(plan['counts']['scoring_steps'], 16)
        self.assertEqual(plan['counts']['selected_unique_rgb'], 5)
        self.assertEqual(plan['cases'][0]['frames'], plan['cases'][1]['frames'])
        render_report(self.args, preview=True)
        directory = Path(self.args.output)/'preview'
        data = json.loads((directory/'data.json').read_text(encoding='utf-8'))
        self.assertEqual(data['cases'][0]['steps'][4]['phase'], 'turnaround')
        self.assertEqual(data['cases'][0]['steps'][5]['phase'], 'returning')
        for case in data['cases']:
            for step in case['steps']:
                self.assertIsNone(step['metrics'])
                self.assertIsNone(step['prediction'])
                self.assertTrue((directory/step['image']).is_file())
        page = (directory/'index.html').read_text(encoding='utf-8')
        self.assertIn('const PHASES=', page)
        self.assertIn('返程按原路重访', page)
        with self.assertRaises(ValueError):
            prepare(self.args)

    def test_changed_source_hash_or_data_rejected(self):
        plan = prepare(self.args)
        broken = copy.deepcopy(plan)
        broken['route_selection_code']['dinoexp/livos_roundtrip_routes.py'] = 'changed'
        write_json(Path(self.args.output)/'plan.json', broken)
        with self.assertRaisesRegex(ValueError, 'route implementation changed'):
            load_roundtrip_plan(self.args)
        write_json(Path(self.args.output)/'plan.json', plan)
        (self.root/'scenes/s/scene.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            load_roundtrip_plan(self.args)


if __name__ == '__main__':
    unittest.main()
