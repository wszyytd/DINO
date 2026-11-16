"""Pure-data checks of fixed-heading follow routes and shared evaluation compatibility."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from dinoexp.livos_follow_routes import choose_routes, load_follow_plan, prepare, render_report
from dinoexp.livos_vehicle_data import code_hashes, load_plan
from dinoexp.region_ab_data import sha, write_json


def fixture(root):
    directory = root/'scenes/s'
    (directory/'audit').mkdir(parents=True)
    (directory/'rgb').mkdir()
    (directory/'labels').mkdir()
    records, audits, states = [], [], {}
    for i in range(3):
        state = f'f{i}'
        image = directory/'rgb'/f'{state}.png'
        Image.new('RGB', (16, 16), (40+i, 60, 80)).save(image)
        objects = {}
        for uid, region in [('s:A', (2, 6)), ('s:B', (9, 13))]:
            mask = np.zeros((16, 16), np.uint8)
            lo, hi = region
            mask[lo:hi, lo:hi] = 255
            path = directory/'labels'/f'{state}_{uid[-1]}.png'
            Image.fromarray(mask).save(path)
            objects[uid] = dict(mask=dict(path=path.relative_to(root).as_posix(), sha256=sha(path)),
                                quality=dict(visible_pixels=16, recognition_quality='usable'))
        records.append(dict(state_id=state, annotation_validity='valid',
                            image_path=image.relative_to(root).as_posix(), objects=objects,
                            ignore_mask=None))
        audits.append(dict(state_id=state, rgb=dict(file_sha256=sha(image))))
        edges = {}
        if i > 0:
            edges['row_minus'] = dict(to=f'f{i-1}', cost_s=2)
        if i < 2:
            edges['row_plus'] = dict(to=f'f{i+1}', cost_s=2)
        states[state] = dict(pose=[4*i, 0, 8, -45, 0, 0], edges=edges)
    (directory/'frames.jsonl').write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
    (directory/'audit/observations.jsonl').write_text('\n'.join(map(json.dumps, audits)), encoding='utf-8')
    write_json(directory/'graph.json', dict(states=states))
    write_json(directory/'scene.json', dict(objects={
        's:A': dict(center=[1, 1, 0]), 's:B': dict(center=[6, 1, 0])}))
    write_json(root/'dataset_manifest.json', dict(status='complete', mode='discrete_view_graph',
               cases=[dict(scene_id='s', path='scenes/s')]))
    frames = {s: dict(state=s, scene='s', objects={
        uid: dict(pixels=16, boundary=False) for uid in ['s:A', 's:B']}) for s in states}
    return dict(states=states), frames


class FollowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'data'
        self.graph, self.frames = fixture(self.root)
        self.args = SimpleNamespace(data=str(self.root), output=str(Path(self.temp.name)/'output'),
                                    min_view_pixels=4, max_routes=6, size=32)

    def routes(self):
        return choose_routes('s', self.graph, self.frames, min_pixels=4)

    def test_translation_no_revisit_and_deterministic(self):
        routes = self.routes()
        self.assertEqual(routes, self.routes())
        self.assertEqual(len(routes), 1)
        route = routes[0]
        self.assertEqual(len(route['states']), 3)
        self.assertEqual(len(set(route['states'])), 3)
        self.assertEqual({tuple(p[2:]) for p in route['poses']}, {(8, -45, 0, 0)})
        self.assertTrue(set(route['actions']) <= {'row_plus', 'row_minus', 'col_plus', 'col_minus'})
        self.assertEqual(route['distance_m'], 8)
        self.assertLessEqual(route['turns'], 1)

    def test_never_bridge_missing_or_tiny_vehicle(self):
        self.frames['f1']['objects']['s:B']['pixels'] = 3
        self.assertEqual(self.routes(), [])
        self.frames['f1']['objects']['s:B']['pixels'] = 0
        self.assertEqual(self.routes(), [])

    def test_both_vehicles_unclipped_at_start(self):
        self.frames['f0']['objects']['s:B']['boundary'] = True
        routes = self.routes()
        self.assertTrue(routes)
        self.assertTrue(all(not any(o['boundary'] for o in self.frames[r['states'][0]]['objects'].values())
                            for r in routes))
        self.assertEqual(routes[0]['states'][0], 'f2')

    def test_yaw_edge_is_not_a_translation(self):
        edge = self.graph['states']['f1']['edges'].pop('row_plus')
        self.graph['states']['f1']['edges']['yaw_right'] = edge
        self.graph['states']['f2']['pose'][4] = 45
        self.assertEqual(self.routes(), [])

    def test_same_position_different_id_is_not_a_new_view(self):
        self.graph['states']['f2']['pose'] = list(self.graph['states']['f0']['pose'])
        self.assertEqual(self.routes(), [])

    def test_altitude_stays_constant(self):
        self.graph['states']['f2']['pose'][2] = 10
        self.assertEqual(self.routes(), [])

    def test_bad_limits_and_identity_change_rejected(self):
        with self.assertRaises(ValueError):
            choose_routes('s', self.graph, self.frames, min_pixels=0)
        self.frames['f2']['objects']['s:C'] = self.frames['f2']['objects'].pop('s:B')
        with self.assertRaisesRegex(ValueError, 'consistent vehicle identities'):
            self.routes()

    def test_shared_plan_loading_and_real_preview_without_predictions(self):
        result = prepare(self.args)
        output = Path(self.args.output)
        self.assertEqual(load_follow_plan(self.args), result)
        self.assertEqual(load_plan(output, self.root), result)
        self.assertEqual(result['code'], code_hashes())
        self.assertEqual(result['counts']['selected_tasks'], 2)
        self.assertEqual(result['counts']['total_steps'], 6)
        self.assertEqual(result['counts']['scoring_steps'], 4)
        self.assertEqual(result['counts']['selected_unique_rgb'], 3)
        a, b = result['cases']
        self.assertEqual(a['frames'], b['frames'])
        self.assertNotEqual(a['target'], b['target'])
        render_report(self.args, preview=True)
        page = (output/'preview/index.html').read_text(encoding='utf-8')
        self.assertIn('每步移动 4 米，拍摄朝向固定', page)
        self.assertNotIn('或方向相差 45 度', page)
        payload = json.loads((output/'preview/data.json').read_text(encoding='utf-8'))
        for case in payload['cases']:
            for step in case['steps']:
                self.assertIsNone(step['prediction'])
                self.assertIsNone(step['metrics'])
                self.assertTrue((output/'preview'/step['image']).is_file())
        overview = (output/'routes.html').read_text(encoding='utf-8')
        self.assertIn('arrow_0', overview)
        self.assertIn('原地旋转 0 次', overview)
        with self.assertRaises(ValueError):
            prepare(self.args)

    def test_follow_source_hash_and_source_data_changes_rejected(self):
        result = prepare(self.args)
        broken = copy.deepcopy(result)
        broken['route_selection_code']['dinoexp/livos_follow_routes.py'] = 'changed'
        write_json(Path(self.args.output)/'plan.json', broken)
        with self.assertRaisesRegex(ValueError, 'follow route implementation changed'):
            load_follow_plan(self.args)
        write_json(Path(self.args.output)/'plan.json', result)
        (self.root/'scenes/s/scene.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            load_follow_plan(self.args)


if __name__ == '__main__':
    unittest.main()
