"""Pure-data tests for LiVOS vehicle scoring and causal initialization. No torch."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dinoexp.livos_vehicle_data import binary_mask, load_plan, plan, plan_digest, tags_for
from dinoexp.livos_vehicle_eval import inference_inputs
from dinoexp.livos_vehicle_metrics import build_summary, recovery_events, score_frame, summarize
from dinoexp.livos_vehicle_report import report, verify_results
from dinoexp.region_ab_data import sha, write_json


def fixture(root):
    directory = root/'scenes/s'
    (directory/'audit').mkdir(parents=True)
    (directory/'rgb').mkdir()
    (directory/'labels').mkdir()
    records, audit, states = [], [], {}
    for i in range(2):
        sid = f'f{i}'
        image = directory/'rgb'/f'{sid}.png'
        Image.new('RGB', (12, 12), (40+i, 60, 80)).save(image)
        objects = {}
        for uid in ['s:A', 's:B']:
            mask = np.zeros((12, 12), np.uint8)
            if uid == 's:A' and i == 0:
                mask[2:6, 2:6] = 255
            if uid == 's:B':
                mask[7:10, 7:10] = 255
            entry = None
            if mask.any():
                path = directory/'labels'/f'{sid}_{uid[-1]}.png'
                Image.fromarray(mask).save(path)
                entry = dict(path=path.relative_to(root).as_posix(), sha256=sha(path))
            objects[uid] = dict(mask=entry, quality=dict(visible_pixels=int((mask > 0).sum()),
                                                        recognition_quality='usable'))
        records.append(dict(state_id=sid, scene_id='s', annotation_validity='valid',
                            image_path=image.relative_to(root).as_posix(), objects=objects,
                            ignore_mask=None))
        audit.append(dict(state_id=sid, rgb=dict(file_sha256=sha(image))))
        states[sid] = dict(pose=[i*4, 0, 8, -45, 0, 0],
                           edges={('row_plus' if i == 0 else 'row_minus'):
                                  dict(to=f'f{1-i}', cost_s=2)})
    (directory/'frames.jsonl').write_text('\n'.join(map(json.dumps, records)))
    (directory/'audit/observations.jsonl').write_text('\n'.join(map(json.dumps, audit)))
    (directory/'graph.json').write_text(json.dumps(dict(states=states)))
    (directory/'scene.json').write_text('{}')
    (root/'dataset_manifest.json').write_text(json.dumps(dict(status='complete',
          mode='discrete_view_graph', cases=[dict(scene_id='s', path='scenes/s')])))


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((20, 20), bool)
        self.a[2:6, 2:6] = True
        self.b = np.zeros_like(self.a)
        self.b[10:15, 10:15] = True
        self.zero = np.zeros_like(self.a)

    def score(self, pred, a=None, ignore=None):
        return score_frame(pred, {'A': self.a if a is None else a, 'B': self.b},
                           'A', self.zero if ignore is None else ignore)

    def test_perfect_and_miss(self):
        self.assertEqual(self.score(self.a)['iou'], 1.)
        self.assertTrue(self.score(self.a)['success'])
        self.assertEqual(self.score(self.zero)['iou'], 0.)
        self.assertFalse(self.score(self.zero)['success'])

    def test_wrong_vehicle(self):
        metric = self.score(self.b)
        self.assertTrue(metric['wrong_vehicle_overlap'])
        self.assertEqual(metric['other_identity'], 'B')
        self.assertEqual(metric['other_iou'], 1.)

    def test_ignore_pixels_do_not_count(self):
        self.assertEqual(self.score(self.a | self.b, ignore=self.b)['iou'], 1.)
        self.assertFalse(self.score(self.b, ignore=self.b)['wrong_vehicle_overlap'])

    def test_absence_not_perfect_iou(self):
        metric = self.score(self.zero, a=self.zero)
        self.assertIsNone(metric['iou'])
        self.assertFalse(metric['success'])
        self.assertFalse(metric['absent_false_positive'])
        self.assertTrue(self.score(np.ones_like(self.a), a=self.zero)['absent_false_positive'])

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            self.score(np.zeros((3, 3), bool))

    def test_initial_frame_excluded(self):
        rows = [dict(case='c', step=0, **self.score(self.a)),
                dict(case='c', step=1, **self.score(self.zero))]
        self.assertEqual(summarize(rows)['mean_visible_iou'], 0.)
        self.assertEqual(summarize(rows)['initial_steps_excluded'], 1)

    def test_recovery_ends_at_next_absence(self):
        rows = []
        for i, (visible, success) in enumerate([(1, 1), (0, 0), (1, 0),
                                                (1, 1), (0, 0), (1, 0)]):
            rows.append(dict(case='c', step=i, visible=bool(visible), success=bool(success)))
        events = recovery_events(rows)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['delay_steps'], 1)
        self.assertFalse(events[1]['recovered'])
        self.assertIsNone(events[1]['delay_steps'])

    def test_boundary_and_tiny(self):
        self.a[0, 0] = True
        self.assertTrue(self.score(self.a)['boundary'])
        self.assertTrue(self.score(self.a)['tiny'])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'data'
        fixture(self.root)
        self.output = Path(self.temp.name)/'output'
        self.args = SimpleNamespace(data=str(self.root), output=str(self.output),
                                    length=4, stride=1, min_init_pixels=4, size=32, max_cases=4)

    def test_plan_is_graph_adjacent_and_balanced(self):
        result = plan(self.args)
        self.assertEqual(result['counts']['unique_rgb'], 2)
        self.assertEqual({c['target'] for c in result['cases']}, {'s:A', 's:B'})
        for c in result['cases']:
            self.assertTrue(all(a != b for a, b in zip(c['frames'][:-1], c['frames'][1:], strict=True)))
            self.assertGreater(result['frames'][c['frames'][0]]['objects'][c['target']]['pixels'], 0)
        self.assertEqual(load_plan(self.output, self.root), result)
        with self.assertRaises(ValueError):
            plan(self.args)

    def test_changed_source_rejected(self):
        plan(self.args)
        (self.root/'scenes/s/scene.json').write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            load_plan(self.output, self.root)

    def test_tags_do_not_invent_absence_or_boundary(self):
        f = dict(objects={'A': dict(pixels=10, boundary=False),
                          'B': dict(pixels=5, boundary=False)})
        self.assertEqual(tags_for([f, f], 'A'), ['two_vehicles'])

    def test_no_later_mask_read_or_future_dependency(self):
        result = plan(self.args)
        f = result['frames']['s/f0']
        with patch('dinoexp.livos_vehicle_eval.binary_mask', side_effect=AssertionError('GT read')):
            rgb1, mask1 = inference_inputs(self.root, f, 's:A', False, 32)
            mutated = copy.deepcopy(f)
            mutated['objects'] = {'s:A': {'mask': {'path': 'missing-label'}}}
            rgb2, mask2 = inference_inputs(self.root, mutated, 's:A', False, 32)
        self.assertTrue(np.array_equal(rgb1, rgb2))
        self.assertIsNone(mask1)
        self.assertIsNone(mask2)

    def test_initial_mask_resized_with_nearest(self):
        result = plan(self.args)
        rgb, mask = inference_inputs(self.root, result['frames']['s/f0'], 's:A', True, 32)
        self.assertEqual(rgb.shape, (3, 32, 32))
        self.assertEqual(set(np.unique(mask)), {0, 1})
        raw, initial = inference_inputs(self.root, result['frames']['s/f0'], 's:A', True,
                                        32, resize_rgb=False)
        self.assertEqual(raw.shape, (3, 12, 12))
        self.assertEqual(initial.shape, (32, 32))

    def test_preview_is_real_images_without_prediction(self):
        plan(self.args)
        report(self.args, preview=True)
        payload = json.loads((self.output/'preview/data.json').read_text(encoding='utf-8'))
        self.assertTrue(payload['preview'])
        for c in payload['cases']:
            for step in c['steps']:
                self.assertIsNone(step['metrics'])
                self.assertIsNone(step['prediction'])
                self.assertTrue((self.output/'preview'/step['image']).is_file())
        self.assertIn('slice(0,si+1)', (self.output/'preview/index.html').read_text(encoding='utf-8'))


    def test_report_recomputes_saved_masks(self):
        # Synthetic masks in a temporary test fixture, never a model result artifact.
        result = plan(self.args)
        rows = []
        for c in result['cases']:
            for step, key in enumerate(c['frames']):
                frame = result['frames'][key]
                masks = {uid: binary_mask(self.root, o['mask'], frame['size'])
                         for uid, o in frame['objects'].items()}
                pred = np.zeros_like(masks[c['target']])
                path = self.output/f"{c['id']}_{step}.png"
                Image.fromarray(pred.astype(np.uint8)*255).save(path)
                metric = score_frame(pred, masks, c['target'], np.zeros_like(pred))
                rows.append(dict(case=c['id'], step=step, frame=key, target=c['target'],
                                 initial=step == 0, prediction=path.name,
                                 prediction_sha256=sha(path), inference_seconds=0.,
                                 state=dict(persistent_tensor_bytes=0), **metric))
        write_json(self.output/'records.json', rows)
        write_json(self.output/'summary.json', build_summary(rows))
        meta = dict(kind='livos_vehicle_inference_v1', status='complete',
                    plan_sha256=plan_digest(result),
                    artifacts={n: sha(self.output/n) for n in ('records.json', 'summary.json')})
        write_json(self.output/'inference.json', meta)
        report(self.args)
        self.assertTrue((self.output/'report/metrics.csv').is_file())
        rows[-1]['success'] = not rows[-1]['success']
        write_json(self.output/'records.json', rows)
        meta['artifacts']['records.json'] = sha(self.output/'records.json')
        write_json(self.output/'inference.json', meta)
        with self.assertRaisesRegex(ValueError, 'recomputed mask scores'):
            verify_results(self.output, result, self.root)


if __name__ == '__main__':
    unittest.main()
