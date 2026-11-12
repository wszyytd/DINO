"""Pure-data tests: never import or execute neural networks."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dinoexp.fixed_height_data import (audit, batch_arrays, checked_box, source_bundle,
                                       summarize, verify_code)
from dinoexp.region_ab_data import sha, write_json


def fixture(directory):
    root = directory/'data'; root.mkdir()
    cases = []
    for target in ('A', 'B'):
        name = 'group_ref'+target; base = root/name
        (base/'images').mkdir(parents=True); (base/'crops').mkdir()
        cfg = dict(target_identity=target, candidates=[dict(identity=u, blueprint='car'+u) for u in ('A', 'B')],
                   views=[dict(name=f'orbit_{15*t:03}', azimuth=15*t) for t in range(3)])
        rows = []
        for step in (-1, 0, 1, 2):
            view = 'reference' if step == -1 else cfg['views'][step]['name']
            image = 'images/'+view+'.png'
            Image.new('RGB', (32, 24), (120+step, 80, 160)).save(base/image)
            for uid in ([target] if step == -1 else ['A', 'B']):
                crop = 'crops/reference.png' if step == -1 else f'crops/{uid}_{step}.png'
                Image.new('RGB', (12, 12), (120+step, 80, 160)).save(base/crop)
                rows.append(dict(sample_id=f'{uid}_{step}', identity=uid, blueprint='car'+uid,
                                 role='reference' if step == -1 else 'query', sequence=step,
                                 status='captured', image=image, crop=crop, view=view,
                                 frame=step+5, camera_pose='[0,0,8,0,0,0]',
                                 azimuth=0 if step == -1 else step*15, crop_box='[0,0,12,12]'))
        with (base/'metadata.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(base/'run.json', dict(status='complete', config=cfg))
        cases.append(dict(name=name, group='group', pair_kind='same_model_close_colors', config=cfg))
    write_json(root/'collection.json', dict(status='complete', plan=dict(cases=cases, purpose='test fixture')))
    return SimpleNamespace(data=str(root), output=str(directory/'output'), checkpoint_run=str(directory/'source'))


def source_fixture(directory):
    source = directory/'source'; (source/'training').mkdir(parents=True)
    plan = dict(status='complete', teacher_sha256='teacher', backbone_sha256='backbone',
                teacher_feature_config={}, backbone_source={})
    write_json(source/'plan.json', plan); write_json(source/'prepared.json', dict(status='complete'))
    results = []
    for method in ('last', 'mean', 'memory'):
        for seed in (7, 17, 27):
            file = f'training/{method}_{seed}.pt'; (source/file).write_bytes(b'not executed')
            results.append(dict(method=method, seed=seed, checkpoint=file, checkpoint_sha256=sha(source/file)))
    write_json(source/'training/run.json', dict(kind='region_sequence_training_v1', status='complete',
               code={}, config={}, plan_sha256=sha(source/'plan.json'),
               prepared_sha256=sha(source/'prepared.json'), results=results))
    return source


class FixedHeightDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def plan(self):
        args = fixture(self.base)
        with patch('dinoexp.fixed_height_data.source_bundle', return_value=({}, {'sources': {}})):
            plan = audit(args)
        return args, plan

    def arrays(self):
        _, plan = self.plan(); records = {r['id']: r for r in plan['records']}
        refs = {c['reference_sha256']: dict(tokens=np.ones((64, 256), np.float32), valid=np.ones(64, bool))
                for c in plan['cases']}
        features = {o['key']: dict(observation=np.full((16, 256), r['step'], np.float32), valid=np.ones(16, bool))
                    for r in plan['records'] for o in r['observations']}
        return plan['cases'], records, refs, features

    def test_audit_keeps_reference_and_order_without_mask_claims(self):
        _, plan = self.plan()
        self.assertEqual(plan['counts']['cases'], 2); self.assertEqual(len(plan['records']), 6)
        self.assertEqual([r['step'] for r in plan['records'][:3]], [0, 1, 2])
        self.assertEqual(plan['negative_examples'], 0)
        self.assertNotIn('physical_truth', plan['records'][0])

    def test_saved_box_validation(self):
        self.assertEqual(checked_box(dict(crop_box='[1,2,8,9]', sample_id='x'), (10, 10)), [1, 2, 8, 9])
        for bad in ('[-1,0,2,3]', '[0,0,11,3]', '[2,0,2,3]', '[0,0,2.5,3]'):
            with self.assertRaises(ValueError):
                checked_box(dict(crop_box=bad, sample_id='x'), (10, 10))

    def test_missing_query_is_error_not_silent_filter(self):
        args = fixture(self.base); path = Path(args.data)/'group_refA/metadata.csv'
        lines = path.read_text().splitlines(); path.write_text('\n'.join(lines[:-1])+'\n')
        with patch('dinoexp.fixed_height_data.source_bundle', return_value=({}, {'sources': {}})):
            with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
                audit(args)

    def test_same_angle_must_share_rgb(self):
        args = fixture(self.base); path = Path(args.data)/'group_refA/metadata.csv'
        text = path.read_text().replace('images/orbit_000.png,crops/B_0.png', 'images/orbit_015.png,crops/B_0.png')
        path.write_text(text)
        with patch('dinoexp.fixed_height_data.source_bundle', return_value=({}, {'sources': {}})):
            with self.assertRaisesRegex(ValueError, 'one shared camera'):
                audit(args)

    def test_candidate_reversal_moves_features_and_truth_together(self):
        cases, records, refs, features = self.arrays()
        a = batch_arrays(cases, records, refs, features)
        b = batch_arrays(cases, records, refs, features, reverse_candidates=True)
        np.testing.assert_array_equal(a['observation'][:, :, ::-1], b['observation'])
        np.testing.assert_array_equal(a['truth'], 1-b['truth'])
        np.testing.assert_array_equal(a['reference'], b['reference'])

    def test_exact_dedup_keeps_current_view_available(self):
        cases, records, refs, features = self.arrays()
        first, second = [records[key] for key in cases[0]['records'][:2]]
        second['observations'][0]['key'] = first['observations'][0]['key']
        a = batch_arrays(cases, records, refs, features)
        self.assertFalse(a['valid'][0, 1, 0].any()); self.assertTrue(a['current_valid'][0, 1, 0].all())
        self.assertTrue(a['valid'][0, 1, 1].all())

    def test_future_features_do_not_change_prefix(self):
        cases, records, refs, features = self.arrays()
        a = batch_arrays(cases, records, refs, features)
        later = copy.deepcopy(features)
        for case in cases:
            for obj in records[case['records'][-1]]['observations']:
                later[obj['key']]['observation'][:] = -100
        b = batch_arrays(cases, records, refs, later)
        np.testing.assert_array_equal(a['observation'][:, :2], b['observation'][:, :2])

    def test_summary_splits_null_wrong_and_single_view_pairing(self):
        rows = [dict(case_id='case', group='g', pair_kind='kind', target_identity='A', step=i,
                     correct=i == 0, choice_identity=['A', None, 'B'][i], target_ranks_first=i != 2,
                     single_frame_correct=i == 2, last_step=i == 2) for i in range(3)]
        value = summarize(rows); count = value['all_prefixes']
        self.assertEqual([count[k] for k in ('correct', 'null', 'wrong_vehicle')], [1, 1, 1])
        self.assertEqual((count['history_helped'], count['history_harmed']), (1, 1))
        self.assertEqual(value['final_step']['count'], 1)
        self.assertNotIn('physical_target_accuracy', value)

    def test_requires_all_nine_checkpoints(self):
        source = source_fixture(self.base)
        bundle, _ = source_bundle(source); self.assertEqual(len(bundle['checkpoints']), 9)
        run = json.loads((source/'training/run.json').read_text()); run['results'].pop()
        write_json(source/'training/run.json', run)
        with self.assertRaisesRegex(ValueError, 'all nine'):
            source_bundle(source)

    def test_changed_checkpoint_is_rejected(self):
        source = source_fixture(self.base); (source/'training/last_7.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
            source_bundle(source)

    def test_implementation_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'implementation differs'):
            verify_code({'dinoexp/evidence_memory.py': '0'*64})

    def test_saved_predictions_generate_report_without_network(self):
        from dinoexp.fixed_height_report import report
        args, plan = self.plan(); output = Path(args.output)
        (output/'evaluation').mkdir(); write_json(output/'prepared.json', {'status': 'fixture'})
        results = []
        for method in ('last', 'mean', 'memory'):
            for seed in (7, 17, 27):
                rows = []
                for case in plan['cases']:
                    for step, record in enumerate(case['records']):
                        rows.append(dict(case_id=case['id'], group=case['group'], pair_kind=case['pair_kind'],
                                         target_identity=case['target_identity'], step=step, record_id=record,
                                         correct=False, choice_identity=None, target_ranks_first=False,
                                         single_frame_correct=False, last_step=step == len(case['records'])-1))
                file = f'evaluation/{method}_{seed}.json'
                write_json(output/file, {'predictions': rows})
                results.append(dict(method=method, seed=seed, file=file,
                                    sha256=sha(output/file), metrics=summarize(rows)))
        write_json(output/'evaluation/run.json', dict(kind='fixed_height_test_evaluation_v1', status='complete',
                   plan_sha256=sha(output/'plan.json'), prepared_sha256=sha(output/'prepared.json'),
                   source=plan['source'], results=results))
        with patch('dinoexp.fixed_height_report.load_plan', return_value=plan):
            summary = report(args)
        self.assertEqual(summary['methods']['memory']['accuracy_mean'], 0)
        self.assertTrue((output/'report/index.html').is_file())
        self.assertTrue((output/'report/metrics.csv').is_file())
        self.assertIn('同权重单图', (output/'report/report.md').read_text(encoding='utf-8'))
        self.assertNotIn('torch', sys.modules)

    def test_no_torch_imported(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
