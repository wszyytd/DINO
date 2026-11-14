"""Pure-data query protocol tests: no torch, no encoder execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from dinoexp.region_ab_data import sha, write_json, digest_json
from dinoexp.evidence_query_data import (
    DEFAULT_CONFIG, build_manifest, validate_manifest, pack_query_batch,
    context_digest, load_feature, render_sample, load_old_observation, training_weights,
)


def fixture(root):
    data, source = root / 'data', root / 'source'
    data.mkdir(); source.mkdir()
    records, routes, hashes = [], [], {}
    for split in ('train', 'probe'):
        ids = ['A', 'B']  # Explicitly a shared-identity smoke fixture.
        names = []
        for t in range(12):
            name = f'{split}_{t}.png'
            pix = np.random.default_rng(t + (100 if split == 'probe' else 0)).integers(
                0, 256, (32, 64, 3), dtype=np.uint8)
            Image.fromarray(pix).save(data / name); hashes[name] = sha(data / name)
            rid = f'{split}/{t}'
            obs = [dict(identity=uid, key=f'{split}_{t}_{uid}',
                        box=[j*32, 0, (j+1)*32, 32], write_allowed=True)
                   for j, uid in enumerate(ids) if t != 0 or uid == 'A']
            records.append(dict(id=rid, scene='s', frame=str(t), image=name,
                                image_sha256=hashes[name], split=split,
                                reference_sha256='ref', identities=ids,
                                observations=obs))
            names.append(rid)
        routes.append(dict(id=split+'/route', scene='s', split=split,
                           reference_sha256='ref', identities=ids,
                           records=names[:8], actions=['move']*7))
    write_json(source/'observations.json', records)
    write_json(source/'routes.json', routes)
    write_json(source/'plan.json', dict(status='complete', sources=hashes,
               artifacts={n: sha(source/n) for n in ('observations.json', 'routes.json')},
               teacher_sha256='teacher', backbone_sha256='backbone',
               teacher_feature_config={}, backbone_source={}))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(lengths=[4], stride=4, perturbations={'clean': 1., 'blur': 0., 'cover': 0.})
    return data, source, cfg


def fake_cache(root, manifest):
    prepared = dict(status='complete', observations={}, queries={},
                    encoder_signature='enc', manifest_digest=digest_json(manifest))
    for ep in manifest['episodes']:
        for step in ep['steps']:
            for role, entries in [('observations', step['observations']), ('queries', step['queries'])]:
                for item in entries:
                    key = item['feature_key']
                    if key in prepared[role]:
                        continue
                    n = 16 if role == 'observations' else 64
                    path = root / f'{key}.npz'
                    np.savez_compressed(path, tokens=np.ones((n, 256), np.float32),
                                        valid=np.ones(n, bool))
                    prepared[role][key] = dict(file=path.name, sha256=sha(path),
                                              role=role, sample_id=item['sample_id'],
                                              encoder_signature='enc')
    return prepared


class QueryDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.source, self.cfg = fixture(self.root)
        self.m = build_manifest(self.source, self.data, self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def test_seeded_manifest(self):
        self.assertEqual(self.m, build_manifest(self.source, self.data, self.cfg))
        self.assertGreater(len(self.m['episodes']), 0)
        self.assertTrue(self.m['split_audit']['shared_identities'])
        bad = dict(self.cfg, split_mode='identity_disjoint')
        with self.assertRaisesRegex(ValueError, 'identity'):
            build_manifest(self.source, self.data, bad)

    def test_source_frame_disjoint(self):
        ep = self.m['episodes'][0]
        source_hashes = {s['source_rgb_sha256'] for s in ep['steps']}
        for st in ep['steps']:
            for q in st['queries']:
                self.assertNotIn(self.m['samples'][q['sample_id']]['source_rgb_sha256'], source_hashes)
        for field in ('source_rgb_sha256', 'parent_source_sha256'):
            bad = copy.deepcopy(self.m)
            q = bad['episodes'][0]['steps'][0]['queries'][0]
            bad['samples'][q['sample_id']][field] = next(iter(source_hashes))
            with self.assertRaisesRegex(ValueError, 'leak|source'):
                validate_manifest(bad)

    def test_query_changes_do_not_change_context(self):
        ep = copy.deepcopy(self.m['episodes'][0])
        old = context_digest(ep)
        ep['steps'][0]['queries'] = list(reversed(ep['steps'][0]['queries']))
        self.assertEqual(context_digest(ep), old)

    def test_view_group_missing(self):
        with self.assertRaisesRegex(ValueError, 'group'):
            build_manifest(self.source, self.data, dict(self.cfg, group_mode='grouped'))
        self.assertEqual(self.m['coverage']['history_queries'], 0)

    def test_grouped_queries_really_match_history(self):
        annotations = {}
        for r in json.loads((self.source/'observations.json').read_text()):
            for o in r['observations']:
                annotations[o['key']] = {'group': str(int(r['frame']) % 2), 'source': 'manual-fixture'}
        path = self.root/'groups.json'; write_json(path, annotations)
        m = build_manifest(self.source, self.data, dict(self.cfg, group_mode='grouped',
                                                       view_group_manifest=str(path)))
        self.assertGreater(m['coverage']['history_queries'], 0)
        self.assertEqual(validate_manifest(m)['episodes'], len(m['episodes']))

    def test_permutation_padding_null(self):
        ep = copy.deepcopy(self.m['episodes'][0])
        prepared = fake_cache(self.root, self.m)
        other = copy.deepcopy(ep); other['steps'] = other['steps'][:2]
        a = pack_query_batch([ep, other], prepared, self.root)
        self.assertEqual(a['observation'].shape, (2, 4, 2, 16, 256))
        self.assertEqual(a['query'].shape, (2, 4, 3, 64, 256))
        self.assertFalse(a['step_valid'][1, 2:].any())
        for t, st in enumerate(ep['steps']):
            for j, q in enumerate(st['queries']):
                want = ep['identities'].index(q['label']) if q['label'] is not None else 2
                self.assertEqual(a['query_label'][0,t,j], want)
        self.assertFalse(a['query_active'][1, 2:].any())

    def test_cache_role_and_hash_rejection(self):
        prepared = fake_cache(self.root, self.m)
        key = next(iter(prepared['observations']))
        item = prepared['observations'][key]
        load_feature(item, self.root, 'observations', 'enc')
        with self.assertRaisesRegex(ValueError, 'role'):
            load_feature(item, self.root, 'queries', 'enc')
        with self.assertRaisesRegex(ValueError, 'encoder'):
            load_feature(item, self.root, 'observations', 'wrong')
        (self.root/item['file']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_feature(item, self.root, 'observations', 'enc')

    def test_changed_label_is_rejected(self):
        bad = copy.deepcopy(self.m)
        q = bad['episodes'][0]['steps'][0]['queries'][0]
        q['label'] = 'fabricated'
        with self.assertRaisesRegex(ValueError, 'label'):
            validate_manifest(bad)

    def test_missing_queries_fails_not_copies(self):
        cfg = dict(self.cfg, lengths=[12])
        records = json.loads((self.source/'observations.json').read_text())
        routes = json.loads((self.source/'routes.json').read_text())
        for r in routes:
            r['records'] = [x['id'] for x in records if x['split']==r['split']]
            r['actions'] = ['move']*11
        write_json(self.source/'routes.json', routes)
        meta=json.loads((self.source/'plan.json').read_text())
        meta['artifacts']['routes.json']=sha(self.source/'routes.json')
        write_json(self.source/'plan.json',meta)
        with self.assertRaisesRegex(ValueError, 'episode|quer'):
            build_manifest(self.source,self.data,cfg)

    def test_rgb_perturbation_reproducible(self):
        sample = next(iter(self.m['samples'].values()))
        p = dict(kind='cover', fraction=.3, seed=19)
        a = np.asarray(render_sample(self.data, sample, p))
        b = np.asarray(render_sample(self.data, sample, p))
        clean = np.asarray(render_sample(self.data, sample, {'kind':'clean'}))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,clean))

    def test_no_torch_import(self):
        code = 'import sys; import dinoexp.evidence_query_data; assert "torch" not in sys.modules'
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_old_cache_format(self):
        path=self.root/'old.npz'
        np.savez_compressed(path,observation=np.zeros((16,256),np.float32),valid=np.ones(16,bool))
        x,v=load_old_observation(dict(file=path.name,sha256=sha(path)),self.root)
        self.assertEqual(x.shape,(16,256)); self.assertTrue(v.all())

    def test_training_class_weights_population_mean(self):
        eps=[dict(steps=[dict(queries=[dict(label='A')]*9+[dict(label=None)])])]
        weights,counts=training_weights(eps)
        self.assertAlmostEqual(float(np.dot(weights,counts)/sum(counts)),1.,places=6)
        self.assertAlmostEqual(float(weights[1]),5.)
        weights,counts=training_weights([dict(steps=[dict(queries=[dict(label='A')])])])
        np.testing.assert_array_equal(weights,[1.,0.])

    def test_wrong_cache_sample_rejected(self):
        prepared=fake_cache(self.root,self.m)
        entry=next(iter(prepared['queries'].values()))
        with self.assertRaisesRegex(ValueError,'source'):
            load_feature(entry,self.root,'queries','enc','wrong-source')

    def test_crop_and_outer_split_leakage_rejected(self):
        bad=copy.deepcopy(self.m); ep=bad['episodes'][0]
        q=ep['steps'][0]['queries'][0]; o=ep['steps'][0]['observations'][0]
        bad['samples'][q['sample_id']]['crop_sha256']=bad['samples'][o['sample_id']]['crop_sha256']
        with self.assertRaisesRegex(ValueError,'leak'): validate_manifest(bad)
        bad=copy.deepcopy(self.m)
        probe=next(e for e in bad['episodes'] if e['split']=='probe')
        train=next(e for e in bad['episodes'] if e['split']=='train')
        probe['steps'][0]['parent_source_sha256']=train['steps'][0]['source_rgb_sha256']
        probe['context_sha256']=context_digest(probe)
        with self.assertRaisesRegex(ValueError,'split.*leak'): validate_manifest(bad)

    def test_general_query_uniform_candidates_and_null(self):
        nulls=[q for e in self.m['episodes'] for st in e['steps'] for q in st['queries'] if q['label'] is None]
        self.assertGreater(len(nulls),0)
        # Different candidate capacities must move Null to the batch-wide last slot.
        prepared=fake_cache(self.root,self.m); ep=copy.deepcopy(self.m['episodes'][0])
        ep['identities'].append('C')
        a=pack_query_batch([self.m['episodes'][0],ep],prepared,self.root)
        self.assertEqual(a['observation'].shape[2],3)
        self.assertFalse(a['observation_valid'][0,:,2].any())
        for t,st in enumerate(self.m['episodes'][0]['steps']):
            for j,q in enumerate(st['queries']):
                if q['label'] is None: self.assertEqual(a['query_label'][0,t,j],3)


if __name__ == '__main__':
    unittest.main()
