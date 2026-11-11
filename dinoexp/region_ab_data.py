"""Strict source records and separate proposal/label/feature artifacts for A/B."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .core import checked_path
from .region_geometry import label_regions


ROOT = Path(__file__).resolve().parents[1]
ENCODING_SOURCES = ['dinoexp/region_geometry.py', 'dinoexp/region_ab_prepare.py',
                    'dinoexp/region_ab_data.py', 'dinoexp/dense_views.py',
                    'dinoexp/dense_model.py', 'dinoexp/local_attention.py']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def code_record():
    names = sorted(set(ENCODING_SOURCES + ['run_region_encoding_ab.py', 'dinoexp/evidence_memory.py'] +
                       [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT/'dinoexp').glob('region_ab_*.py')]))
    return dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sources={name: sha(ROOT/name) for name in names if (ROOT/name).is_file()})


def verify_sources(root, sources):
    for name, expected in sources.items():
        if sha(checked_path(root, name)) != expected:
            raise ValueError('source hash changed: '+name)


def collect_frames(root, teacher_cp):
    root = Path(root).resolve()
    manifest = read_json(root/'dataset_manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('mode') != 'discrete_view_graph':
        raise ValueError('need completed grid capture')
    cases = manifest['cases']
    explicit = any(c['split'] in ('val', 'test') for c in cases)
    if not explicit and len(cases) != 1:
        raise ValueError('multiple layouts require explicit train/val/test splits')
    sources = {'dataset_manifest.json': sha(root/'dataset_manifest.json')}
    frames, groups = [], {}
    teacher_images = set(teacher_cp['training_images'])
    for case in cases:
        directory = checked_path(root, case['path'])
        for name in ('scene.json', 'frames.jsonl', 'graph.json', 'audit/observations.jsonl'):
            path = directory/name
            sources[path.relative_to(root).as_posix()] = sha(path)
        scene, graph = read_json(directory/'scene.json'), read_json(directory/'graph.json')
        audit = {r['state_id']: r['rgb'] for r in map(json.loads, (directory/'audit/observations.jsonl').read_text().splitlines())}
        refs = []
        for uid, ref in sorted(scene['references'].items()):
            expected = ref['hashes'][ref['object']]
            if sha(checked_path(root, ref['object'])) != expected:
                raise ValueError('reference hash mismatch')
            sources[ref['object']] = expected
            refs.append(dict(identity=uid, path=ref['object'], sha256=expected))
        cut = (max(p['col'] for p in graph['points'].values())+1)*3//5
        for frame in map(json.loads, (directory/'frames.jsonl').read_text().splitlines()):
            if frame.get('annotation_validity') != 'valid':
                raise ValueError('unvalidated physical annotations cannot be treated as verified truth')
            path, sid = frame['image_path'], frame['state_id']
            image_hash = sha(checked_path(root, path))
            if image_hash != audit[sid]['file_sha256']:
                raise ValueError('RGB differs from capture audit')
            sources[path] = image_hash
            with Image.open(checked_path(root, path)) as im:
                size = list(im.size)
            for label in [*(r['mask'] for r in frame['objects'].values()), frame.get('ignore_mask')]:
                if label:
                    if sha(checked_path(root, label['path'])) != label['sha256']:
                        raise ValueError('mask hash mismatch')
                    sources[label['path']] = label['sha256']
            if explicit:
                split = case['split']
                if split not in ('train', 'val', 'test'):
                    raise ValueError('unknown explicit split')
                if case['group'] in groups and groups[case['group']] != split:
                    raise ValueError('layout group crosses explicit split')
                groups[case['group']] = split
                if split != 'train' and (case['group'] in teacher_cp['training_groups'] or image_hash in teacher_images):
                    raise ValueError('held-out layout overlaps teacher training')
            else:
                point = graph['points'][graph['states'][sid]['point_id']]
                split = 'train' if point['col'] < cut or image_hash in teacher_images else 'probe'
            frames.append(dict(scene=case['scene_id'], state=sid, group=case['group'], split=split,
                               image=path, image_sha256=image_hash, size=size, references=refs,
                               objects=frame['objects'], ignore_mask=frame.get('ignore_mask'),
                               camera_pose=audit[sid]['pose'], fov=audit[sid]['intrinsics']['horizontal_fov_deg']))
    splits = {s: {f['image_sha256'] for f in frames if f['split'] == s} for s in {f['split'] for f in frames}}
    if not splits.get('train') or len(splits) < 2:
        raise ValueError('need nonempty training and held-out images')
    for a in splits:
        for b in splits:
            if a != b and splits[a] & splits[b]:
                raise ValueError('RGB overlap between splits')
    scope = ('explicit train/val/test layouts, reported separately' if explicit else
             'SINGLE-LAYOUT DIAGNOSTIC: same vehicles/background/references; columns holdout, teacher-fit RGB excluded')
    return frames, sources, dict(scope=scope, unique_rgb={s: len(v) for s, v in splits.items()})


def labels_for_frame(root, frame, records, cfg):
    w, h = frame['size']
    masks = {}
    for uid, obj in frame['objects'].items():
        if obj['mask']:
            with Image.open(checked_path(root, obj['mask']['path'])) as im:
                masks[uid] = np.asarray(im.convert('L')) > 0
        else:
            masks[uid] = np.zeros((h, w), bool)
    ignore = None
    if frame['ignore_mask']:
        with Image.open(checked_path(root, frame['ignore_mask']['path'])) as im:
            ignore = np.asarray(im.convert('L')) > 0
    return {r['id']: label_regions([c['box'] for c in r['candidates']], masks,
                                   r['target_identity'], cfg, ignore) for r in records}


def save_features(directory, sample_id, observation, valid, errors):
    name = digest_json(sample_id)+'.npz'
    np.savez_compressed(directory/name, observation=observation, valid=valid)
    return dict(file=name, sha256=sha(directory/name), candidates=len(observation), errors=errors)


def load_bundle(output, data_root, verify_data=True):
    output = Path(output)
    manifest = read_json(output/'prepared.json')
    if manifest.get('status') != 'complete' or manifest.get('kind') != 'region_encoding_ab_v1':
        raise ValueError('need completed region A/B cache')
    for name, digest in manifest['artifacts'].items():
        if sha(checked_path(output, name)) != digest:
            raise ValueError('artifact changed: '+name)
    if verify_data:
        verify_sources(Path(data_root), manifest['sources'])
    for name in ENCODING_SOURCES:
        if sha(ROOT/name) != manifest['code']['sources'][name]:
            raise ValueError('encoding implementation changed; build a new cache: '+name)
    proposals = read_json(output/'proposals/records.json')['records']
    labels = read_json(output/'labels.json')
    expected = set(r['id'] for r in proposals)
    if expected != set(labels):
        raise ValueError('proposal/label sample mismatch')
    indexes = {v: read_json(output/v/'cache.json') for v in ('A', 'B')}
    for variant, index in indexes.items():
        if (index['proposal_sha256'] != manifest['artifacts']['proposals/records.json'] or
                index['labels_sha256'] != manifest['artifacts']['labels.json'] or
                index['kind'] != 'region_encoding_'+variant+'_v1' or set(index['entries']) != expected):
            raise ValueError('A/B proposal/label provenance mismatch')
        for entry in index['entries'].values():
            if sha(checked_path(output/variant, entry['file'])) != entry['sha256']:
                raise ValueError('region feature hash mismatch')
    refs = {}
    for key, entry in manifest['references'].items():
        path = checked_path(output, entry['file'])
        if sha(path) != entry['sha256']:
            raise ValueError('reference feature changed')
        with np.load(path, allow_pickle=False) as data:
            refs[key] = {name: data[name].copy() for name in data.files}
        ref = refs[key]
        if (ref['tokens'].shape != (64, 256) or ref['valid'].shape != (64,) or
                not ref['valid'].any() or not np.isfinite(ref['tokens']).all()):
            raise ValueError('invalid reference feature geometry/content')
    features = {v: {} for v in indexes}
    for r in proposals:
        k = len(r['candidates'])
        if not set(labels[r['id']]['positives']) <= set(range(k)):
            raise ValueError('invalid positive indices')
        for v in features:
            entry = indexes[v]['entries'][r['id']]
            with np.load(checked_path(output/v, entry['file']), allow_pickle=False) as data:
                x, valid = data['observation'].copy(), data['valid'].copy()
            if x.shape != (k, 16, 256) or valid.shape != (k, 16) or not np.isfinite(x).all():
                raise ValueError('bad region feature shape/value')
            features[v][r['id']] = dict(observation=x, valid=valid, errors=entry['errors'])
    return manifest, proposals, labels, refs, features
