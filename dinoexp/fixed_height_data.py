"""Pure-data adapter for the previously captured fixed-height orbit collection.

Boxes and cross-view identities are supplied by capture metadata. They are not
teacher detections or visible-instance masks. No neural-network imports here.
"""
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .core import checked_path
from .region_ab_data import digest_json, read_json, sha, verify_sources, write_json
from .region_sequence_data import source_record

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = ('run_fixed_height_test.py', 'dinoexp/fixed_height_data.py',
               'dinoexp/fixed_height_eval.py', 'dinoexp/fixed_height_report.py')
METHODS = ('last', 'mean', 'memory')
PROTOCOL = ('Supplied projected 3D vehicle boxes and capture identities; RGB crops '
            'from original frames; fixed existing checkpoints; recognition/fusion only. '
            'No teacher localization, mask visibility, automatic association or training.')


def code_record():
    return {**source_record(), **{name: sha(ROOT/name) for name in NEW_SOURCES}}


def verify_code(expected):
    for name, digest in expected.items():
        path = checked_path(ROOT, name)
        normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if digest not in (sha(path), normalized):
            raise ValueError('implementation differs from recorded source: '+name)


def source_bundle(directory):
    directory = Path(directory).resolve()
    run = read_json(directory/'training/run.json')
    prepared = read_json(directory/'prepared.json')
    plan = read_json(directory/'plan.json')
    if (run.get('kind') != 'region_sequence_training_v1' or run['status'] != 'complete'
            or prepared['status'] != 'complete' or plan['status'] != 'complete'):
        raise ValueError('need a complete existing region sequence experiment')
    verify_code(run['code'])
    if (sha(directory/'plan.json') != run['plan_sha256']
            or sha(directory/'prepared.json') != run['prepared_sha256']):
        raise ValueError('source experiment metadata hash mismatch')
    pairs = [(r['method'], r['seed']) for r in run['results']]
    if (len(pairs) != 9 or set(pairs) != {(m, s) for m in METHODS for s in (7, 17, 27)}):
        raise ValueError('test all nine existing checkpoints, without seed selection')
    checkpoints = []
    for entry in run['results']:
        file = checked_path(directory, entry['checkpoint'])
        if sha(file) != entry['checkpoint_sha256']:
            raise ValueError('source checkpoint hash mismatch: '+str(file))
        checkpoints.append({k: entry[k] for k in ('method', 'seed', 'checkpoint', 'checkpoint_sha256')})
    return dict(run_sha256=sha(directory/'training/run.json'),
                plan_sha256=run['plan_sha256'], prepared_sha256=run['prepared_sha256'],
                checkpoints=checkpoints, config=run['config'],
                teacher_sha256=plan['teacher_sha256'], backbone_sha256=plan['backbone_sha256'],
                teacher_feature_config=plan['teacher_feature_config'],
                backbone_source=plan['backbone_source']), plan


def checked_box(row, size):
    import json
    box = json.loads(row['crop_box'])
    if len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError('integer crop box required')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError('crop box outside image: '+row['sample_id'])
    return box


def audit(args):
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('output already exists; preserve it and choose a new directory')
    source, old_plan = source_bundle(args.checkpoint_run)
    collection = read_json(root/'collection.json')
    if collection['status'] != 'complete':
        raise ValueError('capture collection is not complete')
    cases, records, sources, sizes = [], [], {}, {}

    def asset(relative):
        file = checked_path(root, relative)
        if relative not in sources:
            sources[relative] = sha(file)
        return sources[relative]

    asset('collection.json')
    for case in collection['plan']['cases']:
        name = case['name']
        run = read_json(checked_path(root, name+'/run.json'))
        asset(name+'/run.json'); asset(name+'/metadata.csv')
        if run['status'] != 'complete' or run['config'] != case['config']:
            raise ValueError('capture status/config differs: '+name)
        with checked_path(root, name+'/metadata.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        identities = sorted(c['identity'] for c in run['config']['candidates'])
        target = run['config']['target_identity']
        reference = [r for r in rows if r['role'] == 'reference']
        queries = [r for r in rows if r['role'] == 'query']
        if (identities != ['A', 'B'] or len(reference) != 1
                or reference[0]['identity'] != target or len({r['sample_id'] for r in rows}) != len(rows)
                or len(reference)+len(queries) != len(rows)
                or any(r['status'] != 'captured' for r in rows)):
            raise ValueError('invalid reference, identities or capture rows: '+name)
        for row in rows:
            if row['identity'] not in identities:
                raise ValueError('unknown capture identity')
            for field in ('image', 'crop'):
                relative = name+'/'+row[field]
                asset(relative)
                if relative not in sizes:
                    with Image.open(checked_path(root, relative)) as im:
                        sizes[relative] = im.size
            box = checked_box(row, sizes[name+'/'+row['image']])
            if sizes[name+'/'+row['crop']] != (box[2]-box[0], box[3]-box[1]):
                raise ValueError('saved crop dimensions differ from projected box')
            candidate = next(c for c in run['config']['candidates'] if c['identity'] == row['identity'])
            if candidate['blueprint'] != row['blueprint']:
                raise ValueError('capture identity blueprint mismatch')
        case_records = []
        steps = len(run['config']['views'])
        if len(queries) != steps*2 or sorted({int(r['sequence']) for r in queries}) != list(range(steps)):
            raise ValueError('missing or duplicated capture angle')
        for t, view in enumerate(run['config']['views']):
            pair = [r for r in queries if int(r['sequence']) == t]
            if (sorted(r['identity'] for r in pair) != identities
                    or len({(r['image'], r['frame'], r['camera_pose']) for r in pair}) != 1
                    or any(r['view'] != view['name'] or float(r['azimuth']) != view['azimuth'] for r in pair)):
                raise ValueError('A/B are not from one shared camera observation')
            image = name+'/'+pair[0]['image']; digest = asset(image)
            observations = []
            for row in sorted(pair, key=lambda r: r['identity']):
                box = checked_box(row, sizes[image])
                observations.append(dict(identity=row['identity'], box=box, crop=name+'/'+row['crop'],
                                         key=digest_json([digest, row['identity'], box])))
            record = dict(id=f'{name}/{t:02}', case=name, step=t, angle=view['azimuth'],
                          image=image, image_sha256=digest, observations=observations)
            records.append(record); case_records.append(record['id'])
        ref = name+'/'+reference[0]['crop']
        cases.append(dict(id=name, group=case['group'], pair_kind=case['pair_kind'],
                          target_identity=target, identities=identities, reference=ref,
                          reference_sha256=asset(ref), records=case_records,
                          candidates=run['config']['candidates']))
    if len({c['id'] for c in cases}) != len(cases) or len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate case/record IDs')
    images = {r['image'] for r in records} | {c['id']+'/images/reference.png' for c in cases}
    crops = {o['crop'] for r in records for o in r['observations']} | {c['reference'] for c in cases}
    for key, actual in [('expected_full_images', len(images)), ('expected_crops', len(crops))]:
        if key in collection['plan'] and actual != collection['plan'][key]:
            raise ValueError('collection file count differs: '+key)
    overlap = sorted(set(sources[p] for p in images | crops) & set(old_plan['sources'].values()))
    if overlap:
        raise ValueError('test image hashes overlap source training collection')
    summary = dict(cases=len(cases), observations=len(records), full_images=len(images),
                   supplied_crops=len(crops), unique_observation_crops=len({o['key'] for r in records for o in r['observations']}),
                   steps_per_case=dict(Counter(len(c['records']) for c in cases)),
                   pair_kinds=dict(Counter(c['pair_kind'] for c in cases)),
                   source_image_hash_overlap=len(overlap),
                   reference_query_overlap=sum(c['reference_sha256'] in {sources[o['crop']] for r in records
                                               if r['case'] == c['id'] for o in r['observations']} for c in cases))
    meta = dict(kind='fixed_height_test_plan_v1', status='complete', protocol=PROTOCOL,
                dataset_purpose=collection['plan'].get('purpose'), code=code_record(),
                source=source, sources=sources, cases=cases, records=records, counts=summary,
                order='recorded 0..345 degree orbit order; reset state for every case',
                truth='capture identity of the supplied crop; no physical visibility/mask truth',
                negative_examples=0, exact_dedup=True)
    output.mkdir(parents=True)
    write_json(output/'plan.json', meta)
    print('Fixed-height data audit:', summary, flush=True)
    return meta


def load_plan(args):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    plan = read_json(output/'plan.json')
    if plan.get('kind') != 'fixed_height_test_plan_v1' or plan['status'] != 'complete':
        raise ValueError('incomplete fixed-height test plan')
    verify_code(plan['code']); verify_sources(root, plan['sources'])
    current, _ = source_bundle(args.checkpoint_run)
    if current != plan['source']:
        raise ValueError('source experiment changed after test plan')
    return plan


def load_features(args):
    output = Path(args.output).resolve()
    plan = load_plan(args); prepared = read_json(output/'prepared.json')
    if (prepared.get('kind') != 'fixed_height_test_features_v1' or prepared['status'] != 'complete'
            or sha(output/'plan.json') != prepared['plan_sha256']):
        raise ValueError('incomplete features or plan changed')
    features, refs = {}, {}
    for field, target, token_name, length in [('observations', features, 'observation', 16),
                                             ('references', refs, 'tokens', 64)]:
        for key, entry in prepared[field].items():
            file = checked_path(output, entry['file'])
            if sha(file) != entry['sha256']:
                raise ValueError('feature file hash mismatch')
            with np.load(file, allow_pickle=False) as value:
                tokens, valid = value[token_name].copy(), value['valid'].copy()
            if (tokens.shape != (length, 256) or valid.shape != (length,)
                    or tokens.dtype != np.float32 or valid.dtype != np.bool_
                    or not np.isfinite(tokens).all() or not valid.any()):
                raise ValueError('invalid fixed-height feature contract')
            target[key] = {token_name: tokens, 'valid': valid}
    return plan, prepared, refs, features


def batch_arrays(cases, records, refs, features, *, reverse_candidates=False):
    if len({len(c['records']) for c in cases}) != 1:
        raise ValueError('batch requires equal sequence lengths')
    b, t, k = len(cases), len(cases[0]['records']), 2
    observation = np.zeros((b, t, k, 16, 256), np.float32)
    valid = np.zeros((b, t, k, 16), bool); current = np.zeros_like(valid)
    references = np.zeros((b, 64, 256), np.float32); rv = np.zeros((b, 64), bool)
    truth, identities = [], []
    for i, case in enumerate(cases):
        order = list(reversed(case['identities'])) if reverse_candidates else case['identities'][:]
        identities.append(order); truth.append(order.index(case['target_identity']))
        ref = refs[case['reference_sha256']]; references[i], rv[i] = ref['tokens'], ref['valid']
        used = {uid: set() for uid in order}
        for step, record_id in enumerate(case['records']):
            for obj in records[record_id]['observations']:
                j = order.index(obj['identity']); item = features[obj['key']]
                observation[i, step, j] = item['observation']; current[i, step, j] = item['valid']
                if obj['key'] not in used[obj['identity']]:
                    valid[i, step, j] = item['valid']; used[obj['identity']].add(obj['key'])
    return dict(observation=observation, valid=valid, current_valid=current,
                reference=references, reference_valid=rv, truth=np.array(truth, np.int64),
                identities=identities)


def summarize(rows):
    def stats(selected):
        n = len(selected)
        return dict(count=n, correct=sum(r['correct'] for r in selected),
                    wrong_vehicle=sum(r['choice_identity'] not in (None, r['target_identity']) for r in selected),
                    null=sum(r['choice_identity'] is None for r in selected),
                    accuracy=sum(r['correct'] for r in selected)/n if n else None,
                    target_ranking=sum(r['target_ranks_first'] for r in selected)/n if n else None,
                    single_frame_accuracy=sum(r['single_frame_correct'] for r in selected)/n if n else None,
                    history_helped=sum(r['correct'] and not r['single_frame_correct'] for r in selected),
                    history_harmed=sum(not r['correct'] and r['single_frame_correct'] for r in selected))
    result = dict(all_prefixes=stats(rows))
    for field in ('case_id', 'group', 'pair_kind', 'target_identity', 'step'):
        result['by_'+field] = {str(v): stats([r for r in rows if r[field] == v])
                              for v in sorted({r[field] for r in rows})}
    final = [r for r in rows if r['last_step']]
    result['final_step'] = stats(final)
    result['all_views_correct_cases'] = sum(all(r['correct'] for r in rows if r['case_id'] == case)
                                          for case in {r['case_id'] for r in rows})
    return result
