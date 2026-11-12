"""Auditable mask-assisted sequence plans and numpy batches; never imports torch."""
from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .core import checked_path
from .region_ab_data import digest_json, read_json, sha, verify_sources, write_json
from .region_geometry import DEFAULTS, propose, validate_config as validate_region_config


DEFAULT_CONFIG = dict(
    proposal=dict(DEFAULTS['proposal'], relative_peak=.3, uniform_multiple=1.5),
    hints=dict(min_seed_pixels=16, min_seed_fraction=.01, min_visible_pixels=32,
               min_area_518=196., padding_fraction=.04),
    routes=dict(length=24, stride=8),
    training=dict(epochs=100, seeds=[7, 17, 27], lr=1e-4, batch_size=4,
                  weight_decay=.01, heads=4, slots=8, balance_classes=True),
    exact_dedup=True, format_version=1)
SOURCES = ['run_region_sequence.py', 'dinoexp/region_sequence_data.py',
           'dinoexp/region_sequence_geometry.py', 'dinoexp/region_sequence_routes.py',
           'dinoexp/region_sequence_prepare.py', 'dinoexp/region_sequence_training.py',
           'dinoexp/region_sequence_report.py', 'dinoexp/region_sequence_metrics.py',
           'dinoexp/evidence_memory.py', 'dinoexp/local_attention.py',
           'dinoexp/region_ab_prepare.py', 'dinoexp/region_ab_data.py',
           'dinoexp/region_geometry.py', 'dinoexp/dense_model.py', 'dinoexp/dense_views.py',
           'dinoexp/core.py', 'dinoexp/local_paths.py', 'dinoexp/dense_cache.py']
ROOT = Path(__file__).resolve().parents[1]


def source_record():
    return {name: sha(ROOT/name) for name in SOURCES}


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG) or config['format_version'] != 1:
        raise ValueError('expected complete region sequence v1 config')
    for key in ('hints', 'routes', 'training'):
        if not isinstance(config[key], dict) or set(config[key]) != set(DEFAULT_CONFIG[key]):
            raise ValueError('sequence config fields differ: '+key)
    region_config = copy.deepcopy(DEFAULTS)
    region_config['proposal'] = config['proposal']
    validate_region_config(region_config)
    if type(config['exact_dedup']) is not bool:
        raise ValueError('exact_dedup must be boolean')
    h, r, t = (config[key] for key in ('hints', 'routes', 'training'))
    numbers = [*h.values(), *r.values(), *(v for k, v in t.items()
                                         if k not in ('seeds', 'balance_classes'))]
    if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError('finite nonnegative configuration numbers required')
    if (not 0 < h['min_seed_fraction'] <= 1 or min(h['min_seed_pixels'],
            h['min_visible_pixels'], h['min_area_518']) <= 0 or h['padding_fraction'] > .5):
        raise ValueError('invalid mask hint thresholds')
    for value in (r['length'], r['stride'], t['epochs'], t['batch_size'], t['heads'], t['slots']):
        if type(value) is not int or value < 1:
            raise ValueError('length/stride/training counts must be positive integers')
    if r['length'] < 2 or t['lr'] <= 0 or 256 % t['heads']:
        raise ValueError('invalid route length, learning rate or heads')
    if (type(t['balance_classes']) is not bool or not isinstance(t['seeds'], list)
            or not t['seeds'] or any(type(s) is not int or not 0 <= s < 2**32 for s in t['seeds'])
            or len(set(t['seeds'])) != len(t['seeds'])):
        raise ValueError('invalid balancing option or random seeds')


def _source(regions, root):
    meta = read_json(regions/'prepared.json')
    if meta.get('status') != 'complete' or meta.get('kind') != 'region_encoding_ab_v1':
        raise ValueError('--regions must be a completed teacher-region A/B preparation')
    name = 'proposals/records.json'
    if sha(regions/name) != meta['artifacts'][name]:
        raise ValueError('source teacher proposals changed')
    verify_sources(root, meta['sources'])
    records = read_json(regions/name)['records']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('empty or duplicate source teacher records')
    return meta, records


def plan(args, config):
    """CPU-only geometry/annotation pass over saved heatmaps; no model loading."""
    from .region_sequence_geometry import group_regions
    from .region_sequence_routes import build_routes, validate_route
    validate_config(config)
    output, root, regions = (Path(p).resolve() for p in (args.output, args.data, args.regions))
    if output.exists():
        raise ValueError('output exists; choose a new --output to preserve previous runs')
    meta, records = _source(regions, root)
    capture = read_json(root/'dataset_manifest.json')
    original, graphs = {}, {}
    for case in capture['cases']:
        directory = checked_path(root, case['path'])
        graphs[case['scene_id']] = read_json(directory/'graph.json')
        for row in map(json.loads, (directory/'frames.jsonl').read_text(encoding='utf-8').splitlines()):
            original[case['scene_id'], row['state_id']] = row
    audit, frames, by_scene, frame_splits = [], {}, {}, {}
    for source in records:
        if source.get('teacher_error'):
            raise ValueError('teacher preparation failed: '+source['id'])
        r = dict(source, original_candidates=source['candidates'],
                 original_threshold=source['threshold'],
                 **propose(source['spatial'], source['size'], config['proposal']))
        key = (r['scene'], r['frame'])
        frame = original[key]
        if (frame['annotation_validity'] != 'valid' or frame['image_path'] != r['image']
                or sha(checked_path(root, r['image'])) != r['image_sha256']):
            raise ValueError('source RGB/capture mismatch: '+r['id'])
        if key in frame_splits and frame_splits[key] != r['split']:
            raise ValueError('a physical frame crosses splits')
        frame_splits[key] = r['split']
        masks = {}
        for uid, obj in sorted(frame['objects'].items()):
            if obj['mask']:
                with Image.open(checked_path(root, obj['mask']['path'])) as im:
                    masks[uid] = np.asarray(im.convert('L')) > 0
            else:
                masks[uid] = np.zeros((r['size'][1], r['size'][0]), bool)
        grouped = group_regions(r, masks, config['hints'])
        for observation in grouped['observations']:
            # Identity is a routing hint only; the frozen crop encoder sees RGB.
            observation['key'] = digest_json([r['image_sha256'], observation['identity'], observation['box']])
        item = dict(r, **grouped, identities=sorted(masks),
                    physical_visible={uid: bool(m.any()) for uid, m in masks.items()},
                    physical_pixels={uid: int(m.sum()) for uid, m in masks.items()})
        audit.append(item)
        state = graphs[r['scene']]['states'][r['frame']]
        frames[key] = dict(scene=r['scene'], state=r['frame'], split=r['split'],
                           image_sha256=r['image_sha256'], camera_pose=state['pose'])
        by_scene.setdefault(r['scene'], {}).setdefault(r['target_identity'], {})[r['frame']] = r['id']
    lookup = {r['id']: r for r in audit}
    routes = []
    for scene in sorted(by_scene):
        selected = [f for (name, _), f in frames.items() if name == scene]
        for split in sorted({f['split'] for f in selected}):
            geometric = build_routes(scene, split, graphs[scene], selected, **config['routes'])
            allowed = {f['state'] for f in selected if f['split'] == split}
            for route in geometric:
                validate_route(route, graphs[scene], allowed)
                for uid, references in sorted(by_scene[scene].items()):
                    if any(s not in references for s in route['states']):
                        raise ValueError('source regions lack a route frame/reference; rebuild teacher proposals')
                    ids = [references[s] for s in route['states']]
                    first = lookup[ids[0]]
                    if any(lookup[i]['reference_sha256'] != first['reference_sha256']
                           or lookup[i]['identities'] != first['identities'] for i in ids):
                        raise ValueError('reference or identity registry changed inside a route')
                    routes.append(dict(route, id=route['id']+'/'+uid.rsplit(':', 1)[-1],
                                       target_identity=uid, reference_sha256=first['reference_sha256'],
                                       reference=first['reference'], identities=first['identities'], records=ids))
    if not routes or not any(r['split'] == 'train' for r in routes) or len({r['split'] for r in routes}) < 2:
        raise ValueError('need nonempty training and held-out sequences')
    splits = {s: {lookup[i]['image_sha256'] for r in routes if r['split'] == s for i in r['records']}
              for s in {r['split'] for r in routes}}
    for s, hashes in splits.items():
        if any(hashes & other for t, other in splits.items() if s != t):
            raise ValueError('route RGB leakage across splits')
    counts = {}
    for split, hashes in sorted(splits.items()):
        selected_routes = [r for r in routes if r['split'] == split]
        used = {rid for r in selected_routes for rid in r['records']}
        rows = [r for r in audit if r['split'] == split]
        seen_counts = Counter()
        for route in selected_routes:
            accepted, physical = set(), set()
            for rid in route['records']:
                row = lookup[rid]
                accepted.update(o['identity'] for o in row['observations'] if o['write_allowed'])
                physical.update(uid for uid, yes in row['physical_visible'].items() if yes)
                seen_counts['accepted_target_prefixes'] += int(route['target_identity'] in accepted)
                seen_counts['physical_target_prefixes'] += int(route['target_identity'] in physical)
                seen_counts['two_accepted_vehicle_prefixes'] += int(len(accepted) >= 2)
        counts[split] = dict(tasks=len(selected_routes), prefixes=sum(len(r['records']) for r in selected_routes),
                             unique_rgb=len(hashes), source_pairs=len(rows),
                             teacher_seeded_vehicle_observations=sum(len(r['observations']) for r in rows),
                             writable_observations=sum(o['write_allowed'] for r in rows for o in r['observations']),
                             uncovered_records=[r['id'] for r in rows if r['id'] not in used], **seen_counts)
    output.mkdir(parents=True)
    write_json(output/'observations.json', audit)
    write_json(output/'routes.json', routes)
    provenance = dict(kind='region_sequence_plan_v1', status='complete', config=copy.deepcopy(config),
                      code=source_record(), sources=meta['sources'], split=meta['split'], counts=counts,
                      teacher_sha256=meta['teacher_sha256'], backbone_sha256=meta['backbone_sha256'],
                      teacher_feature_config=meta['feature_config'], backbone_source=meta['backbone_source'],
                       source_proposal_config=meta['config']['proposal'],
                      source_prepared_sha256=sha(regions/'prepared.json'),
                      source_proposals_sha256=sha(regions/'proposals/records.json'), source_regions=str(regions),
                      assistance='SIMULATION MASKS provide grouping, full visible extent and stable identity. '
                                 'Only vehicles intersecting retained teacher active seeds enter memory. '
                                 'Not autonomous detection/association and not comparable to unassisted A/B.',
                      sampling='fixed graph patrol windows, both directions; no target-based route selection; '
                               'nominal graph paths, no collision-free flight claim',
                      truth='accepted_seen for matching; physical_seen reported separately and never filtered',
                      artifacts={name: sha(output/name) for name in ('observations.json', 'routes.json')})
    write_json(output/'plan.json', provenance)
    print('Sequence plan complete:', counts, flush=True)
    return provenance


def load_plan(output, root, verify_data=True):
    output, root = Path(output), Path(root)
    meta = read_json(output/'plan.json')
    if meta.get('status') != 'complete' or meta.get('kind') != 'region_sequence_plan_v1':
        raise ValueError('need complete region sequence plan')
    validate_config(meta['config'])
    for name, expected in meta['artifacts'].items():
        if sha(checked_path(output, name)) != expected:
            raise ValueError('sequence plan artifact changed: '+name)
    for name, expected in meta['code'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('sequence implementation changed; make a new plan: '+name)
    if verify_data:
        verify_sources(root, meta['sources'])
    records, routes = read_json(output/'observations.json'), read_json(output/'routes.json')
    lookup = {r['id']: r for r in records}
    if len(lookup) != len(records):
        raise ValueError('duplicate sequence record')
    for route in routes:
        if len(route['states']) != len(route['records']) or len(route['actions'])+1 != len(route['states']):
            raise ValueError('invalid stored route lengths')
        for state, rid in zip(route['states'], route['records'], strict=True):
            row = lookup[rid]
            if (row['frame'] != state or row['scene'] != route['scene'] or row['split'] != route['split']
                    or row['target_identity'] != route['target_identity']
                    or row['reference_sha256'] != route['reference_sha256']
                    or row['identities'] != route['identities']):
                raise ValueError('route order/split/reference/identity changed')
    return meta, lookup, routes


def sequence_arrays(routes, records, refs, features, *, dedup=True, rng=None):
    """Stable per-route permutation; truth is computed before and outside the model."""
    if not routes:
        raise ValueError('empty sequence batch')
    length = len(routes[0]['records'])
    if any(len(r['records']) != length for r in routes):
        raise ValueError('batch route lengths differ')
    capacity = max(len(r['identities']) for r in routes)
    x = np.zeros((len(routes), length, capacity, 16, 256), np.float32)
    raw_valid, write_valid = np.zeros(x.shape[:-1], bool), np.zeros(x.shape[:-1], bool)
    truths = np.full((len(routes), length), capacity, np.int64)
    physical_truths, current_truths = np.full_like(truths, capacity), np.full_like(truths, capacity)
    physical_now = np.zeros((len(routes), length, capacity), bool)
    references, rv, identities, keys, targets = [], [], [], [], []
    for i, route in enumerate(routes):
        order = list(route['identities'])
        if rng is not None:
            rng.shuffle(order)
        if len(order) != len(set(order)) or not order:
            raise ValueError('need unique stable physical identity slots')
        identities.append(order)
        target = order.index(route['target_identity']); targets.append(target)
        accepted_seen, physical_seen, used = set(), set(), {uid: set() for uid in order}
        row_keys = []
        reference = refs[route['reference_sha256']]
        references.append(reference['tokens']); rv.append(reference['valid'])
        for t, rid in enumerate(route['records']):
            record = records[rid]
            current_keys, current = {}, set()
            for j, uid in enumerate(order):
                if record['physical_visible'][uid]:
                    physical_seen.add(uid)
                physical_now[i, t, j] = record['physical_visible'][uid]
            for obs in record['observations']:
                uid, key = obs['identity'], obs['key']
                if uid not in order:
                    raise ValueError('observation identity absent from route registry')
                if not obs['write_allowed']:
                    continue
                if uid in current:
                    raise ValueError('fragments must be grouped before batching')
                f = features[key]
                if not f['valid'].any():
                    raise ValueError('allowed observation has no features; never silently relabel')
                j = order.index(uid)
                x[i, t, j], raw_valid[i, t, j] = f['observation'], f['valid']
                current_keys[uid] = key
                current.add(uid); accepted_seen.add(uid)
                if not dedup or key not in used[uid]:
                    write_valid[i, t, j] = f['valid']; used[uid].add(key)
            truths[i, t] = target if route['target_identity'] in accepted_seen else capacity
            physical_truths[i, t] = target if route['target_identity'] in physical_seen else capacity
            current_truths[i, t] = target if route['target_identity'] in current else capacity
            row_keys.append(current_keys)
        keys.append(row_keys)
    return dict(observation=x, valid=write_valid, current_valid=raw_valid,
                reference=np.stack(references), reference_valid=np.stack(rv), truth=truths,
                current_truth=current_truths, physical_truth=physical_truths, physical_now=physical_now,
                target_index=np.asarray(targets), identities=identities, observation_keys=keys)


def training_weights(routes, records, balanced=True):
    counts = Counter()
    for route in routes:
        if route['split'] != 'train':
            raise ValueError('class weights must use training routes only')
        seen = set()
        for rid in route['records']:
            seen.update(o['identity'] for o in records[rid]['observations'] if o['write_allowed'])
            counts[route['target_identity'] if route['target_identity'] in seen else 'Null'] += 1
    if not any(k != 'Null' for k in counts):
        raise ValueError('training routes contain no accepted target observation')
    total = sum(counts.values())
    weights = {k: total/(len(counts)*v) if balanced else 1. for k, v in counts.items()}
    return dict(counts=dict(counts), weights=weights, computed_from='train prefixes only')


def load_prepared(output, root):
    output = Path(output)
    meta, records, routes = load_plan(output, root)
    prepared = read_json(output/'prepared.json')
    if (prepared.get('status') != 'complete' or prepared.get('kind') != 'region_sequence_features_v1'
            or prepared['plan_sha256'] != sha(output/'plan.json')):
        raise ValueError('need complete features tied to this sequence plan')
    refs, features = {}, {}
    for entries, destination, is_ref in ((prepared['references'], refs, True),
                                         (prepared['observations'], features, False)):
        for key, entry in entries.items():
            path = checked_path(output, entry['file'])
            if sha(path) != entry['sha256']:
                raise ValueError('sequence feature hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                value = {n: data[n].copy() for n in data.files}
            token_key, side = ('tokens', 64) if is_ref else ('observation', 16)
            if (value[token_key].shape != (side, 256) or value['valid'].shape != (side,)
                    or value['valid'].dtype != np.bool_ or not value['valid'].any()
                    or not np.isfinite(value[token_key]).all()):
                raise ValueError('bad sequence features')
            destination[key] = value
    expected = {o['key'] for r in records.values() for o in r['observations'] if o['write_allowed']}
    if expected != set(features) or not {r['reference_sha256'] for r in routes} <= set(refs):
        raise ValueError('missing or extra sequence features')
    return meta, records, routes, refs, features
