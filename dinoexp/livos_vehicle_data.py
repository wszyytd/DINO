"""LiVOS vehicle test plans: captured graph walks and verified simulation masks, no torch."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .core import checked_path
from .region_ab_data import digest_json, read_json, sha, write_json
from .region_sequence_routes import build_routes, validate_route

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['run_livos_vehicle_test.py', 'dinoexp/livos_vehicle_data.py',
           'dinoexp/livos_vehicle_eval.py', 'dinoexp/livos_vehicle_metrics.py',
           'dinoexp/livos_vehicle_report.py', 'dinoexp/region_sequence_routes.py',
           'dinoexp/core.py', 'dinoexp/region_ab_data.py']


def code_hashes():
    return {p: sha(ROOT/p) for p in SOURCES}


def binary_mask(root, entry, size):
    if entry is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    with Image.open(checked_path(root, entry['path'])) as image:
        mask = np.asarray(image.convert('L')) > 0
    if mask.shape != (size[1], size[0]):
        raise ValueError('mask/image dimensions differ')
    return mask


def tags_for(frames, target):
    present = [f['objects'][target]['pixels'] > 0 for f in frames]
    tags = []
    if any(f['objects'][target]['boundary'] for f in frames[1:]):
        tags.append('boundary')
    if any(present[i] and not present[i-1] for i in range(1, len(present))):
        tags.append('reappearance')
    if any(f['objects'][target]['pixels'] and
           any(o['pixels'] for uid, o in f['objects'].items() if uid != target)
           for f in frames[1:]):
        tags.append('two_vehicles')
    if any(not p for p in present[1:]):
        tags.append('absence')
    return tags


def select_cases(candidates, maximum):
    """Balanced deterministic stress sample using annotations only, before inference."""
    if maximum < 1:
        raise ValueError('max-cases must be positive')
    remaining = sorted(candidates, key=lambda c: c['id'])
    selected, targets, tags = [], Counter(), Counter()
    while remaining and len(selected) < maximum:
        def priority(c):
            diversity = sum(1/(1+tags[t]) for t in c['tags'])
            return (targets[c['target']], -diversity, c['id'])
        case = min(remaining, key=priority)
        remaining.remove(case)
        case = dict(case, id=f'case_{len(selected):03d}', source_id=case['id'])
        selected.append(case)
        targets[case['target']] += 1
        tags.update(case['tags'])
    return selected


def plan(args):
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('output exists; choose a new directory to preserve results')
    if args.length < 2 or args.stride < 1 or args.min_init_pixels < 1:
        raise ValueError('invalid route length/stride/initial area')
    manifest = read_json(root/'dataset_manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('mode') != 'discrete_view_graph':
        raise ValueError('need completed discrete grid capture with pixel masks')
    sources = {'dataset_manifest.json': sha(root/'dataset_manifest.json')}
    frames, candidates = {}, []
    for capture in manifest['cases']:
        scene, directory = capture['scene_id'], checked_path(root, capture['path'])
        for name in ('frames.jsonl', 'graph.json', 'scene.json', 'audit/observations.jsonl'):
            source = directory/name
            sources[source.relative_to(root).as_posix()] = sha(source)
        graph = read_json(directory/'graph.json')
        audits = {r['state_id']: r['rgb']['file_sha256'] for r in
                  map(json.loads, (directory/'audit/observations.jsonl').read_text().splitlines())}
        raw = list(map(json.loads, (directory/'frames.jsonl').read_text().splitlines()))
        route_frames = []
        for row in raw:
            if row['annotation_validity'] != 'valid' or row['state_id'] not in graph['states']:
                raise ValueError('unverified capture annotation/graph state')
            image_path = checked_path(root, row['image_path'])
            image_hash = sha(image_path)
            if image_hash != audits[row['state_id']]:
                raise ValueError('capture RGB audit mismatch')
            sources[row['image_path']] = image_hash
            with Image.open(image_path) as image:
                size = list(image.size)
            ignore = binary_mask(root, row.get('ignore_mask'), size)
            objects, occupied = {}, np.zeros(ignore.shape, bool)
            for uid, obj in sorted(row['objects'].items()):
                mask = binary_mask(root, obj['mask'], size)
                if np.any(mask & (occupied | ignore)):
                    raise ValueError('target masks overlap each other or ignored pixels')
                occupied |= mask
                ys, xs = np.nonzero(mask)
                count = int(mask.sum())
                if count != obj['quality']['visible_pixels']:
                    raise ValueError('mask area differs from capture quality')
                boundary = bool(count and (xs.min() == 0 or ys.min() == 0 or
                                           xs.max() == size[0]-1 or ys.max() == size[1]-1))
                objects[uid] = dict(mask=obj['mask'], pixels=count, boundary=boundary,
                                    quality=obj['quality']['recognition_quality'])
            for entry in [*(o['mask'] for o in objects.values()), row.get('ignore_mask')]:
                if entry:
                    if sha(checked_path(root, entry['path'])) != entry['sha256']:
                        raise ValueError('capture mask hash mismatch')
                    sources[entry['path']] = entry['sha256']
            key = scene+'/'+row['state_id']
            if key in frames:
                raise ValueError('duplicate captured state')
            frames[key] = dict(id=key, scene=scene, state=row['state_id'], image=row['image_path'],
                               size=size, image_sha256=image_hash, objects=objects,
                               ignore=row.get('ignore_mask'), pose=graph['states'][row['state_id']]['pose'])
            route_frames.append(dict(scene=scene, split='capture', state=row['state_id']))
        routes = build_routes(scene, 'capture', graph, route_frames,
                              length=args.length, stride=args.stride)
        for route in routes:
            validate_route(route, graph, [f['state'] for f in route_frames])
            keys = [scene+'/'+s for s in route['states']]
            sequence = [frames[k] for k in keys]
            if len({tuple(f['size']) for f in sequence}) != 1:
                raise ValueError('official key-sum inference requires constant frame size')
            for uid, obj in sequence[0]['objects'].items():
                if obj['pixels'] < args.min_init_pixels or obj['boundary']:
                    continue
                candidates.append(dict(id=route['id']+'/'+uid, target=uid, frames=keys,
                                       actions=route['actions'], costs_s=route['costs_s'],
                                       tags=tags_for(sequence, uid)))
    cases = select_cases(candidates, args.max_cases)
    if not cases:
        raise ValueError('no route starts with a sufficiently visible, non-boundary target')
    result = dict(kind='livos_vehicle_plan_v1', status='complete',
                  config=dict(length=args.length, stride=args.stride, max_cases=args.max_cases,
                              min_init_pixels=args.min_init_pixels, size=args.size,
                              success_iou=.5, wrong_iou=.1, wrong_margin=.05, absent_pixels=32),
                  sources=sources, code=code_hashes(), frames=frames, cases=cases,
                  counts=dict(captured_states=len(frames),
                               unique_rgb=len({f['image_sha256'] for f in frames.values()}),
                               candidate_tasks=len(candidates),
                              selected_tasks=len(cases), total_steps=sum(len(c['frames']) for c in cases),
                              by_target=dict(Counter(c['target'] for c in cases)),
                               selected_tags=dict(Counter(t for c in cases for t in c['tags']))),
                  protocol='Single target per independent run; GT target mask at step 0 ONLY. '
                           'Later target/distractor/ignore masks used only for scoring after prediction. '
                           'No DINO model, teacher region, reference photo or training used.',
                  sampling='All captured states; deterministic graph patrols, annotation-selected stress '
                           'sample balanced across targets; no predictions used to select cases. '
                           'Repeated and overlapping frames/routes are not independent samples.',
                  limitations='One captured layout/two static vehicles; camera steps are 4 m/45 degree '
                              'discrete graph edges, not smooth video or certified collision-free flight. '
                              'Absence may be leaving field of view; not evidence of physical occlusion. '
                              'No new held-out identity claim; this is pretrained VOS transfer, not photo matching.')
    output.mkdir(parents=True)
    write_json(output/'plan.json', result)
    print('Vehicle plan:', result['counts'], flush=True)
    return result


def load_plan(output, data, verify=True):
    result = read_json(Path(output)/'plan.json')
    if result.get('kind') != 'livos_vehicle_plan_v1' or result.get('status') != 'complete':
        raise ValueError('need complete LiVOS vehicle plan')
    if result['code'] != code_hashes():
        raise ValueError('test implementation changed; create a new output')
    if verify:
        for name, expected in result['sources'].items():
            if sha(checked_path(data, name)) != expected:
                raise ValueError('input changed: '+name)
    return result


def plan_digest(result):
    return digest_json(result)
