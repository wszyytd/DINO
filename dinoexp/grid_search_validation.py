"""Verify source grids and publish reproducible, target-conditioned sequence indices."""

from collections import Counter
import json

import numpy as np

from .core import camera_pose_matches, checked_path
from .dense_cache import save_json
from .grid_search_capture import graph_for, grid_plan
from .local_pairs import sha256
from .search_episodes import (QUALITY_SPEC, _image, check_mask, check_pair, instance_labels, json_rows,
                              load_json, pixel_digest, recognition_quality, validate_splits)
from .search_grid import TEMPLATES, action_spec, derive_routes, different_view, route_events, route_rows


def read_json(path):
    return json.loads(load_json(path))


def read_layout(root, config, case):
    directory = checked_path(root, case['path'])
    scene = read_json(directory / 'scene.json')
    if (scene['status'] != 'captured' or scene['cleanup_errors'] or scene['case'] != case
        or scene['map'].rsplit('/', 1)[-1] != config['map']
        or len(set(scene['carla_versions'])) != 1
        or scene['instance_id_method'] != 'remove_restore_probe'):
        raise ValueError('layout did not finish capture and cleanup')
    objects = scene['objects']
    expected_objects = {case['scene_id'] + ':' + item['identity']: item
                        for item in config['capture']['candidates']}
    mappings, tags = scene['instance_mapping'], scene['vehicle_tags']
    refs = scene['references']
    if set(objects) != set(expected_objects) or set(mappings) != set(objects) or set(refs) != set(objects):
        raise ValueError('missing mapped vehicle or independent reference')
    for uid, expected in expected_objects.items():
        if (any(objects[uid][key] != value for key, value in expected.items()) or
            objects[uid]['actual_blueprint'] != expected['blueprint']):
            raise ValueError('configured and captured asset disagree')
    a, b = objects.values()
    clearance = (float(np.linalg.norm(np.subtract(a['center'][:2], b['center'][:2])))
                 - a['bbox_radius_xy_m'] - b['bbox_radius_xy_m'])
    if (not np.isclose(clearance, scene['bbox_clearance_lower_bound_m']) or
        clearance < config['minimum_vehicle_gap_m']):
        raise ValueError('vehicle clearance check failed')
    opaque_ids = [i for row in mappings.values() for i in row['ids']]
    if len(set(opaque_ids)) != len(opaque_ids) or any(not m['ids'] or not m['tags'] for m in mappings.values()):
        raise ValueError('missing or ambiguous opaque instance IDs')
    ref_frames = set()
    probe_frame = max(m['restored_frame'] for m in mappings.values())
    for uid, ref in refs.items():
        check_pair(ref['audit'])
        ref_frame = ref['audit']['rgb']['frame']
        if ref_frame <= probe_frame or ref_frame in ref_frames or ref['target_uid'] != uid:
            raise ValueError('reference capture is not independent')
        ref_frames.add(ref_frame)
        full, crop, mask, raw = [
            _image(root, ref[k], ref['hashes'][ref[k]], 'L' if k == 'mask' else 'RGB')
            for k in ('full', 'object', 'mask', 'raw')]
        metadata = ref['audit']['rgb']
        if (full.shape != raw.shape or full.shape[:2] != (metadata['height'], metadata['width'])
            or metadata['intrinsics'] != scene['intrinsics']
            or pixel_digest(full) != ref['rgb_sensor_sha256']):
            raise ValueError('reference pixel integrity check failed')
        masks, _ = instance_labels(raw, mappings, tags)
        quality = recognition_quality(masks[uid])
        x0, y0, x1, y1 = ref['crop_xyxy']
        if (not 0 <= x0 < x1 <= full.shape[1] or not 0 <= y0 < y1 <= full.shape[0]
            or quality != ref['quality'] or quality['recognition_quality'] != 'usable'
            or not np.array_equal(crop, full[y0:y1, x0:x1])
            or not np.array_equal(mask, masks[uid].astype(np.uint8) * 255)):
            raise ValueError('reference crop or mask failed validation')
        a, b, c, d = quality['bbox_xyxy']
        if (not (x0 <= a < c <= x1 and y0 <= b < d <= y1)
            or max(c-a, d-b) < QUALITY_SPEC['reference_min_long_side']
            or min(c-a, d-b) < QUALITY_SPEC['reference_min_short_side']):
            raise ValueError('reference is too small or cropped outside its mask')
    graph = graph_for(config, case, scene['anchor'])
    for point in graph['points'].values():
        if point['capture_eligible']:
            point['capture_status'] = 'captured'
    if graph != read_json(directory / 'graph.json'):
        raise ValueError('graph is inconsistent with configured camera poses or actions')
    rows = json_rows(directory / 'frames.jsonl')
    audits = json_rows(directory / 'audit' / 'observations.jsonl')
    count = len(graph['states'])
    if len(rows) != count or len(audits) != count or scene['capture_count'] != count:
        raise ValueError('some eligible point/direction captures are missing')
    frames = {}
    previous_frame, previous_time = max(ref_frames), -1
    for (sid, state), row, audit in zip(graph['states'].items(), rows, audits, strict=True):
        rgb = audit['rgb']
        if (sid != row['state_id'] or sid != audit['state_id']
            or (row['point_id'], row['direction_id']) != (state['point_id'], state['direction_id'])
            or row['scene_id'] != case['scene_id'] or row['layout_id'] != case['scene_id']
            or row['group'] != case['group'] or row['annotation_validity'] != 'valid'
            or row['sensor_frame'] != rgb['frame'] or row['sensor_timestamp'] != rgb['timestamp']
            or rgb['frame'] <= previous_frame or rgb['timestamp'] <= previous_time
            or rgb['intrinsics'] != scene['intrinsics']
            or not camera_pose_matches(rgb['pose'], state['pose'])):
            raise ValueError(f'invalid grid capture ordering or pose for {sid}')
        check_pair(audit)
        pixels = _image(root, row['image_path'], rgb['file_sha256'], 'RGB')
        instance = audit['instance']
        raw = _image(root, instance['path'], instance['file_sha256'], 'RGB')
        if (pixels.shape != raw.shape or pixels.shape[:2] != (rgb['height'], rgb['width'])
            or pixel_digest(pixels) != rgb['sensor_sha256']):
            raise ValueError(f'RGB or instance dimensions or pixel integrity failed for {sid}')
        masks, ignored = instance_labels(raw, mappings, tags)
        if set(row['objects']) != set(masks):
            raise ValueError('frame lacks per-vehicle labels')
        for uid, mask in masks.items():
            label = row['objects'][uid]
            if label['quality'] != recognition_quality(mask):
                raise ValueError(f'wrong visibility or quality for {sid}:{uid}')
            check_mask(root, label['mask'], mask)
        check_mask(root, row['ignore_mask'], ignored)
        previous_frame, previous_time = rgb['frame'], rgb['timestamp']
        frames[sid] = row
    return scene, graph, frames


def tasks_for(stage, case, objects):
    targets = sorted(objects)
    if stage == 'smoke':
        return [(kind, target) for kind in TEMPLATES for target in targets]
    index = int(case['scene_id'].rsplit('_', 1)[1])
    return [(kind, targets[(slot + index//2) % 2]) for slot, kind in enumerate(TEMPLATES)]


def save_routes(root, case, scene, graph, frames, stage):
    routes, gaps = derive_routes(graph, frames, scene['objects'], tasks_for(stage, case, scene['objects']),
                                 case['seed'])
    directory = root / case['path'] / 'episodes'
    directory.mkdir()
    index = []
    for n, route in enumerate(routes):
        episode = f'{case["scene_id"]}_ep{n:02d}'
        path = directory / episode
        path.mkdir()
        ref = scene['references'][route['target_uid']]['object']
        labels, inputs = route_rows(graph, frames, route, ref)
        route.update(episode_id=episode, group=case['group'], split=case['split'],
                     scene_id=case['scene_id'], path=path.relative_to(root).as_posix(),
                     terminator='scripted_horizon', stop_success=None, recommendation_labels=None,
                     selection='annotation_filtered_graph_walk')
        save_json(path / 'episode.json', route)
        for filename, rows in [('frames.jsonl', labels), ('inputs.jsonl', inputs)]:
            (path / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        index.append(route['path'])
    save_json(directory / 'index.json', dict(paths=index, generation_gaps=gaps))


def read_routes(root, case, scene, graph, frames, stage):
    index = read_json(root / case['path'] / 'episodes' / 'index.json')
    tasks = set(tasks_for(stage, case, scene['objects']))
    attempted = set()
    routes, seen = [], set()
    for path in index['paths']:
        directory = checked_path(root, path)
        route = read_json(directory / 'episode.json')
        task = (route['template'], route['target_uid'])
        identity = (route['episode_id'], route['scene_id'], route['group'], route['split'])
        if (task not in tasks or task in attempted or
            identity != (directory.name, case['scene_id'], case['group'], case['split'])
            or directory.parent != (root / case['path'] / 'episodes').resolve()
            or route['path'] != path or tuple(route['states']) in seen
            or not 16 <= len(route['states']) <= 32
            or len(route['actions']) != len(route['states']) - 1
            or route['terminator'] != 'scripted_horizon' or route['stop_success'] is not None
            or route['recommendation_labels'] is not None
            or route['selection'] != 'annotation_filtered_graph_walk'):
            raise ValueError('invalid route identity, split, size or provenance')
        attempted.add(task)
        seen.add(tuple(route['states']))
        ref = scene['references'][route['target_uid']]['object']
        labels, inputs = route_rows(graph, frames, route, ref)
        if (json_rows(directory / 'frames.jsonl') != labels or
            json_rows(directory / 'inputs.jsonl') != inputs or
            not route_events(graph, frames, route, scene['objects'][route['target_uid']]['pose'][:2])):
            raise ValueError('route frame sequence, runtime allowlist or event check failed')
        routes.append(route)
    gaps = [(g['template'], g['target_uid']) for g in index['generation_gaps']]
    if len(set(gaps)) != len(gaps) or attempted.intersection(gaps) or attempted.union(gaps) != tasks:
        raise ValueError('route index has unaccounted or duplicated generation attempts')
    return routes, index['generation_gaps']


def layout_stats(scene, graph, frames, routes, gaps):
    targets = sorted(scene['objects'])
    vis = Counter()
    quality = {uid: Counter() for uid in targets}
    usable = {uid: [] for uid in targets}
    for sid, row in frames.items():
        present = []
        for uid in targets:
            q = row['objects'][uid]['quality']
            present.append(q['visibility_present'])
            quality[uid]['absent' if not present[-1] else q['recognition_quality']] += 1
            if q['recognition_quality'] == 'usable':
                usable[uid].append(sid)
        vis[{(True, True): 'both', (True, False): 'A_only',
             (False, True): 'B_only', (False, False): 'neither'}[tuple(present)]] += 1
    diversity = {uid: any(different_view(graph['states'], a, b, scene['objects'][uid]['pose'][:2])
                         for a in states for b in states)
                 for uid, states in usable.items()}
    stats = dict(source_images=len(frames), independent_references=len(targets), visibility=dict(vis),
                 quality_by_target={k: dict(v) for k, v in quality.items()},
                 both_usable=len(set(usable[targets[0]]) & set(usable[targets[1]])),
                 useful_bearing_change=diversity,
                 derived_sequences=len(routes), route_steps=sum(len(r['states']) for r in routes),
                 templates=dict(Counter(r['template'] for r in routes)),
                 reference_targets=dict(Counter(r['target_uid'].split(':')[-1] for r in routes)),
                 clearance_lower_bound_m=scene['bbox_clearance_lower_bound_m'], generation_gaps=gaps,
                 assets=[dict(uid=uid, blueprint=obj['actual_blueprint'], configured_color=obj['color'],
                              actor_color=obj['actor_attributes'].get('color'))
                         for uid, obj in scene['objects'].items()])
    stats['coverage_gaps'] = [key for key in ('both', 'A_only', 'B_only', 'neither') if not vis[key]]
    if not stats['both_usable']:
        stats['coverage_gaps'].append('both_usable')
    for uid in targets:
        if not diversity[uid] or not quality[uid]['weak']:
            stats['coverage_gaps'].append(f'{uid}: requires differing usable bearings and weak frames')
    return stats


def validate_collection(root, manifest, *, publishing=False):
    stage = manifest['stage']
    config = manifest['config']
    if (stage not in ('smoke', 'pilot') or manifest['mode'] != 'discrete_view_graph' or
        (not publishing and manifest['status'] != 'complete') or
        manifest['cases'] != grid_plan(config, stage)):
        raise ValueError('dataset is incomplete or its layout plan is inconsistent')
    if (read_json(root / 'splits.json') != {c['group']: c['split'] for c in manifest['cases']}
        or read_json(root / 'action_spec.json') != action_spec(config['grid'])
        or read_json(root / 'quality_spec.json') != QUALITY_SPEC):
        raise ValueError('split, action or quality specification mismatch')
    scenes, routes, image_hashes = {}, [], {}
    for case in manifest['cases']:
        scene, graph, frames = read_layout(root, config, case)
        episode_routes, gaps = read_routes(root, case, scene, graph, frames, stage)
        scenes[case['scene_id']] = layout_stats(scene, graph, frames, episode_routes, gaps)
        routes.extend(episode_routes)
        image_paths = [f['image_path'] for f in frames.values()] + [
            ref['full'] for ref in scene['references'].values()]
        for path in image_paths:
            fingerprint = sha256(checked_path(root, path))
            if image_hashes.setdefault(fingerprint, case['split']) != case['split']:
                raise ValueError('exact RGB source duplicates across splits')
    validate_splits(routes)
    splits = {}
    for case in manifest['cases']:
        part = splits.setdefault(case['split'], dict(independent_layouts=0, source_images=0,
                                  derived_sequences=0, route_steps=0, templates=Counter(),
                                  reference_targets=Counter(), visibility=Counter(), quality=Counter()))
        part['independent_layouts'] += 1
        stats = scenes[case['scene_id']]
        for key in ('source_images', 'derived_sequences', 'route_steps'):
            part[key] += stats[key]
        for key in ('templates', 'reference_targets', 'visibility'):
            part[key].update(stats[key])
        for counts in stats['quality_by_target'].values():
            part['quality'].update(counts)
    gaps = [f'{sid}: {row["coverage_gaps"] or []}; missing routes={row["generation_gaps"]}'
            for sid, row in scenes.items() if row['coverage_gaps'] or row['generation_gaps']]
    return dict(status='verified', research_readiness='manual_image_review_required',
                offline_coverage_gaps=gaps, independent_layouts=len(scenes),
                source_images=sum(p['source_images'] for p in splits.values()),
                derived_sequences=len(routes), scenes=scenes, splits=splits)
