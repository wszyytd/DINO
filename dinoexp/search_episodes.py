"""Data contracts for the reference-guided kinematic camera pilot."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from .core import camera_pose_matches, checked_path, pose_matches
from .dense_cache import save_json
from .dense_capture import decode_instance_ids
from .local_pairs import sha256


ACTIONS = {
    'forward': (0, .5, 0, 0), 'back': (1, -.5, 0, 0),
    'left': (2, 0, -.5, 0), 'right': (3, 0, .5, 0),
    'yaw_left': (4, 0, 0, -15), 'yaw_right': (5, 0, 0, 15),
    'hold_capture': (6, 0, 0, 0),
}
ACTION_SPEC = {
    'version': 'kinematic-v1-050cm-15deg', 'fixed_delta_s': .05, 'motion_ticks': 4,
    'sensor_settle_ticks': 4, 'mode': 'kinematic_camera_sim',
    'coordinates': 'CARLA world xyz metres; pose [x,y,z,pitch,yaw,roll] in degrees; '
                   'forward/right relative to the heading at action start; positive yaw turns right',
    'capture_timing': 'after 4 interpolation ticks, 4 settle ticks, and a synchronized sensor pair',
    'actions': [{'id': i, 'name': name, 'forward_m': f, 'right_m': r, 'yaw_delta_deg': y,
                 'command_duration_s': .2} for name, (i, f, r, y) in ACTIONS.items()],
    'terminal': 'stop or horizon is recorded in episode.json; no additional observation',
}
QUALITY_SPEC = {
    'version': 'v1-provisional-518', 'letterbox_size': 518,
    'min_native_area': 900, 'min_effective_area': 225, 'min_effective_short_side': 10,
    'boundary_rule': 'any boundary contact is weak; severity requires preview review',
    'reference_min_long_side': 240, 'reference_min_short_side': 140,
    'interpretation': 'nearest-neighbor mask resize to nominal 518 letterbox; '
                      'human review checks blur, truncation and identity',
}
TEMPLATES = {
    'discover': ((0, 0, 0), [('yaw_right', 10), ('left', 3), ('yaw_right', 2),
                             ('hold_capture', 2), ('forward', 3)]),
    'compare': ((0, -8, 75), [('yaw_right', 2), ('forward', 5), ('left', 4),
                               ('hold_capture', 2), ('yaw_left', 3), ('right', 4)]),
    'reappear': ((0, 0, 180), [('yaw_right', 6), ('hold_capture', 2), ('yaw_left', 6),
                                ('forward', 4), ('right', 3)]),
    'repeat_then_useful': ((0, 0, 120), [('hold_capture', 3), ('yaw_right', 4), ('forward', 5),
                                          ('left', 4), ('yaw_left', 3)]),
}
INPUT_FIELDS = {'reference', 'rgb', 'time_s', 'previous_action_id', 'legal_action_mask'}


def load_json(path):
    return Path(path).read_text(encoding='utf-8-sig')


def json_rows(path):
    return [json.loads(line) for line in load_json(path).splitlines() if line]


def append_row(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pixel_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def action_trace(start, name):
    _, forward, right, yaw = ACTIONS[name]
    angle = math.radians(start[4])
    dx = forward * math.cos(angle) - right * math.sin(angle)
    dy = forward * math.sin(angle) + right * math.cos(angle)
    count = ACTION_SPEC['motion_ticks']
    return [[start[0] + dx * i / count, start[1] + dy * i / count, start[2], start[3],
             (start[4] + yaw * i / count + 180) % 360 - 180, start[5]]
            for i in range(1, count + 1)]


def make_plan(config, stage):
    """Each layout owns one split, including every route and derived reference."""
    rng = random.Random(config['seed'])
    cases = []
    for index, layout in enumerate(config['layouts']):
        if stage == 'smoke' and index >= 3:
            break
        if stage == 'smoke':
            templates = [layout['smoke_template']]
        else:
            templates = list(TEMPLATES)
        group = 'layout-' + digest({k: v for k, v in layout.items()
                                    if k not in ('split', 'target', 'smoke_template')})[:16]
        for template in templates:
            start, route = TEMPLATES[template]
            start = list(start)
            target = layout['target']
            if target == 'B' and template in ('discover', 'reappear'):
                start[2] = (start[2] + 180) % 360
            actions = [name for name, count in route for _ in range(count)]
            actions.extend(['hold_capture'] * rng.randrange(3))
            cases.append(dict(scene_id=f'scene_{index:02d}', group=group, split=layout['split'],
                              episode_id=f'ep_{index:02d}_{template}', layout=layout,
                              template=template, start=start, actions=actions,
                              seed=rng.getrandbits(32), target=target))
    return cases


def instance_labels(raw, mappings, vehicle_tags):
    ids = decode_instance_ids(raw)
    tags = raw[:, :, 0]
    masks = {uid: np.isin(ids, item['ids']) & np.isin(tags, item['tags'])
             for uid, item in mappings.items()}
    mapped = np.zeros(ids.shape, dtype=bool)
    for mask in masks.values():
        if np.any(mapped & mask):
            raise ValueError('verified instance masks overlap')
        mapped |= mask
    ignore = ((np.isin(tags, vehicle_tags) | (tags == 0)) & ~mapped
              if vehicle_tags else ~mapped)
    return masks, ignore


def recognition_quality(mask):
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    count = int(xs.size)
    scale = QUALITY_SPEC['letterbox_size'] / max(width, height)
    effective_area = 0
    if count:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox = [x0, y0, x1, y1]
        edge = x0 == 0 or y0 == 0 or x1 == width or y1 == height
        resized = Image.fromarray(mask).resize((max(1, round(width * scale)),
                                                max(1, round(height * scale))), Image.Resampling.NEAREST)
        ey, ex = np.nonzero(np.asarray(resized))
        effective_area = int(ex.size)
        short_side = int(min(ex.max() - ex.min() + 1, ey.max() - ey.min() + 1)) if ex.size else 0
        small = (count < QUALITY_SPEC['min_native_area'] or
                 effective_area < QUALITY_SPEC['min_effective_area'] or
                 short_side < QUALITY_SPEC['min_effective_short_side'])
    else:
        bbox, edge, small = None, False, False
    return dict(visibility_present=bool(count), visible_pixels=count,
                visible_fraction=count / (width * height), bbox_xyxy=bbox,
                touches_boundary=edge, too_small=small, severely_truncated=None if edge else False,
                effective_area_518=effective_area,
                recognition_quality=('unknown' if not count else 'weak' if edge or small else 'usable'),
                quality_rule=QUALITY_SPEC['version'])


def iter_v1_inputs(root, episode_path):
    """Decode the allowlist. Paths and all other fields stay outside model features."""
    episode = checked_path(root, episode_path)
    for row in json_rows(episode / 'inputs.jsonl'):
        if set(row) != INPUT_FIELDS:
            raise ValueError('unexpected field in V1 inputs')
        with Image.open(checked_path(root, row['rgb'])) as rgb, Image.open(
            checked_path(root, row['reference'])
        ) as reference:
            yield dict(rgb=np.asarray(rgb.convert('RGB')).copy(),
                       reference=np.asarray(reference.convert('RGB')).copy(),
                       time_s=row['time_s'], previous_action_id=row['previous_action_id'],
                       legal_action_mask=row['legal_action_mask'])


def _image(root, path, expected, mode):
    filename = checked_path(root, path)
    if sha256(filename) != expected:
        raise ValueError('file hash mismatch: ' + path)
    with Image.open(filename) as image:
        if image.mode != mode:
            raise ValueError('image mode mismatch: ' + path)
        return np.array(image)


def validate_episode(root, case, *, allow_pending=False):
    directory = checked_path(root, case['path'])
    episode = json.loads(load_json(directory / 'episode.json'))
    scene = json.loads(load_json(directory.parent.parent / 'scene.json'))
    status_ok = ('captured', 'complete') if allow_pending else ('complete',)
    if (episode['status'] not in status_ok or episode.get('failure') or
        episode.get('cleanup_errors') or episode['case'] != case or scene['group'] != case['group']):
        raise ValueError('episode is not complete for the configured case')
    if episode['mode'] != 'kinematic_camera_sim' or episode['terminator'] != 'scripted_horizon':
        raise ValueError('unexpected episode mode or termination')
    frames = json_rows(directory / 'frames.jsonl')
    inputs = json_rows(directory / 'inputs.jsonl')
    audit = json_rows(directory / 'audit' / 'observations.jsonl')
    if len(frames) != len(inputs) or len(frames) != len(audit) or not 16 <= len(frames) <= 32:
        raise ValueError('incomplete observation records')
    if len(frames) != len(case['actions']) + 1:
        raise ValueError('observation count does not match executed route')
    ref = episode['reference']
    full = _image(root, ref['full'], ref['hashes'][ref['full']], 'RGB')
    obj = _image(root, ref['object'], ref['hashes'][ref['object']], 'RGB')
    refmask = _image(root, ref['mask'], ref['hashes'][ref['mask']], 'L')
    raw = _image(root, ref['raw'], ref['hashes'][ref['raw']], 'RGB')
    mappings = episode['instance_mapping']
    if len(mappings) != 2 or episode['instance_id_method'] != 'remove_restore_probe':
        raise ValueError('missing verified opaque instance mapping')
    target = episode['target_uid']
    ref_masks, _ = instance_labels(raw, mappings, episode['vehicle_tags'])
    x0, y0, x1, y1 = ref['crop_xyxy']
    if (target != ref['target_uid'] or target != case['scene_id'] + ':' + case['target'] or
        pixel_digest(full) != ref['rgb_sensor_sha256'] or raw.shape != full.shape or
        not np.array_equal(refmask, ref_masks[target].astype(np.uint8) * 255) or
        not np.array_equal(obj, full[y0:y1, x0:x1]) or
        ref['quality'] != recognition_quality(ref_masks[target])):
        raise ValueError('reference RGB, crop, identity, or mask provenance mismatch')
    if ref['quality']['recognition_quality'] != 'usable':
        raise ValueError('reference failed quality screening')
    check_pair(ref['audit'])
    box = ref['quality']['bbox_xyxy']
    ref_sides = [box[2] - box[0], box[3] - box[1]]
    if (max(ref_sides) < QUALITY_SPEC['reference_min_long_side'] or
        min(ref_sides) < QUALITY_SPEC['reference_min_short_side']):
        raise ValueError('reference box fails the fixed pilot size threshold')
    stats = Counter()
    quality = Counter()
    ids = set()
    previous = None
    present, usable = [], []
    for step, (frame, runtime, observation) in enumerate(zip(frames, inputs, audit, strict=True)):
        if frame['step_index'] != step or frame['episode_id'] != case['episode_id']:
            raise ValueError('invalid step or episode ID')
        if frame['frame_id'] in ids or frame['annotation_validity'] != 'valid':
            raise ValueError('duplicate frame ID or invalid annotation')
        ids.add(frame['frame_id'])
        check_pair(observation)
        if (observation['frame_id'] != frame['frame_id'] or frame['target_uid'] != target or
            frame['sensor_timestamp'] != observation['rgb']['timestamp'] or
            frame['sensor_frame'] != observation['rgb']['frame'] or
            frame['sensor_frame'] <= ref['audit']['rgb']['frame']):
            raise ValueError('frame identity or reference separation mismatch')
        if not previous:
            if frame['action_from_previous'] is not None or observation['action_from_previous'] is not None:
                raise ValueError('first frame must have no prior action')
        else:
            action = observation['action_from_previous']
            check_transition(previous, observation, action, case['actions'][step - 1])
            if frame['action_from_previous'] != {k: action[k] for k in ('id', 'name', 'status')}:
                raise ValueError('frame action and audit action disagree')
        previous = observation
        expected_input = dict(reference=ref['object'], rgb=frame['image_path'],
                              time_s=frame['sensor_timestamp'] - frames[0]['sensor_timestamp'],
                              previous_action_id=(None if step == 0 else
                                                  ACTIONS[case['actions'][step - 1]][0]),
                              legal_action_mask=None)
        if runtime != expected_input:
            raise ValueError('V1 input allowlist or runtime ordering mismatch')
        rgb = _image(root, frame['image_path'], observation['rgb']['file_sha256'], 'RGB')
        raw = _image(root, observation['instance']['path'], observation['instance']['file_sha256'], 'RGB')
        if (rgb.shape != raw.shape or rgb.shape[:2] != (observation['rgb']['height'],
                                                       observation['rgb']['width']) or
            pixel_digest(rgb) != observation['rgb']['sensor_sha256']):
            raise ValueError('raw RGB integrity or paired dimensions mismatch')
        masks, ignored = instance_labels(raw, mappings, episode['vehicle_tags'])
        if set(frame['objects']) != set(mappings):
            raise ValueError('missing per-object annotations')
        for uid, mask in masks.items():
            label = frame['objects'][uid]
            if label['quality'] != recognition_quality(mask):
                raise ValueError('visibility or recognition quality mismatch')
            check_mask(root, label['mask'], mask)
        check_mask(root, frame['ignore_mask'], ignored)
        has_target = bool(masks[target].any())
        has_other = any(mask.any() for uid, mask in masks.items() if uid != target)
        if frame['visibility_present'] != has_target:
            raise ValueError('target presence mismatch')
        present.append(has_target)
        usable.append(frame['objects'][target]['quality']['recognition_quality'] == 'usable')
        stats[('both' if has_other else 'target_only') if has_target else
              ('distractor_only' if has_other else 'neither')] += 1
        quality[frame['objects'][target]['quality']['recognition_quality']] += 1
    if (episode['observation_count'] != len(frames) or
        episode['start'] != audit[0]['rgb'] or episode['end'] != audit[-1]['rgb']):
        raise ValueError('episode endpoint or length mismatch')
    return dict(observations=len(frames), visibility=dict(stats), quality=dict(quality),
                first_target_step=next((i for i, seen in enumerate(present) if seen), None),
                absent_then_visible=not present[0] and any(present),
                disappeared_and_reappeared=any(present[i - 1] and not present[i] and any(present[i+1:])
                                              for i in range(1, len(present))),
                held_steps=sum(a == 'hold_capture' for a in case['actions']),
                usable_after_hold=any(a == 'hold_capture' and any(usable[i+2:])
                                      for i, a in enumerate(case['actions'])),
                reference=ref['quality'])


def check_mask(root, record, expected):
    if expected.any():
        if record is None or not np.array_equal(_image(root, record['path'], record['sha256'], 'L'),
                                               expected.astype(np.uint8) * 255):
            raise ValueError('mask missing or differs from verified raw instance pixels')
    elif record is not None:
        raise ValueError('empty masks should be represented by null paths')


def check_pair(observation):
    rgb, instance = observation['rgb'], observation['instance']
    if any(rgb[k] != instance[k] for k in ('frame', 'timestamp', 'width', 'height', 'intrinsics')):
        raise ValueError('RGB and instance sensor frames or calibrations do not match')
    if not camera_pose_matches(rgb['pose'], instance['pose'], .005, .05):
        raise ValueError('RGB and instance camera pose mismatch')


def check_transition(previous, current, action, name):
    start, end = previous['rgb'], current['rgb']
    if (not action or action['name'] != name or action['id'] != ACTIONS[name][0] or
        action['status'] != 'executed' or action['collision_check'] != 'unknown' or
        action['start_pose'] != start['pose'] or action['end_pose'] != end['pose'] or
        action['start_frame'] != start['frame'] or action['end_frame'] != end['frame'] or
        action['observed_elapsed_s'] != end['timestamp'] - start['timestamp'] or
        action['command_duration_s'] != .2 or end['frame'] <= start['frame'] or
        end['timestamp'] <= start['timestamp']):
        raise ValueError('missing or inconsistent action transition')
    expected = action_trace(start['pose'], name)
    ticks = action['interpolation']
    if len(ticks) != len(expected):
        raise ValueError('incomplete interpolation trace')
    previous_tick = start['frame']
    for tick, desired in zip(ticks, expected, strict=True):
        if not previous_tick < tick['frame'] < end['frame'] or tick['commanded_pose'] != desired:
            raise ValueError('invalid interpolation tick or commanded pose')
        if not pose_matches(tick['actual_pose'], desired):
            raise ValueError('motion interpolation deviated from command')
        previous_tick = tick['frame']
    if not pose_matches(end['pose'], expected[-1]):
        raise ValueError('end pose does not match action command')


def validate_splits(cases):
    groups, lineage, ids = {}, {}, set()
    for case in cases:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', case['episode_id']) or case['episode_id'] in ids:
            raise ValueError('unsafe or duplicate episode ID')
        ids.add(case['episode_id'])
        if groups.setdefault(case['group'], case['split']) != case['split']:
            raise ValueError('layout group leaks across splits')
        if lineage.setdefault(case.get('source_episode_id', case['episode_id']),
                              case['split']) != case['split']:
            raise ValueError('episode lineage leaks across splits')
    return groups


def validate_dataset(root):
    root = Path(root).resolve()
    manifest = json.loads(load_json(root / 'dataset_manifest.json'))
    groups = validate_splits(manifest['cases'])
    if json.loads(load_json(root / 'splits.json')) != groups:
        raise ValueError('split file differs from layout groups')
    if json.loads(load_json(root / 'action_spec.json')) != ACTION_SPEC:
        raise ValueError('action specification differs from the collector')
    rows, errors = {}, {}
    for case in manifest['cases']:
        try:
            rows[case['episode_id']] = validate_episode(root, case)
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors[case['episode_id']] = str(error)
    if manifest['status'] == 'complete' and errors:
        raise ValueError(f'complete manifest has invalid episodes: {errors}')
    result = dict(status='verified' if not errors else 'incomplete', episodes=rows, errors=errors,
                  layout_count=len({c['group'] for c in manifest['cases'] if c['episode_id'] in rows}),
                  observation_count=sum(row['observations'] for row in rows.values()),
                  templates=dict(Counter(c['template'] for c in manifest['cases'] if c['episode_id'] in rows)),
                  appearance_assets=sorted({(c['blueprint'], c['color'])
                                            for c in manifest['config']['capture']['candidates']}) if rows else [],
                  coverage=dict(sum((Counter(r['visibility']) for r in rows.values()), Counter())),
                  quality=dict(sum((Counter(r['quality']) for r in rows.values()), Counter())))
    save_json(root / 'validation.json', result)
    return result


def smoke_coverage_gaps(result, cases):
    if result['errors'] or len(result['episodes']) != 3:
        return ['three validated smoke episodes required']
    by_template = {c['template']: result['episodes'][c['episode_id']] for c in cases}
    gaps = []
    if not by_template.get('discover', {}).get('absent_then_visible'):
        gaps.append('discover: target should first be out of view, then visible')
    if not by_template.get('compare', {}).get('visibility', {}).get('both'):
        gaps.append('compare: at least one two-vehicle observation required')
    if not by_template.get('reappear', {}).get('disappeared_and_reappeared'):
        gaps.append('reappear: a visible, absent, visible progression is required')
    if not result['coverage'].get('distractor_only'):
        gaps.append('at least one distractor-only observation required')
    if not any(item['usable_after_hold'] for item in result['episodes'].values()):
        gaps.append('at least one usable observation after a hold is required')
    return gaps
