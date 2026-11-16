"""Deterministic multi-background, two-car capture plans; no CARLA or model imports."""
import copy
import hashlib
import json
import math
import random

from .scene import validate_scene
from .search_grid import build_graph


def seed_for(seed, *parts):
    text = json.dumps([seed, *parts], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def finite(value, name, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')


def validate_config(config):
    if type(config['seed']) is not int or not isinstance(config['map'], str) or not config['map']:
        raise ValueError('need integer seed and explicit map name')
    cap, grid, sampling, policy = (config[k] for k in ('capture', 'grid', 'sampling', 'route'))
    if len(cap['candidates']) != 2:
        raise ValueError('exactly two distinct vehicle candidates required')
    validate_scene(dict(candidates=cap['candidates'], target_identity=cap['candidates'][0]['identity']))
    for name in ('width', 'height', 'port', 'settle_ticks'):
        positive_int(cap[name], name)
    for name in ('timeout', 'height_m'):
        finite(cap[name], name, .01)
    finite(cap['fov'], 'fov', 1)
    finite(cap['pitch_deg'], 'pitch_deg')
    if cap['fov'] >= 179 or not -89 <= cap['pitch_deg'] <= -10:
        raise ValueError('need valid FOV and downward camera pitch')
    for value in cap['weather'].values():
        finite(value, 'weather')
    for name in ('azimuth', 'elevation', 'distance'):
        finite(cap['reference'][name], 'reference '+name)
    if cap['reference']['distance'] <= 0:
        raise ValueError('positive reference distance required')
    finite(config['minimum_vehicle_gap_m'], 'minimum_vehicle_gap_m', .01)
    positive_int(grid['size'], 'grid size', 3)
    for name in ('spacing_m', 'move_speed', 'turn_speed', 'hold_s'):
        finite(grid[name], name, .01)
    build_graph(grid['size'], grid['spacing_m'], [0, 0, cap['height_m']], cap['pitch_deg'],
                move_speed=grid['move_speed'], turn_speed=grid['turn_speed'], hold_s=grid['hold_s'])
    for name in ('backgrounds', 'layouts_per_background', 'max_attempts'):
        positive_int(sampling[name], name)
    positive_int(sampling['train_backgrounds'], 'train_backgrounds')
    if sampling['train_backgrounds'] >= sampling['backgrounds']:
        raise ValueError('reserve at least one entire validation background')
    for name in ('min_background_distance_m', 'max_anchor_tilt_deg', 'max_settled_tilt_deg',
                 'max_vertical_shift_m'):
        finite(sampling[name], name, .01)
    finite(sampling['center_jitter_m'], 'center_jitter_m', 0)
    # Keep distinct background footprints apart, including the outer camera grid.
    footprint = math.sqrt(2)*(grid['size']-1)*grid['spacing_m']/2
    if sampling['min_background_distance_m'] <= 2*(footprint+sampling['center_jitter_m']):
        raise ValueError('background distance must separate camera grid footprints')
    for key in ('separation_m', 'yaw_offset_deg'):
        bounds = sampling[key]
        if len(bounds) != 2:
            raise ValueError(f'{key} needs two bounds')
        for b in bounds:
            finite(b, key)
        if bounds[0] >= bounds[1]:
            raise ValueError(f'{key} bounds must increase')
    if sampling['separation_m'][0] <= config['minimum_vehicle_gap_m']:
        raise ValueError('separation must exceed required vehicle gap')
    anchors = sampling.get('anchor_spawn_indices')
    if anchors is not None:
        if len(anchors) != sampling['backgrounds'] or len(set(anchors)) != len(anchors):
            raise ValueError('explicit anchors need one distinct index per background')
        for a in anchors:
            positive_int(a, 'anchor index', 0)
    for name in ('min_view_pixels', 'min_outward', 'max_outward', 'beam_width', 'max_routes'):
        positive_int(policy[name], name)
    if not 3 <= policy['min_outward'] <= policy['max_outward'] <= 33:
        raise ValueError('need 3 <= min_outward <= max_outward <= 33')
    return config


def choose_backgrounds(config, spawn_poses):
    """Farthest-point selection in world XY; never load/change a CARLA map."""
    validate_config(config)
    cfg = config['sampling']
    eligible = []
    for i, pose in enumerate(spawn_poses):
        if len(pose) != 6 or any(not math.isfinite(x) for x in pose):
            raise ValueError('spawn poses must contain six finite values')
        if max(abs(pose[3]), abs(pose[5])) <= cfg['max_anchor_tilt_deg']:
            eligible.append(i)
    explicit = cfg.get('anchor_spawn_indices')
    if explicit is not None:
        if any(i not in eligible for i in explicit):
            raise ValueError('explicit anchor unavailable or too steep on this map')
        selected = list(explicit)
    else:
        if not eligible:
            raise ValueError('no level background anchors available')
        rng = random.Random(seed_for(config['seed'], config['map'], 'backgrounds'))
        selected = [rng.choice(eligible)]
        while len(selected) < cfg['backgrounds']:
            remaining = [i for i in eligible if i not in selected]
            if not remaining:
                raise ValueError('not enough distinct background anchors')
            nxt = max(remaining, key=lambda i: (
                min(math.dist(spawn_poses[i][:2], spawn_poses[j][:2]) for j in selected), -i))
            selected.append(nxt)
    if any(math.dist(spawn_poses[a][:2], spawn_poses[b][:2]) < cfg['min_background_distance_m']
           for j, a in enumerate(selected) for b in selected[:j]):
        raise ValueError('cannot fit requested separated backgrounds; change map/count or explicit anchors')
    return [dict(id=f'background_{j:02d}', anchor_spawn_index=i, anchor_pose=list(spawn_poses[i]),
                 split='train' if j < cfg['train_backgrounds'] else 'val',
                 group='background-'+hashlib.sha256(json.dumps(
                     [config['map'], i, list(spawn_poses[i])], separators=(',', ':')).encode()).hexdigest()[:16])
            for j, i in enumerate(selected)]


def slots_for(config):
    validate_config(config)
    cfg = config['sampling']
    return [dict(scene_id=f'scene_{b:02d}_{n:02d}', background_id=f'background_{b:02d}',
                 split='train' if b < cfg['train_backgrounds'] else 'val')
            for b in range(cfg['backgrounds']) for n in range(cfg['layouts_per_background'])]


def case_for(config, background, slot, attempt):
    """An attempt has its own seed: failed layouts never change later scenes."""
    positive_int(attempt, 'attempt', 1)
    cfg = config['sampling']
    seed = seed_for(config['seed'], background['id'], slot['scene_id'], attempt)
    rng = random.Random(seed)
    center = [rng.uniform(-cfg['center_jitter_m'], cfg['center_jitter_m']) for _ in range(2)]
    separation = rng.uniform(*cfg['separation_m'])
    angle = rng.uniform(-math.pi, math.pi)
    delta = [separation*math.cos(angle)/2, separation*math.sin(angle)/2]
    placements = [[center[0]+sign*delta[0], center[1]+sign*delta[1],
                   rng.uniform(*cfg['yaw_offset_deg'])] for sign in (-1, 1)]
    return dict(scene_id=slot['scene_id'], path=f'attempts/{slot["scene_id"]}/attempt_{attempt:02d}',
                seed=seed, group=background['group'], split=background['split'],
                layout=dict(anchor_spawn_index=background['anchor_spawn_index'],
                            placements=placements, grid_center_offset=center),
                randomization=dict(background_id=background['id'], attempt=attempt,
                                   seed=seed, anchor_pose=background['anchor_pose'],
                                   position_units='anchor-local forward/right metres',
                                   yaw_units='independent offsets in degrees relative to anchor yaw'))


def dry_plan(config):
    """Logical plan only. World positions require the actual map's spawn-point list."""
    slots = slots_for(config)
    cfg = config['sampling']
    examples = []
    for slot in slots:
        background = dict(id=slot['background_id'], split=slot['split'], group='unresolved',
                          anchor_spawn_index=None, anchor_pose=None)
        examples.append(case_for(config, background, slot, 1))
    images = config['grid']['size']**2*8
    return dict(captured=False, carla_connected=False, models_run=False,
                backgrounds=cfg['backgrounds'], requested_scenes=len(slots),
                images_per_attempt=images, images_if_first_attempts_pass=images*len(slots),
                maximum_attempts=len(slots)*cfg['max_attempts'],
                maximum_grid_images=images*len(slots)*cfg['max_attempts'],
                independent_reference_images_per_attempt=2,
                note='No images are created. References/calibration add captures. Anchors resolve at preflight. '
                     'Return replays outward images; raw grid images and route steps are counted separately.',
                config=copy.deepcopy(config), slots=slots, first_attempts=examples)
