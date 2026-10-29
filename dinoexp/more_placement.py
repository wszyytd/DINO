"""Bounded test-only placement retries, before either reference is collected."""

from copy import deepcopy
import math

from capture import as_transform, find_blueprint
from .scene import candidate_poses


def _test_group(manifest, completed):
    cases = [c for c in manifest['plan']['cases'] if c['split'] == 'test']
    if any(c['name'] in completed for c in cases):
        raise ValueError('cannot change test placement after a test reference was completed')
    if (len(cases) != 2 or len({c['capture_group'] for c in cases}) != 1
            or {c['config']['target_identity'] for c in cases} != {'A', 'B'}):
        raise ValueError('placement retry requires one unstarted A/B test reference group')
    keys = ('candidates', 'separation', 'anchor_spawn_index', 'query_yaw_mode', 'views')
    if any(cases[0]['config'].get(k) != cases[1]['config'].get(k) for k in keys):
        raise ValueError('test reference pair has inconsistent physical configuration')
    return cases


def placement_options(manifest, completed):
    cases = _test_group(manifest, completed)
    cfg = cases[0]['config']
    original = manifest.get('original_plan', manifest['plan'])
    reserved = original['split_anchors']['test']
    forbidden = {c['config']['anchor_spawn_index'] for c in manifest['plan']['cases']
                 if c['split'] != 'test'}
    if set(reserved) & forbidden:
        raise ValueError('reserved test anchors overlap training/validation')
    # Try the other reserved test positions before the position that failed.
    anchors = [a for a in reserved if a != cfg['anchor_spawn_index']]
    anchors += [a for a in reserved if a == cfg['anchor_spawn_index']]
    spacings = list(dict.fromkeys([cfg['separation'], max(cfg['separation'], 12),
                                   max(cfg['separation'], 16)]))
    return [dict(anchor_spawn_index=a, separation=s) for a in anchors for s in spacings]


def apply_placement(manifest, completed, placement):
    option = {k: placement[k] for k in ('anchor_spawn_index', 'separation')}
    if option not in placement_options(manifest, completed):
        raise ValueError('new placement must use a reserved test anchor and bounded spacing')
    result = deepcopy(manifest)
    cases = _test_group(result, completed)
    previous = {k: cases[0]['config'][k] for k in option}
    for case in cases:
        case['config'].update(option)
    result['plan']['split_anchors']['test'] = [option['anchor_spawn_index']]
    result.setdefault('placement_revisions', []).append(dict(
        cases=[c['name'] for c in cases], previous=previous, selected=option,
        attempts=placement.get('attempts', []),
        reason='Two-vehicle spawn feasibility preflight; no matching scores used.'))
    return result


def apply_camera_radius(manifest, completed, radius):
    """Raise pending test queries to cap XY offset, preserving slant range/azimuth.

    This is an explicit high-angle test variant, not a visibility certificate.
    The normal capture quality/crop checks still decide whether a frame is saved.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('test camera radius must be finite and positive')
    result = deepcopy(manifest)
    cases = _test_group(result, completed)
    previous = deepcopy(cases[0]['config']['views'])
    adjusted = deepcopy(previous)
    changed = False
    for view in adjusted:
        minimum = math.degrees(math.acos(min(1.0, radius / view['distance'])))
        if minimum > view['elevation'] + 1e-9:
            view['elevation'] = minimum
            if not view['name'].endswith('_raised'):
                view['name'] += '_raised'
            changed = True
    if not changed:
        return result
    for case in cases:
        case['config']['views'] = deepcopy(adjusted)
    result['plan']['test_camera_radius_m'] = radius
    result['plan']['test_view_scope'] = ('Raised query elevations at original azimuths/ranges; '
        'low-angle references unchanged. This does not evaluate the original low-angle test views.')
    result.setdefault('camera_revisions', []).append(dict(
        cases=[c['name'] for c in cases], horizontal_radius_m=radius,
        previous_views=previous, selected_views=deepcopy(adjusted),
        reason='Synchronized failed frames show building interior at low-angle camera positions.'))
    return result


def probe_placement(carla, world, manifest, completed, catalog, take_tick=False):
    cfg = _test_group(manifest, completed)[0]['config']
    synchronous = world.get_settings().synchronous_mode
    if synchronous and not take_tick:
        raise RuntimeError('placement probe requires exclusive tick ownership in synchronous mode')
    points = {p['index']: p['pose'] for p in catalog['spawn_points']}
    library, attempts = world.get_blueprint_library(), []
    for option in placement_options(manifest, completed):
        index, spacing = option['anchor_spawn_index'], option['separation']
        if index not in points:
            raise ValueError(f'reserved test anchor unavailable: {index}')
        poses = candidate_poses(points[index], spacing,
                                [c.get('yaw_offset', 0) for c in cfg['candidates']])
        owned, placed, cleanup_errors = [], [], []
        try:
            for candidate, pose in zip(cfg['candidates'], poses, strict=True):
                bp = find_blueprint(library, candidate['blueprint'])
                bp.set_attribute('color', candidate['color'])
                if bp.has_attribute('role_name'):
                    bp.set_attribute('role_name', 'dino_placement_probe')
                actor = world.try_spawn_actor(bp, as_transform(carla, pose))
                if actor is None:
                    break
                owned.append(actor)
                actor.set_simulate_physics(False)
                placed.append(candidate['identity'])
        finally:
            for actor in reversed(owned):
                try:
                    if not actor.destroy():
                        cleanup_errors.append('destroy returned false')
                except Exception as error:
                    cleanup_errors.append(str(error))
            if cleanup_errors:
                raise RuntimeError(f'placement probe cleanup failed: {cleanup_errors}')
            if owned:
                # Flush actor destruction before another placement or photography.
                if synchronous:
                    world.tick(cfg['timeout'])
                else:
                    world.wait_for_tick(cfg['timeout'])
        attempts.append(dict(option, placed=placed))
        print(f'[placement] test anchor={index}, separation={spacing}m: '
              f'{len(placed)}/2 spawned; probe actors removed', flush=True)
        if len(placed) == 2:
            return dict(option, attempts=attempts)
    raise RuntimeError('No feasible two-vehicle placement at reserved test anchors; '
                       'successful training/validation photos were not changed. '
                       f'Attempts: {attempts}')
