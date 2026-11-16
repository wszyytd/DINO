"""Audited, resumable CARLA collection; imported without connecting or loading models."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .core import checked_path, pose_values
from .grid_search_capture import capture_layout
from .grid_search_validation import read_layout
from .livos_roundtrip_routes import choose_routes
from .multiscene_plan import case_for, choose_backgrounds, slots_for, validate_config
from .search_capture import preflight
from .search_episodes import QUALITY_SPEC
from .search_grid import action_spec

REPO = Path(__file__).resolve().parents[1]
SOURCES = ('capture_multiscene_roundtrip.py', 'run_multiscene_roundtrip_test.py',
           'dinoexp/multiscene_plan.py', 'dinoexp/multiscene_capture.py',
           'dinoexp/multiscene_routes.py', 'capture.py', 'dinoexp/core.py',
           'dinoexp/dense_capture.py', 'dinoexp/dense_cache.py', 'dinoexp/scene.py',
           'dinoexp/search_capture.py', 'dinoexp/search_episodes.py',
           'dinoexp/search_grid.py', 'dinoexp/grid_search_capture.py',
           'dinoexp/grid_search_validation.py', 'dinoexp/livos_roundtrip_routes.py',
           'dinoexp/region_sequence_routes.py', 'dinoexp/local_pairs.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def provenance():
    # Windows capture and Linux evaluation must agree despite Git CRLF conversion.
    return {name: hashlib.sha256((REPO/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCES}


@contextmanager
def capture_lock(root):
    path = root/'.capture.lock'
    # Exclusive creation also protects against two resume processes.
    with path.open('x', encoding='utf-8') as file:
        file.write(now()+'\n')
    try:
        yield
    finally:
        path.unlink()


def routes_for(case, scene, graph, raw_frames, config):
    frames = {sid: dict(scene=case['scene_id'], state=sid, objects={
        uid: dict(pixels=row['quality']['visible_pixels'], boundary=row['quality']['touches_boundary'])
        for uid, row in frame['objects'].items()}) for sid, frame in raw_frames.items()}
    cfg = config['route']
    routes = choose_routes(case['scene_id'], graph, frames, min_pixels=cfg['min_view_pixels'],
                           max_routes=cfg['max_routes'], max_outward=cfg['max_outward'],
                           beam_width=cfg['beam_width'])
    routes = [r for r in routes if r['outward_frames'] >= cfg['min_outward']]
    if not routes:
        raise ValueError(f'no qualifying roundtrip with >= {cfg["min_outward"]} different outward views')
    for obj in scene['objects'].values():
        pose = obj['pose']
        anchor = scene['anchor']
        if max(abs(pose[3]), abs(pose[5])) > config['sampling']['max_settled_tilt_deg']:
            raise ValueError('settled car tilt too large')
        if abs(pose[2]-(anchor[2]+.3)) > config['sampling']['max_vertical_shift_m']:
            raise ValueError('settled car height too far from requested road level')
    return routes


def create_manifest(root, config, backgrounds, versions, spawn_poses):
    root.mkdir(parents=True, exist_ok=False)
    slots = [dict(**slot, status='pending', attempts=[]) for slot in slots_for(config)]
    manifest = dict(schema='multiscene_roundtrip_collection_v1', status='incomplete',
                    collection_status='capturing', mode='discrete_view_graph', stage='random_multiscene',
                    started_utc=now(), config=config, code=provenance(), carla_versions=versions,
                    backgrounds=backgrounds, spawn_poses=spawn_poses, slots=slots, cases=[],
                    sampling_note='Offline GT-based route acceptance, not learned navigation. '
                                  'Cars are static; the return reuses outward RGB images.')
    atomic_json(root/'action_spec.json', action_spec(config['grid']))
    atomic_json(root/'quality_spec.json', QUALITY_SPEC)
    atomic_json(root/'splits.json', {b['group']: b['split'] for b in backgrounds})
    atomic_json(root/'dataset_manifest.json', manifest)
    return manifest


def check_manifest(root, config=None):
    manifest = read_json(root/'dataset_manifest.json')
    if manifest.get('schema') != 'multiscene_roundtrip_collection_v1':
        raise ValueError('not a multi-scene collection')
    validate_config(manifest['config'])
    if config is not None and config != manifest['config']:
        raise ValueError('resume requires exactly the original configuration')
    if manifest['code'] != provenance():
        raise ValueError('capture code changed; use the original checkout or a new collection')
    bg = manifest['backgrounds']
    selected = choose_backgrounds(manifest['config'], manifest['spawn_poses'])
    if selected != bg or len(bg) != manifest['config']['sampling']['backgrounds']:
        raise ValueError('saved backgrounds do not match deterministic selection')
    if read_json(root/'splits.json') != {b['group']: b['split'] for b in bg}:
        raise ValueError('background split mismatch')
    if (read_json(root/'action_spec.json') != action_spec(manifest['config']['grid']) or
            read_json(root/'quality_spec.json') != QUALITY_SPEC):
        raise ValueError('action/quality specification changed')
    expected = slots_for(manifest['config'])
    if [{k: s[k] for k in ('scene_id', 'background_id', 'split')} for s in manifest['slots']] != expected:
        raise ValueError('scene slot list changed')
    backgrounds = {b['id']: b for b in bg}
    accepted = []
    for slot in manifest['slots']:
        attempts = slot['attempts']
        if len(attempts) > manifest['config']['sampling']['max_attempts']:
            raise ValueError('attempt limit exceeded')
        for i, attempt in enumerate(attempts, 1):
            expected_case = case_for(manifest['config'], backgrounds[slot['background_id']], slot, i)
            if attempt['case'] != expected_case:
                raise ValueError('saved random case differs from deterministic attempt')
            if attempt['status'] not in ('running', 'interrupted', 'rejected', 'accepted'):
                raise ValueError('invalid attempt status')
            if attempt['status'] == 'running' and (i != len(attempts) or slot['status'] != 'pending'):
                raise ValueError('only the last pending attempt may be running')
        successes = [a for a in attempts if a['status'] == 'accepted']
        if slot['status'] == 'accepted':
            if len(successes) != 1 or attempts[-1] is not successes[0]:
                raise ValueError('inconsistent accepted slot')
            accepted.append(successes[0]['case'])
        elif successes or slot['status'] not in ('pending', 'exhausted'):
            raise ValueError('invalid slot status')
        if slot['status'] == 'exhausted' and len(attempts) != manifest['config']['sampling']['max_attempts']:
            raise ValueError('exhausted slot must have used all attempts')
    if manifest['cases'] != accepted:
        raise ValueError('accepted manifest and attempt journal disagree')
    return manifest


def audit_accepted(root, manifest):
    rows = []
    for case in manifest['cases']:
        scene, graph, frames = read_layout(root, manifest['config'], case)
        routes = routes_for(case, scene, graph, frames, manifest['config'])
        saved = read_json(checked_path(root, case['path'])/'roundtrip_routes.json')
        if saved != dict(case=case, routes=routes):
            raise ValueError('accepted route index changed')
        rows.append(dict(scene=case['scene_id'], group=case['group'], split=case['split'],
                         raw_rgb=len(frames), routes=len(routes),
                         outward_views=[r['outward_frames'] for r in routes],
                         route_steps=[len(r['states']) for r in routes]))
    return dict(schema='multiscene_roundtrip_validation_v1', accepted_scenes=len(rows),
                requested_scenes=len(manifest['slots']),
                exhausted_scenes=sum(s['status'] == 'exhausted' for s in manifest['slots']),
                pending_scenes=sum(s['status'] == 'pending' for s in manifest['slots']),
                by_split=dict(Counter(c['split'] for c in manifest['cases'])),
                accepted_backgrounds=len({c['group'] for c in manifest['cases']}),
                raw_rgb=sum(r['raw_rgb'] for r in rows), rows=rows,
                route_selection='GT masks only; no predictions or model scores')


def validate_collection(root):
    root = Path(root).resolve()
    manifest = check_manifest(root)
    if manifest['status'] != 'complete' or not manifest['cases']:
        raise ValueError('collection has no published, complete accepted subset')
    report = audit_accepted(root, manifest)
    if report['pending_scenes']:
        raise ValueError('collection still has pending slots')
    expected = 'partial' if report['exhausted_scenes'] else 'complete'
    if manifest['collection_status'] != expected:
        raise ValueError('collection completion flag disagrees with slots')
    return report


def collect(config, output, carla, world, versions, *, resume=False):
    """Full attempts are kept under attempts/; never replace an existing photograph."""
    validate_config(config)
    root = Path(output).resolve()
    if len(versions) != 2 or versions[0] != versions[1]:
        raise RuntimeError('exact CARLA client/server version agreement required')
    preflight(world, config['map'])
    spawn_poses = [list(pose_values(t)) for t in world.get_map().get_spawn_points()]
    backgrounds = choose_backgrounds(config, spawn_poses)
    # Detect missing assets before recording hundreds of photographs.
    for item in config['capture']['candidates']:
        world.get_blueprint_library().find(item['blueprint'])
    if resume:
        if not root.is_dir():
            raise ValueError('resume requires an existing collection directory')
    else:
        if root.exists():
            raise FileExistsError('use a new output directory, or --resume on the same collection')
        create_manifest(root, config, backgrounds, versions, spawn_poses)
    with capture_lock(root):
        manifest = check_manifest(root, config)
        if manifest['carla_versions'] != versions or manifest['spawn_poses'] != spawn_poses:
            raise ValueError('resume CARLA version/map spawn list changed')
        audit_accepted(root, manifest)  # Refuse damaged old successes before new work.
        manifest.update(status='incomplete', collection_status='capturing')
        atomic_json(root/'dataset_manifest.json', manifest)
        by_id = {b['id']: b for b in backgrounds}
        for slot in manifest['slots']:
            if slot['status'] != 'pending':
                continue
            # Interrupted attempt paths are preserved; use a new deterministic attempt next.
            for old in slot['attempts']:
                if old['status'] == 'running':
                    old.update(status='interrupted', error='previous process interrupted')
            while len(slot['attempts']) < config['sampling']['max_attempts']:
                preflight(world, config['map'])
                attempt = len(slot['attempts'])+1
                case = case_for(config, by_id[slot['background_id']], slot, attempt)
                entry = dict(case=case, status='running', started_utc=now())
                slot['attempts'].append(entry)
                atomic_json(root/'dataset_manifest.json', manifest)
                print(f'[scene] {case["scene_id"]} attempt {attempt}/{config["sampling"]["max_attempts"]}', flush=True)
                try:
                    capture_layout(carla, world, versions, config, root, case)
                    scene, graph, frames = read_layout(root, config, case)
                    routes = routes_for(case, scene, graph, frames, config)
                    atomic_json(root/case['path']/'roundtrip_routes.json', dict(case=case, routes=routes))
                    entry.update(status='accepted', finished_utc=now(),
                                 outward_views=[r['outward_frames'] for r in routes],
                                 route_steps=[len(r['states']) for r in routes])
                    slot['status'] = 'accepted'
                    manifest['cases'].append(case)
                except BaseException as error:
                    entry.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'rejected',
                                 error=f'{type(error).__name__}: {error}', finished_utc=now())
                    atomic_json(root/'dataset_manifest.json', manifest)
                    scene_path = root/case['path']/'scene.json'
                    if scene_path.exists() and read_json(scene_path).get('cleanup_errors'):
                        raise RuntimeError('CARLA cleanup failed; stopped collection; inspect scene.json') from error
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    preflight(world, config['map'])  # No retry in an unclean simulation.
                    print(f'[rejected] {case["scene_id"]}: {entry["error"]}', flush=True)
                atomic_json(root/'dataset_manifest.json', manifest)
                if slot['status'] == 'accepted':
                    break
            if slot['status'] != 'accepted':
                slot['status'] = 'exhausted'
                atomic_json(root/'dataset_manifest.json', manifest)
        report = audit_accepted(root, manifest)
        manifest.update(status='complete' if manifest['cases'] else 'incomplete',
                        collection_status='partial' if report['exhausted_scenes'] else 'complete',
                        finished_utc=now(), counts={k: v for k, v in report.items() if k != 'rows'})
        atomic_json(root/'dataset_manifest.json', manifest)
        atomic_json(root/'validation.json', report)
        return manifest, report
