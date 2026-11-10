"""CARLA capture of ordered, interpolated camera actions in a private empty session."""

from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import queue

import numpy as np
from PIL import Image

from capture import as_transform, find_blueprint
from .core import camera_pose, camera_pose_matches, checked_path, image_quality, pose_matches, pose_values
from .dense_cache import save_json
from .dense_capture import (_receive, _rgb, decode_instance_ids, discover_instance_ids,
                            managed_capture, next_pair, occlusion_probe_pose)
from .local_pairs import sha256
from .scene import candidate_poses
from .search_episodes import (ACTIONS, ACTION_SPEC, QUALITY_SPEC, action_trace, append_row,
                              check_pair, instance_labels, json_rows, pixel_digest,
                              recognition_quality, validate_dataset, validate_episode, validate_splits)


def preflight(world, expected_map):
    map_name = world.get_map().name
    if map_name.rsplit('/', 1)[-1] != expected_map:
        raise RuntimeError(f'expected map {expected_map}, found {map_name}; use a dedicated session')
    if world.get_settings().synchronous_mode:
        raise RuntimeError('CARLA already has synchronous tick enabled; use a dedicated idle session')
    actors = Counter(a.type_id.split('.')[0] for a in world.get_actors())
    active = {k: actors[k] for k in ('vehicle', 'sensor', 'walker', 'controller') if actors[k]}
    if active:
        raise RuntimeError(f'CARLA session has existing actors: {active}; use an idle session')
    return {'map': map_name, 'synchronous_mode': False, 'existing_active_actors': active}


def intrinsics(cfg):
    width, height = cfg['width'], cfg['height']
    focal = width / (2 * math.tan(math.radians(cfg['fov']) / 2))
    return dict(model='nominal_pinhole', width=width, height=height, horizontal_fov_deg=cfg['fov'],
                K=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                lens_attributes=cfg['lens_attributes'])


def frame_metadata(frame, cfg):
    if abs(frame.fov - cfg['fov']) > .01:
        raise ValueError('unexpected sensor field of view')
    if (frame.width, frame.height) != (cfg['width'], cfg['height']):
        raise ValueError('unexpected sensor image dimensions')
    return dict(frame=frame.frame, timestamp=frame.timestamp, width=frame.width, height=frame.height,
                pose=list(pose_values(frame.transform)), intrinsics=intrinsics(cfg))


def save_png(root, file, array):
    Image.fromarray(array).save(file)
    return dict(path=file.relative_to(root).as_posix(), sha256=sha256(file))


def save_mask(root, file, mask):
    return save_png(root, file, mask.astype(np.uint8) * 255) if mask.any() else None


def place_camera(carla, sensors, pose):
    transform = as_transform(carla, pose)
    for sensor in sensors:
        sensor.set_transform(transform)


def get_pair(world, queues, pose, cfg, failures, context):
    for attempt in range(1, 4):
        try:
            rgb, instance = next_pair(world, queues, pose, cfg)
            metadata = dict(rgb=frame_metadata(rgb, cfg), instance=frame_metadata(instance, cfg))
            check_pair(metadata)
            pixels, raw = _rgb(rgb), _rgb(instance)
            if image_quality(Image.fromarray(pixels))['near_black'] or not raw.any():
                raise ValueError('near-black RGB or empty raw segmentation buffer')
            return pixels, raw, metadata
        except (TimeoutError, ValueError) as error:
            append_row(failures, dict(context=context, attempt=attempt, error=str(error),
                                      annotation_validity='invalid', retry=attempt < 3))
            if attempt == 3:
                raise


def spawn_scene(carla, world, owned, cfg, case, anchor, *, actor_poses=None):
    layout = case['layout']
    library = world.get_blueprint_library()
    candidates = cfg['candidates']
    if actor_poses is None:
        actor_poses = candidate_poses(anchor, layout['separation_m'], layout['car_yaw_offsets'])
    actors = []
    for candidate, pose in zip(candidates, actor_poses, strict=True):
        bp = find_blueprint(library, candidate['blueprint'])
        bp.set_attribute('color', candidate['color'])
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'dino_search_owned')
        actor = world.try_spawn_actor(bp, as_transform(carla, pose))
        if actor is None:
            raise RuntimeError(f'could not spawn owned vehicle {candidate["identity"]}')
        owned.append(actor)
        actors.append(actor)
        actor.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
    for _ in range(cfg['settle_ticks']):
        previous = [list(pose_values(actor.get_transform())) for actor in actors]
        world.tick(cfg['timeout'])
    geometry = {}
    for candidate, actor, prev in zip(candidates, actors, previous, strict=True):
        tf = actor.get_transform()
        velocity = actor.get_velocity()
        if not pose_matches(pose_values(tf), prev) or math.hypot(velocity.x, velocity.y, velocity.z) > .1:
            raise RuntimeError('owned vehicle did not settle')
        actor.set_simulate_physics(False)
        center = np.mean([[p.x, p.y, p.z] for p in actor.bounding_box.get_world_vertices(tf)], axis=0)
        uid = f'{case["scene_id"]}:{candidate["identity"]}'
        geometry[uid] = (actor, tf, center, candidate)
    return geometry


def spawn_sensors(carla, world, owned, cfg):
    library = world.get_blueprint_library()
    sensors, queues = [], []
    for kind in ('rgb', 'instance_segmentation'):
        bp = find_blueprint(library, 'sensor.camera.' + kind)
        attributes = dict(image_size_x=cfg['width'], image_size_y=cfg['height'], fov=cfg['fov'],
                          sensor_tick=0, **cfg['lens_attributes'])
        if kind == 'rgb':
            attributes.update(cfg['camera_attributes'])
        for name, value in attributes.items():
            if not bp.has_attribute(name):
                raise ValueError(f'unsupported camera attribute {kind}.{name}')
            bp.set_attribute(name, str(value))
        sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=100)))
        owned.append(sensor)
        sensor_queue = queue.Queue(maxsize=24)
        sensor.listen(_receive(sensor_queue))
        sensors.append(sensor)
        queues.append(sensor_queue)
    return sensors, queues


def calibrate(world, carla, sensors, photograph, center, anchor, geometry, cfg):
    # Probe images never enter the observations or reference files.
    pose = [float(center[0]), float(center[1]), anchor[2] + cfg['height_m'], -90, anchor[4], 0]
    place_camera(carla, sensors, pose)
    _, before, base = photograph(pose, 'calibration_base')
    before_ids = decode_instance_ids(before)
    mappings, used = {}, set()
    for uid, (actor, tf, _, _) in geometry.items():
        hidden = occlusion_probe_pose(pose_values(tf), pose)
        try:
            actor.set_transform(as_transform(carla, hidden))
            _, removed, removal = photograph(pose, 'calibration_hidden')
        finally:
            actor.set_transform(tf)
        _, restored, restore = photograph(pose, 'calibration_restored')
        ids = discover_instance_ids(before_ids, decode_instance_ids(removed), decode_instance_ids(restored))
        selected = np.isin(before_ids, ids)
        if used.intersection(ids) or int(selected.sum()) < 64:
            raise ValueError('ambiguous or undersized instance mapping at calibration pose')
        used.update(ids)
        mappings[uid] = dict(ids=ids, tags=sorted(map(int, np.unique(before[:, :, 0][selected]))),
                             python_actor_id=actor.id, base_frame=base['rgb']['frame'],
                             hidden_frame=removal['rgb']['frame'], restored_frame=restore['rgb']['frame'])
    return mappings


def vehicle_tags(carla, mappings):
    enum = getattr(carla, 'CityObjectLabel', None)
    names = ('Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle', 'Rider', 'Vehicles')
    tags = {int(getattr(enum, name)) for name in names if hasattr(enum, name)}
    # A release that does not expose the complete vehicle enum uses conservative ignore.
    if not hasattr(enum, 'Car') or not all(hasattr(enum, name) for name in names[:6]):
        return []
    return sorted(tags.union(*(record['tags'] for record in mappings.values())))


def save_reference(root, directory, pixels, raw, metadata, mask, target):
    ref_dir = directory / 'reference'
    ref_dir.mkdir()
    quality = recognition_quality(mask)
    if not quality['bbox_xyxy']:
        raise ValueError('reference target is absent')
    x0, y0, x1, y1 = quality['bbox_xyxy']
    width, height = x1 - x0, y1 - y0
    dx, dy = max(2, round(width * .04)), max(2, round(height * .04))
    crop = [max(0, x0 - dx), max(0, y0 - dy),
            min(pixels.shape[1], x1 + dx), min(pixels.shape[0], y1 + dy)]
    a, b, c, d = crop
    records = {
        'full': save_png(root, ref_dir / 'reference_full.png', pixels),
        'object': save_png(root, ref_dir / 'reference_object.png', pixels[b:d, a:c]),
        'mask': save_mask(root, ref_dir / 'reference_mask.png', mask),
        'raw': save_png(root, ref_dir / 'reference_instance_raw.png', raw),
    }
    result = dict(target_uid=target, crop_xyxy=crop, quality=quality, audit=metadata,
                  rgb_sensor_sha256=pixel_digest(pixels),
                  hashes={record['path']: record['sha256'] for record in records.values()},
                  **{name: record['path'] for name, record in records.items()})
    save_json(ref_dir / 'reference.json', result)
    if (quality['recognition_quality'] != 'usable' or max(width, height) < QUALITY_SPEC['reference_min_long_side']
        or min(width, height) < QUALITY_SPEC['reference_min_short_side']):
        raise ValueError(f'reference below pilot quality threshold: {width}x{height}, {quality}')
    return result


def execute_action(carla, client, world, sensors, cfg, previous, name, trace):
    start = previous['pose']
    for desired in action_trace(start, name):
        entry = dict(commanded_pose=desired)
        trace.append(entry)
        # CARLA 0.10.0 Actor.set_transform uses an async RPC. Both transforms
        # must finish on the server before the sole tick for this substep.
        transform = as_transform(carla, desired)
        commands = [carla.command.ApplyTransform(sensor.id, transform) for sensor in sensors]
        responses = client.apply_batch_sync(commands, False)
        errors = [response.error for response in responses]
        if len(responses) != len(commands) or any(errors):
            entry['command_errors'] = errors
            raise RuntimeError(f'kinematic action {name}: transform batch failed: {errors}')
        frame = world.tick(cfg['timeout'])
        snapshot = world.get_snapshot()
        entry.update(frame=frame, snapshot_frame=snapshot.frame)
        if snapshot.frame != frame:
            raise RuntimeError(f'kinematic action {name}: tick {frame}, snapshot {snapshot.frame}')
        states = [snapshot.find(sensor.id) for sensor in sensors]
        if any(state is None for state in states):
            raise RuntimeError(f'kinematic action {name}: camera missing from snapshot {frame}')
        actual, actual_instance = [list(pose_values(state.get_transform())) for state in states]
        entry.update(actual_pose=actual, actual_instance_pose=actual_instance)
        if (not camera_pose_matches(actual, desired) or
                not camera_pose_matches(actual_instance, actual, .005, .05)):
            raise RuntimeError(f'kinematic action {name} at frame {frame}: commanded={desired}, '
                               f'RGB={actual}, instance={actual_instance}')
    return trace[-1]['commanded_pose']


def save_observation(root, directory, case, step, pixels, raw, metadata, masks, ignored, action,
                     reference, initial_time):
    name = f'{step:06d}'
    rgb = save_png(root, directory / 'rgb' / (name + '.png'), pixels)
    instance = save_png(root, directory / 'audit' / (name + '_instance.png'), raw)
    labels = {uid: dict(mask=save_mask(root, directory / 'labels' / f'{name}_{uid.split(":")[-1]}.png', mask),
                        quality=recognition_quality(mask)) for uid, mask in masks.items()}
    time = metadata['rgb']['timestamp']
    target = reference['target_uid']
    frame_id = case['episode_id'] + ':' + name
    append_row(directory / 'frames.jsonl', dict(
        frame_id=frame_id, sensor_timestamp=time, sensor_frame=metadata['rgb']['frame'],
        episode_id=case['episode_id'], step_index=step, image_path=rgb['path'],
        action_from_previous=({k: action[k] for k in ('id', 'name', 'status')} if action else None),
        objects=labels, target_uid=target, annotation_validity='valid',
        visibility_present=labels[target]['quality']['visibility_present'],
        ignore_mask=save_mask(root, directory / 'labels' / f'{name}_ignore.png', ignored),
    ))
    metadata['rgb'].update(file_sha256=rgb['sha256'], sensor_sha256=pixel_digest(pixels))
    metadata['instance'].update(path=instance['path'], file_sha256=instance['sha256'])
    append_row(directory / 'audit' / 'observations.jsonl',
               dict(frame_id=frame_id, action_from_previous=action, **metadata))
    append_row(directory / 'inputs.jsonl', dict(
        rgb=rgb['path'], reference=reference['object'], time_s=time - initial_time,
        previous_action_id=action['id'] if action else None, legal_action_mask=None,
    ))
    return metadata['rgb']


def world_start(anchor, center, case, cfg):
    forward, right, yaw = case['start']
    angle = math.radians(anchor[4])
    return [float(center[0]) + forward * math.cos(angle) - right * math.sin(angle),
            float(center[1]) + forward * math.sin(angle) + right * math.cos(angle),
            anchor[2] + cfg['height_m'], cfg['pitch_deg'], (anchor[4] + yaw + 180) % 360 - 180, 0]


def capture_episode(carla, client, world, versions, config, root, case):
    cfg = dict(config['capture'], fixed_delta=ACTION_SPEC['fixed_delta_s'],
               warmup_ticks=ACTION_SPEC['sensor_settle_ticks'])
    state = preflight(world, config['map'])
    points = world.get_map().get_spawn_points()
    anchor_index = case['layout']['anchor_spawn_index']
    if not 0 <= anchor_index < len(points):
        raise ValueError('layout anchor not available on current map')
    anchor = list(pose_values(points[anchor_index]))
    directory = checked_path(root, case['path'])
    directory.mkdir(parents=True)
    for folder in ('rgb', 'labels', 'audit'):
        (directory / folder).mkdir()
    failures = directory / 'audit' / 'failures.jsonl'
    episode = dict(status='incomplete', case=case, mode=ACTION_SPEC['mode'], carla_versions=versions,
                   map=state['map'], cleanup_errors=[], observation_count=0, terminator=None,
                   started_utc=datetime.now(timezone.utc).isoformat(), action_version=ACTION_SPEC['version'],
                   calibration=intrinsics(cfg), capture_config=cfg,
                   instance_id_method='remove_restore_probe')
    save_json(directory / 'episode.json', episode)
    owned = []
    step, name, trace = None, None, []
    try:
        with managed_capture(world, owned, episode):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = cfg['fixed_delta']
            settings.no_rendering_mode = False
            settings.substepping = True
            settings.max_substep_delta_time = .01
            settings.max_substeps = 10
            world.apply_settings(settings)
            weather = world.get_weather()
            for key, value in cfg['weather'].items():
                setattr(weather, key, value)
            world.set_weather(weather)
            geometry = spawn_scene(carla, world, owned, cfg, case, anchor)
            center = np.mean([item[2] for item in geometry.values()], axis=0)
            sensors, queues = spawn_sensors(carla, world, owned, cfg)
            episode['sensor_attributes'] = [dict(s.attributes) for s in sensors]
            episode['objects'] = {uid: dict(pose=list(pose_values(tf)), **candidate)
                                  for uid, (_, tf, _, candidate) in geometry.items()}

            def photograph(pose, context):
                return get_pair(world, queues, pose, cfg, failures, context)

            mappings = calibrate(world, carla, sensors, photograph, center, anchor, geometry, cfg)
            tags = vehicle_tags(carla, mappings)
            episode.update(instance_mapping=mappings, vehicle_tags=tags,
                           unknown_policy='unmapped vehicle or unlabeled pixels' if tags else
                                          'conservative: all pixels outside mapped instances')
            target = case['scene_id'] + ':' + case['target']
            episode['target_uid'] = target
            _, tf, target_center, _ = geometry[target]
            ref = cfg['reference']
            ref_pose = list(map(float, camera_pose(target_center, tf.rotation.yaw, ref['azimuth'],
                                                   ref['elevation'], ref['distance'])))
            place_camera(carla, sensors, ref_pose)
            pixels, raw, metadata = photograph(ref_pose, 'reference')
            masks, _ = instance_labels(raw, mappings, tags)
            reference = save_reference(root, directory, pixels, raw, metadata, masks[target], target)
            episode['reference'] = reference
            print(f'{case["episode_id"]}: reference {reference["quality"]["bbox_xyxy"]}', flush=True)
            start = world_start(anchor, center, case, cfg)
            place_camera(carla, sensors, start)
            previous = None
            initial_time = None
            for step, name in enumerate([None, *case['actions']]):
                trace = []
                pose = (execute_action(carla, client, world, sensors, cfg, previous, name, trace)
                        if name else start)
                pixels, raw, metadata = photograph(pose, f'observation_{step}')
                if any(not pose_matches(pose_values(actor.get_transform()), pose_values(tf))
                       for actor, tf, _, _ in geometry.values()):
                    raise RuntimeError('owned vehicle moved during an episode')
                end = metadata['rgb']
                action = None
                if name:
                    action = dict(name=name, id=ACTIONS[name][0], status='executed',
                                  start_pose=previous['pose'], end_pose=end['pose'],
                                  start_frame=previous['frame'], end_frame=end['frame'],
                                  command_duration_s=.2, observed_elapsed_s=end['timestamp'] - previous['timestamp'],
                                  interpolation=trace, collision_check='unknown',
                                  collision_reason='kinematic camera has no swept-body collision validation')
                if initial_time is None:
                    initial_time = end['timestamp']
                masks, ignored = instance_labels(raw, mappings, tags)
                previous = save_observation(root, directory, case, step, pixels, raw, metadata,
                                            masks, ignored, action, reference, initial_time)
                episode.setdefault('start', previous)
                episode['end'] = previous
                episode['observation_count'] = step + 1
                save_json(directory / 'episode.json', episode)
                print(f'{case["episode_id"]}: {step + 1}/{len(case["actions"]) + 1}', flush=True)
        if episode['cleanup_errors']:
            raise RuntimeError(f'CARLA cleanup failed: {episode["cleanup_errors"]}')
        episode.update(status='captured', terminator='scripted_horizon',
                       finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(directory / 'episode.json', episode)
        episode['statistics'] = validate_episode(root, case, allow_pending=True)
        episode['status'] = 'complete'
    except BaseException as error:
        episode.update(status='incomplete', failure=str(error), terminator='capture_failed')
        append_row(failures, dict(step=step, action=name, executed_interpolation=trace,
                                  status='failed', error=str(error)))
        raise
    finally:
        save_json(directory / 'episode.json', episode)
    return episode


def register_resume(manifest, cases, config, code, allow_code_update):
    if manifest['cases'] != cases or manifest['config'] != config:
        raise ValueError('resume requires the same plan and configuration')
    previous = manifest['code']
    if previous != code and not allow_code_update:
        raise ValueError('capture source changed; use --resume-after-code-update for a compatible repair')
    episode_code = manifest.setdefault('episode_code', {})
    for episode in manifest['completed_episodes']:
        episode_code.setdefault(episode, previous)
    if previous != code:
        manifest.setdefault('code_history', []).append(dict(previous=previous, current=code,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        manifest['code'] = code
    return previous


def capture_collection(carla, client, world, versions, config, root, cases, code, resume=False,
                       allow_code_update=False):
    root = Path(root).resolve()
    groups = validate_splits(cases)
    preflight(world, config['map'])
    manifest_path = root / 'dataset_manifest.json'
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for filename, expected in [('action_spec.json', ACTION_SPEC), ('quality_spec.json', QUALITY_SPEC)]:
            if json.loads((root / filename).read_text(encoding='utf-8')) != expected:
                raise ValueError(f'resume requires an unchanged {filename}')
        previous_code = register_resume(manifest, cases, config, code, allow_code_update)
    else:
        root.mkdir(parents=True, exist_ok=False)
        manifest = dict(format_version=1, status='incomplete', config=config, cases=cases,
                        code=code, completed_episodes=[], episode_code={}, capture_mode=ACTION_SPEC['mode'],
                        support='offline_memory_only', collision_validation='unknown',
                        visual_review='pending', independent_unseen_asset_test=False)
        previous_code = code
    with (root / '.capture.lock').open('x', encoding='utf-8'):
        pass
    try:
        save_json(root / 'action_spec.json', ACTION_SPEC)
        save_json(root / 'quality_spec.json', QUALITY_SPEC)
        save_json(root / 'splits.json', groups)
        save_json(manifest_path, manifest)
        for case in cases:
            directory = checked_path(root, case['path'])
            if directory.exists():
                marker = directory / 'episode.json'
                status = json.loads(marker.read_text(encoding='utf-8')).get('status') if marker.exists() else None
                if status == 'complete' or case['episode_id'] in manifest['completed_episodes']:
                    validate_episode(root, case)
                    manifest['episode_code'].setdefault(case['episode_id'], previous_code)
                    if case['episode_id'] not in manifest['completed_episodes']:
                        manifest['completed_episodes'].append(case['episode_id'])
                    continue
                archive = checked_path(root, 'failed_attempts')
                archive.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                directory.rename(checked_path(archive, f'{case["episode_id"]}_{stamp}'))
            scene = directory.parent.parent
            scene.mkdir(parents=True, exist_ok=True)
            save_json(scene / 'scene.json', dict(scene_id=case['scene_id'], group=case['group'],
                      split=case['split'], layout=case['layout'], ambiguity='not_flagged',
                      layout_source='fixed parked pair with recorded spawn anchor and offsets'))
            capture_episode(carla, client, world, versions, config, root, case)
            manifest['completed_episodes'].append(case['episode_id'])
            manifest['episode_code'][case['episode_id']] = code
            save_json(manifest_path, manifest)
        result = validate_dataset(root)
        if result['errors']:
            raise ValueError(f'dataset validation failed: {result["errors"]}')
        write_preview(root, cases, result)
        manifest['status'] = 'complete'
        manifest.pop('failure', None)
    except BaseException as error:
        manifest.update(status='incomplete', failure=str(error))
        save_json(manifest_path, manifest)
        try:
            write_preview(root, cases, validate_dataset(root))
        except (OSError, ValueError, KeyError) as report_error:
            manifest['report_error'] = str(report_error)
        raise
    finally:
        save_json(manifest_path, manifest)
        (root / '.capture.lock').unlink()
    return result


def write_preview(root, cases, result):
    preview = root / 'preview'
    preview.mkdir(exist_ok=True)
    entries = []
    for case in cases:
        name = case['episode_id']
        if name not in result['episodes']:
            continue
        directory = checked_path(root, case['path'])
        episode = json.loads((directory / 'episode.json').read_text(encoding='utf-8'))
        frames = json_rows(directory / 'frames.jsonl')
        cards = []
        for row in frames:
            step = row['step_index']
            image_path = row['image_path']
            with Image.open(checked_path(root, image_path)) as image:
                overlay = image.convert('RGB')
            for uid, label in row['objects'].items():
                if label['mask']:
                    color = (0, 230, 70) if uid == row['target_uid'] else (40, 130, 255)
                    with Image.open(checked_path(root, label['mask']['path'])) as mask:
                        overlay = Image.composite(Image.new('RGB', overlay.size, color), overlay,
                                                  mask.convert('L').point(lambda v: round(v * .43)))
            overlay.thumbnail((720, 540))
            overlay_path = f'{name}_{step:03d}.jpg'
            overlay.save(preview / overlay_path, quality=90)
            action = row['action_from_previous']
            q = row['objects'][row['target_uid']]['quality']
            label = (f'{step:02d}  {action["name"] if action else "start"}  '
                     f't={row["sensor_timestamp"]:.3f}  visible={q["visibility_present"]}  '
                     f'quality={q["recognition_quality"]}  pixels={q["visible_pixels"]}')
            cards.append(f'<h3>{html.escape(label)}</h3><div class="pair">'
                         f'<a href="../{image_path}"><img loading="lazy" src="../{image_path}"></a>'
                         f'<a href="{overlay_path}"><img loading="lazy" src="{overlay_path}"></a></div>')
        ref = episode['reference']
        entries.append(f'<h2>{name}</h2><p>{html.escape(str(result["episodes"][name]))}</p>'
                       f'<div class="pair"><img src="../{ref["full"]}"><img src="../{ref["object"]}"></div>'
                       + ''.join(cards))
    (preview / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>CARLA search pilot</title>'
        '<style>body{font:16px sans-serif;max-width:1250px;margin:auto;padding:24px}'
        '.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;object-fit:contain}'
        'h3{margin:24px 0 6px}</style><h1>Ordered CARLA observations</h1>'
        '<p>Left: raw RGB. Right: audit overlay. Green is the reference target; blue is the distractor.</p>'
        + ''.join(entries), encoding='utf-8')
    report = ['# Capture validation', '', f'Automated validation: {result["status"]}.',
              'Human preview review: pending.', '',
              'Scope: offline memory data. Swept-body collision checks and flight dynamics are unavailable.',
              '', f'Layouts: {result["layout_count"]}; episodes: {len(result["episodes"])}; '
              f'observations: {result["observation_count"]}.',
              f'Template counts: {result["templates"]}', f'Appearance assets: {result["appearance_assets"]}',
              '', f'Coverage: {result["coverage"]}', '', f'Quality: {result["quality"]}', '']
    report.extend(f'- {name}: {stats}' for name, stats in result['episodes'].items())
    report.extend(f'- Failed: {name}: {error}' for name, error in result['errors'].items())
    (root / 'validation_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
