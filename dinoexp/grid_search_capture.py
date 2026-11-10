"""Capture raw, independently photographed states for the offline search graph."""

from datetime import datetime, timezone
import math

import numpy as np

from .core import camera_pose, pose_matches, pose_values
from .dense_cache import save_json
from .dense_capture import managed_capture
from .scene import validate_scene
from .search_capture import (calibrate, get_pair, intrinsics, place_camera, preflight,
                             save_mask, save_png, save_reference, spawn_scene, spawn_sensors,
                             vehicle_tags)
from .search_episodes import QUALITY_SPEC, append_row, digest, instance_labels, pixel_digest, recognition_quality
from .search_grid import action_spec, build_graph


def grid_plan(config, stage):
    cfg = config['capture']
    identities = [item['identity'] for item in cfg['candidates']]
    validate_scene(dict(candidates=cfg['candidates'], target_identity=identities[0]))
    if not math.isfinite(cfg['height_m']) or cfg['height_m'] <= 0:
        raise ValueError('positive finite fixed camera height required')
    if not math.isfinite(config['minimum_vehicle_gap_m']) or config['minimum_vehicle_gap_m'] <= 0:
        raise ValueError('positive minimum vehicle gap required')
    layouts = config['layouts'][:1] if stage == 'smoke' else config['layouts']
    if not layouts:
        raise ValueError('at least one layout is required')
    cases, groups, anchors = [], set(), {}
    for index, layout in enumerate(layouts):
        split = layout['split']
        anchor = layout['anchor_spawn_index']
        placements = layout['placements']
        if (split not in ('train', 'val') or type(anchor) is not int or anchor < 0
            or len(placements) != len(identities)
            or any(len(p) != 3 or not all(map(math.isfinite, p)) for p in placements)
            or len(layout['grid_center_offset']) != 2 or not all(map(math.isfinite, layout['grid_center_offset']))):
            raise ValueError('invalid layout placements, anchor, or split')
        if anchors.setdefault(anchor, split) != split:
            raise ValueError('all captures at one background anchor must share a split')
        group = 'layout-' + digest([config['map'], anchor, sorted(placements)])[:16]
        if group in groups:
            raise ValueError('duplicate physical vehicle layout')
        groups.add(group)
        scene = f'scene_{index:02d}'
        cases.append(dict(scene_id=scene, group=group, split=split, layout=layout,
                          path=f'scenes/{scene}', seed=config['seed'] + index))
        graph_for(config, cases[-1], [0, 0, 0, 0, 0, 0])
    if stage == 'pilot' and set(anchors.values()) != {'train', 'val'}:
        raise ValueError('pilot requires independent train and validation background anchors')
    return cases


def local_xy(anchor, forward, right):
    angle = math.radians(anchor[4])
    return [anchor[0] + forward*math.cos(angle) - right*math.sin(angle),
            anchor[1] + forward*math.sin(angle) + right*math.cos(angle)]


def graph_for(config, case, anchor):
    grid = config['grid']
    cfg = config['capture']
    xy = local_xy(anchor, *case['layout']['grid_center_offset'])
    return build_graph(grid['size'], grid['spacing_m'], [*xy, anchor[2] + cfg['height_m']],
                       cfg['pitch_deg'], move_speed=grid['move_speed'], turn_speed=grid['turn_speed'],
                       hold_s=grid['hold_s'], excluded=case['layout'].get('excluded_points', ()))


def save_frame(root, directory, case, state_id, state, pixels, raw, metadata, masks, ignore):
    rgb = save_png(root, directory / 'rgb' / f'{state_id}.png', pixels)
    instance = save_png(root, directory / 'audit' / f'{state_id}_instance.png', raw)
    labels = {
        uid: dict(quality=recognition_quality(mask),
                  mask=save_mask(root, directory / 'labels' / f'{state_id}_{uid.split(":")[-1]}.png', mask))
        for uid, mask in masks.items()
    }
    metadata['rgb'].update(file_sha256=rgb['sha256'], sensor_sha256=pixel_digest(pixels))
    metadata['instance'].update(path=instance['path'], file_sha256=instance['sha256'])
    append_row(directory / 'frames.jsonl', dict(
        state_id=state_id, point_id=state['point_id'], direction_id=state['direction_id'],
        scene_id=case['scene_id'], layout_id=case['scene_id'], group=case['group'],
        sensor_timestamp=metadata['rgb']['timestamp'], sensor_frame=metadata['rgb']['frame'],
        image_path=rgb['path'], objects=labels, annotation_validity='valid',
        ignore_mask=save_mask(root, directory / 'labels' / f'{state_id}_ignore.png', ignore),
    ))
    append_row(directory / 'audit' / 'observations.jsonl', dict(state_id=state_id, **metadata))


def capture_layout(carla, world, versions, config, root, case):
    cfg = dict(config['capture'], fixed_delta=.05, warmup_ticks=4)
    state = preflight(world, config['map'])
    points = world.get_map().get_spawn_points()
    anchor_index = case['layout']['anchor_spawn_index']
    if anchor_index >= len(points):
        raise ValueError('layout anchor not available on current map')
    anchor = list(pose_values(points[anchor_index]))
    graph = graph_for(config, case, anchor)
    actor_poses = [[*local_xy(anchor, f, r), anchor[2]+.3, anchor[3], anchor[4]+yaw, anchor[5]]
                   for f, r, yaw in case['layout']['placements']]
    directory = root / case['path']
    directory.mkdir(parents=True)
    for folder in ('rgb', 'labels', 'audit', 'references'):
        (directory / folder).mkdir()
    failures = directory / 'audit' / 'failures.jsonl'
    scene = dict(case=case, status='incomplete', cleanup_errors=[], carla_versions=versions,
                 map=state['map'], anchor=anchor, intrinsics=intrinsics(cfg),
                 started_utc=datetime.now(timezone.utc).isoformat(), capture_count=0)
    save_json(directory / 'scene.json', scene)
    save_json(directory / 'graph.json', graph)
    owned = []
    context = 'setup'
    try:
        with managed_capture(world, owned, scene):
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
            geometry = spawn_scene(carla, world, owned, cfg, case, anchor, actor_poses=actor_poses)
            objects = {}
            for uid, (actor, tf, center, candidate) in geometry.items():
                corners = [[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(tf)]
                radius = max(math.dist(v[:2], center[:2]) for v in corners)
                objects[uid] = dict(**candidate, pose=list(pose_values(tf)), center=list(map(float, center)),
                                    bbox_radius_xy_m=radius, actual_blueprint=actor.type_id,
                                    actor_attributes=dict(actor.attributes))
            a, b = objects.values()
            # Circumscribed projected boxes yield a conservative lower bound on vehicle clearance.
            clearance = math.dist(a['center'][:2], b['center'][:2]) - a['bbox_radius_xy_m'] - b['bbox_radius_xy_m']
            scene.update(objects=objects, bbox_clearance_lower_bound_m=clearance,
                         weather={key: getattr(weather, key) for key in cfg['weather']})
            if clearance < config['minimum_vehicle_gap_m']:
                raise ValueError(f'vehicle footprint clearance lower bound {clearance:.2f} m is too small')
            sensors, queues = spawn_sensors(carla, world, owned, cfg)
            scene['sensor_attributes'] = [dict(sensor.attributes) for sensor in sensors]

            def photograph(pose, purpose):
                return get_pair(world, queues, pose, cfg, failures, purpose)

            context = 'instance_calibration'
            center = np.mean([item[2] for item in geometry.values()], axis=0)
            # The higher nadir probe is isolated from the fixed-height observation graph.
            mappings = calibrate(world, carla, sensors, photograph, center, anchor, geometry,
                                 dict(cfg, height_m=14))
            tags = vehicle_tags(carla, mappings)
            scene.update(instance_mapping=mappings, instance_id_method='remove_restore_probe',
                         vehicle_tags=tags, ignore_policy='unmapped vehicle or unlabeled pixels' if tags
                         else 'conservative: all pixels outside mapped vehicles')
            refs = {}
            for uid, (_, tf, center, _) in geometry.items():
                context = 'reference_' + uid.split(':')[-1]
                ref = cfg['reference']
                pose = list(map(float, camera_pose(center, tf.rotation.yaw, ref['azimuth'],
                                                   ref['elevation'], ref['distance'])))
                place_camera(carla, sensors, pose)
                pixels, raw, metadata = photograph(pose, context)
                masks, _ = instance_labels(raw, mappings, tags)
                ref_parent = directory / 'references' / uid.split(':')[-1]
                ref_parent.mkdir()
                refs[uid] = save_reference(root, ref_parent, pixels, raw, metadata, masks[uid], uid)
            scene['references'] = refs
            save_json(directory / 'scene.json', scene)
            for state_id, observation in graph['states'].items():
                context = state_id
                place_camera(carla, sensors, observation['pose'])
                pixels, raw, metadata = photograph(observation['pose'], context)
                if any(not pose_matches(pose_values(actor.get_transform()), pose_values(tf))
                       for actor, tf, _, _ in geometry.values()):
                    raise RuntimeError('owned vehicle moved after calibration')
                masks, ignore = instance_labels(raw, mappings, tags)
                save_frame(root, directory, case, state_id, observation, pixels, raw, metadata, masks, ignore)
                scene['capture_count'] += 1
                if scene['capture_count'] % 8 == 0:
                    point = observation['point_id']
                    graph['points'][point]['capture_status'] = 'captured'
                    print(f'{case["scene_id"]} {point} {scene["capture_count"]}/{len(graph["states"])}',
                          flush=True)
        if scene['cleanup_errors']:
            raise RuntimeError(f'CARLA cleanup failed: {scene["cleanup_errors"]}')
        scene['status'] = 'captured'
    except BaseException as error:
        scene.update(status='incomplete', failure=str(error))
        append_row(failures, dict(context=context, status='failed', error=str(error)))
        raise
    finally:
        save_json(directory / 'graph.json', graph)
        scene['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(directory / 'scene.json', scene)


def init_collection(root, config, cases, stage, code):
    root.mkdir(parents=True)
    save_json(root / 'action_spec.json', action_spec(config['grid']))
    save_json(root / 'quality_spec.json', QUALITY_SPEC)
    save_json(root / 'splits.json', {case['group']: case['split'] for case in cases})
    manifest = dict(status='incomplete', mode='discrete_view_graph', stage=stage, code=code,
                    config=config, cases=cases)
    save_json(root / 'dataset_manifest.json', manifest)
    return manifest
