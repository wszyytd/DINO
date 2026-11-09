import json
from types import SimpleNamespace

import numpy as np
import pytest

from dinoexp.dense_cache import save_json
from dinoexp.search_capture import (execute_action, preflight, register_resume, save_observation,
                                    save_reference)
from dinoexp.search_episodes import (action_trace, instance_labels, iter_v1_inputs,
                                    recognition_quality, validate_episode, validate_splits)


def test_relative_motion_and_yaw_share_fixed_height_and_pitch():
    pose = [10, 20, 8.6, -45, 90, 0]
    trace = action_trace(pose, 'forward')
    assert len(trace) == 4
    assert trace[0] == pytest.approx([10, 20.125, 8.6, -45, 90, 0])
    assert trace[-1] == pytest.approx([10, 20.5, 8.6, -45, 90, 0])
    assert action_trace(trace[-1], 'yaw_right')[-1] == pytest.approx(
        [10, 20.5, 8.6, -45, 105, 0]
    )


def test_yaw_substeps_acknowledge_both_cameras_before_tick(monkeypatch):
    start = [28.70821, 139.00932, 8.6000004, -45, -89.6795349, 0]
    sensors = [SimpleNamespace(id=40), SimpleNamespace(id=41)]
    carla = SimpleNamespace(command=SimpleNamespace(ApplyTransform=lambda actor, pose: (actor, pose)))
    monkeypatch.setattr('dinoexp.search_capture.as_transform', lambda _, pose: pose)
    monkeypatch.setattr('dinoexp.search_capture.pose_values', tuple)
    poses = {camera.id: start for camera in sensors}
    events, trace = [], []
    tick_number = 126136

    def apply_batch_sync(commands, do_tick):
        assert do_tick is False
        assert [camera for camera, _ in commands] == [40, 41]
        poses.update(commands)
        events.append('acknowledged')
        return [SimpleNamespace(error='') for _ in commands]

    def tick(_):
        nonlocal tick_number
        assert events[-1] == 'acknowledged'
        events.append('tick')
        tick_number += 1
        return tick_number

    def snapshot():
        events.append('snapshot')
        return SimpleNamespace(frame=tick_number,
                               find=lambda actor: SimpleNamespace(get_transform=lambda: poses[actor]))

    end = execute_action(carla, SimpleNamespace(apply_batch_sync=apply_batch_sync),
                         SimpleNamespace(tick=tick, get_snapshot=snapshot), sensors,
                         {'timeout': 15}, {'pose': start}, 'yaw_left', trace)

    assert events == ['acknowledged', 'tick', 'snapshot'] * 4
    assert trace[0]['actual_pose'][4] == pytest.approx(-93.4295349)
    assert end[4] == pytest.approx(-104.6795349)
    assert [row['snapshot_frame'] for row in trace] == list(range(126137, 126141))
    assert all(row['actual_pose'] == row['actual_instance_pose'] for row in trace)


def test_opaque_ids_and_ignore_mask_are_separate():
    raw = np.array([[[14, 1, 2], [14, 2, 2], [14, 3, 2], [7, 4, 2]]], np.uint8)
    mappings = {'s:A': {'ids': [513], 'tags': [14]}, 's:B': {'ids': [514], 'tags': [14]}}
    masks, ignore = instance_labels(raw, mappings, [14])
    assert masks['s:A'].tolist() == [[True, False, False, False]]
    assert masks['s:B'].tolist() == [[False, True, False, False]]
    assert ignore.tolist() == [[False, False, True, False]]


def test_edge_fragment_is_visible_but_weak():
    mask = np.zeros((768, 1024), bool)
    mask[:7, 10:20] = True
    row = recognition_quality(mask)
    assert row['visibility_present'] is True
    assert row['visible_pixels'] == 70
    assert row['touches_boundary'] is True
    assert row['recognition_quality'] == 'weak'
    assert recognition_quality(np.zeros_like(mask))['visibility_present'] is False


def test_preflight_refuses_existing_synchronous_session():
    world = SimpleNamespace(get_map=lambda: SimpleNamespace(name='Carla/Maps/Town10HD_Opt'),
                            get_settings=lambda: SimpleNamespace(synchronous_mode=True))
    with pytest.raises(RuntimeError, match='synchronous tick'):
        preflight(world, 'Town10HD_Opt')


def test_explicit_code_update_preserves_completed_capture_provenance():
    config, cases = {'seed': 7}, [{'episode_id': 'old'}, {'episode_id': 'pending'}]
    original = dict(config=config, cases=cases, code={'git_head': 'old'}, completed_episodes=['old'])
    new = {'git_head': 'new'}
    with pytest.raises(ValueError, match='resume-after-code-update'):
        register_resume(original, cases, config, new, False)
    register_resume(original, cases, config, new, True)
    assert original['code'] == new
    assert original['episode_code'] == {'old': {'git_head': 'old'}}
    assert original['code_history'][-1]['previous'] == {'git_head': 'old'}
    with pytest.raises(ValueError, match='plan'):
        register_resume(original, [], config, new, True)


def test_derivatives_and_layouts_cannot_cross_splits():
    source = dict(group='s1', split='train', episode_id='original')
    with pytest.raises(ValueError, match='lineage'):
        validate_splits([source, dict(group='s2', split='val', episode_id='slice',
                                     source_episode_id='original')])
    with pytest.raises(ValueError, match='group'):
        validate_splits([source, dict(group='s1', split='val', episode_id='other_ref')])


def test_episode_roundtrip_verifies_raw_observations_and_actions(tmp_path):
    # Synthetic fixture only. Nothing from this test is exported as collected data.
    case = dict(scene_id='s', group='layout', split='train', episode_id='ep', target='A',
                path='scenes/s/episodes/ep', actions=['hold_capture'] * 15)
    directory = tmp_path / case['path']
    for folder in ('rgb', 'labels', 'audit'):
        (directory / folder).mkdir(parents=True)
    save_json(directory.parent.parent / 'scene.json', dict(group='layout'))
    mappings = {'s:A': {'ids': [513], 'tags': [14]}, 's:B': {'ids': [514], 'tags': [14]}}
    shape = (256, 360, 3)
    pixels = np.full(shape, 145, np.uint8)
    raw = np.full(shape, [7, 0, 0], np.uint8)
    raw[25:175, 30:290] = [14, 1, 2]
    pose = [4, 5, 8.6, -45, 0, 0]

    def pair(frame):
        metadata = dict(frame=frame, timestamp=frame * .05, width=360, height=256,
                        pose=pose, intrinsics={'model': 'test_fixture'})
        return dict(rgb=dict(metadata), instance=dict(metadata))

    masks, _ = instance_labels(raw, mappings, [14])
    reference = save_reference(tmp_path, directory, pixels, raw, pair(1), masks['s:A'], 's:A')
    previous, start = None, None
    for step in range(16):
        frame = 20 + step * 10
        query_raw = raw.copy()
        query_raw[25:175, 30:290] = [7, 0, 0]
        query_raw[30:70, 30:70] = [14, 2, 2]
        if step >= 8:
            query_raw[100:170, 100:190] = [14, 1, 2]
        masks, ignored = instance_labels(query_raw, mappings, [14])
        metadata = pair(frame)
        action = None
        if previous:
            action = dict(name='hold_capture', id=6, status='executed', collision_check='unknown',
                          start_pose=pose, end_pose=pose, start_frame=previous['frame'], end_frame=frame,
                          command_duration_s=.2, observed_elapsed_s=metadata['rgb']['timestamp'] - previous['timestamp'],
                          interpolation=[dict(frame=previous['frame'] + i, commanded_pose=pose,
                                              actual_pose=pose) for i in range(1, 5)])
        previous = save_observation(tmp_path, directory, case, step, pixels, query_raw, metadata,
                                    masks, ignored, action, reference, 1.0)
        if start is None:
            start = previous
    save_json(directory / 'episode.json', dict(status='complete', case=case, mode='kinematic_camera_sim',
              terminator='scripted_horizon', target_uid='s:A', reference=reference,
              instance_mapping=mappings, instance_id_method='remove_restore_probe', vehicle_tags=[14],
              observation_count=16, start=start, end=previous))
    stats = validate_episode(tmp_path, case)
    assert stats['visibility'] == {'distractor_only': 8, 'both': 8}
    assert stats['first_target_step'] == 8
    row = next(iter_v1_inputs(tmp_path, case['path']))
    assert set(row) == {'rgb', 'reference', 'time_s', 'previous_action_id', 'legal_action_mask'}
    assert np.array_equal(row['rgb'], pixels)

    path = directory / 'audit' / 'observations.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]['action_from_previous']['end_pose'] = [7, *pose[1:]]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='inconsistent action transition'):
        validate_episode(tmp_path, case)
