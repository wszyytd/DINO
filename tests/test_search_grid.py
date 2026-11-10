import pytest

from dinoexp.search_grid import HEADINGS, build_graph, derive_routes, route_rows


def test_grid_actions_use_world_axes_and_distinct_time():
    graph = build_graph(3, 4, [10, 20, 8], -45, move_speed=2, turn_speed=90, hold_s=.25)
    state = graph['states']['r01c01_d07']
    assert state['pose'] == [10, 20, 8, -45, 315, 0]
    assert HEADINGS == list(range(0, 360, 45))
    assert state['edges']['row_plus'] == {
        'to': 'r02c01_d07', 'distance_m': 4, 'yaw_delta_deg': 0,
        'cost_s': 2, 'traversability': 'unknown', 'collision_check': 'unknown',
    }
    assert state['edges']['yaw_right']['to'] == 'r01c01_d00'
    assert state['edges']['yaw_right']['cost_s'] == .5
    assert 'row_minus' not in graph['states']['r00c00_d00']['edges']
    assert graph['states']['r00c00_d00']['edges']['hold_capture']['to'] == 'r00c00_d00'
    assert 'stop' not in state['edges']


def test_routes_use_labels_only_for_selection_and_real_graph_edges():
    graph = build_graph(3, 4, [0, 0, 8], -45, move_speed=2, turn_speed=90, hold_s=.25)
    frames = {}
    for sid, state in graph['states'].items():
        direction = state['direction_id']
        qa = 'usable' if direction in (0, 1) else 'weak' if direction == 2 else 'unknown'
        qb = 'usable' if direction in (0, 2, 3) else 'unknown'
        frames[sid] = dict(
            image_path=f'{sid}.png', sensor_frame=len(frames)+100,
            objects={uid: dict(quality=dict(recognition_quality=q, visibility_present=q != 'unknown'))
                     for uid, q in [('s:A', qa), ('s:B', qb)]},
        )
    objects = {'s:A': {'pose': [1, 1, 0]}, 's:B': {'pose': [-1, -1, 0]}}
    tasks = [(kind, 's:A') for kind in ('discover', 'compare', 'reappear', 'repeat_then_useful')]
    routes, gaps = derive_routes(graph, frames, objects, tasks, 4)
    assert not gaps
    assert len(routes) == 4
    for route in routes:
        labels, inputs = route_rows(graph, frames, route, 'reference.png')
        assert 16 <= len(labels) <= 32
        assert inputs[0]['time_s'] == 0
        for before, after, model in zip(labels[:-1], labels[1:], inputs[1:], strict=True):
            name = after['action_from_previous']['name']
            edge = graph['states'][before['state_id']]['edges'][name]
            assert edge['to'] == after['state_id']
            assert model['time_s'] - before['time_s'] == pytest.approx(edge['cost_s'])
            assert model['legal_action_mask'] is None
            assert set(model) == {'rgb', 'reference', 'time_s', 'previous_action_id', 'legal_action_mask'}
    repeat = next(r for r in routes if r['template'] == 'repeat_then_useful')
    assert repeat['actions'][:2] == ['hold_capture', 'hold_capture']
    assert len(set(repeat['states'][:3])) == 1


def test_grid_dataset_roundtrip_and_corruption(tmp_path):
    from pathlib import Path

    import numpy as np

    from dinoexp.core import checked_path
    from dinoexp.dense_cache import save_json
    from dinoexp.grid_search_capture import graph_for, grid_plan, init_collection, save_frame
    from dinoexp.grid_search_preview import publish_preview
    from dinoexp.grid_search_validation import read_json, read_layout, save_routes, validate_collection
    from dinoexp.search_capture import intrinsics, save_reference
    from dinoexp.search_episodes import instance_labels, iter_v1_inputs

    config = read_json(Path(__file__).resolve().parents[1] / 'configs/search_grid_v1.json')
    config['grid']['size'] = 3
    config['capture'].update(width=384, height=288)
    case, = grid_plan(config, 'smoke')
    root = tmp_path / 'synthetic_grid'
    manifest = init_collection(root, config, [case], 'smoke', {})
    directory = root / case['path']
    for folder in ('audit', 'rgb', 'labels', 'references/A', 'references/B'):
        (directory / folder).mkdir(parents=True)
    anchor = [0, 0, 0, 0, 0, 0]
    graph = graph_for(config, case, anchor)
    calibration = intrinsics(config['capture'])
    tags = [14]
    candidates = config['capture']['candidates']
    objects = {}
    mappings = {}
    refs = {}
    pixels = np.full((288, 384, 3), 127, dtype=np.uint8)

    def metadata(frame, pose):
        pair = dict(frame=frame, timestamp=frame*.05, width=384, height=288,
                    intrinsics=calibration, pose=pose)
        return dict(rgb=dict(pair), instance=dict(pair))

    for index, candidate in enumerate(candidates, 1):
        uid = case['scene_id'] + ':' + candidate['identity']
        xyz = [-4 if index == 1 else 4, 0, 0]
        objects[uid] = dict(**candidate, actual_blueprint=candidate['blueprint'],
                            actor_attributes={'color': candidate['color']}, pose=[*xyz, 0, 0, 0],
                            center=xyz, bbox_radius_xy_m=2.5)
        mappings[uid] = dict(ids=[256+index], tags=tags, restored_frame=10)
        ref_raw = np.zeros_like(pixels)
        ref_raw[60:220, 60:320] = [14, index, 1]
        mask = (ref_raw[:, :, 0] == 14)
        parent = directory / 'references' / candidate['identity']
        refs[uid] = save_reference(root, parent, pixels, ref_raw,
                                   metadata(10+index, [0, -7, 4, -25, 90, 0]), mask, uid)
    for point in graph['points'].values():
        point['capture_status'] = 'captured'
    save_json(directory / 'graph.json', graph)
    for index, (sid, state) in enumerate(graph['states'].items(), 100):
        raw = np.zeros_like(pixels)
        direction = state['direction_id']
        if direction in (0, 1, 2):
            start = 0 if direction == 2 else 20
            raw[30:110, start:start+90] = [14, 1, 1]
        if direction in (0, 2, 3, 4):
            start = 0 if direction == 4 else 100
            raw[start:start+90, 200:300] = [14, 2, 1]
        masks, ignore = instance_labels(raw, mappings, tags)
        save_frame(root, directory, case, sid, state, pixels, raw, metadata(index, state['pose']),
                   masks, ignore)
    save_json(directory / 'scene.json', dict(
        status='captured', case=case, cleanup_errors=[], map=config['map'], carla_versions=['test']*2,
        instance_id_method='remove_restore_probe', instance_mapping=mappings, vehicle_tags=tags,
        objects=objects, bbox_clearance_lower_bound_m=3, references=refs, intrinsics=calibration,
        anchor=anchor, capture_count=len(graph['states'])))
    scene, graph, frames = read_layout(root, config, case)
    save_routes(root, case, scene, graph, frames, 'smoke')
    manifest['status'] = 'complete'
    result = validate_collection(root, manifest)
    assert (result['source_images'], result['derived_sequences'], result['offline_coverage_gaps']) == (72, 8, [])
    assert result['splits']['train']['reference_targets'] == {'A': 4, 'B': 4}
    publish_preview(root, manifest, result)
    assert (root / 'preview/index.html').is_file()
    assert (root / 'preview/scene_00_r00c00_d00_518.png').is_file()
    routes = read_json(directory / 'episodes/index.json')
    example = next(iter_v1_inputs(root, routes['paths'][0]))
    assert example['rgb'].shape == (288, 384, 3)
    inputs = checked_path(root, routes['paths'][0]) / 'inputs.jsonl'
    inputs.write_text(inputs.read_text().replace('"time_s": 0,', '"time_s": 99,', 1), encoding='utf-8')
    with pytest.raises(ValueError, match='runtime allowlist'):
        validate_collection(root, manifest)
