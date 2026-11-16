"""Multi-scene route export and viewer wrapper, preserving all older entry points."""
from collections import Counter
import html
from pathlib import Path

from . import livos_roundtrip_routes as roundtrip
from .multiscene_capture import (
    atomic_json, provenance, read_json, validate_collection,
)


def prepare(args):
    root = Path(args.data).resolve()
    report = validate_collection(root)
    manifest = read_json(root/'dataset_manifest.json')
    policy = manifest['config']['route']
    # Use the exact capture acceptance settings, so every accepted scene is represented.
    for key in ('min_view_pixels', 'max_routes', 'max_outward', 'beam_width'):
        setattr(args, key, policy[key])
    result = roundtrip.prepare(args)
    # A requested second route may be shorter than the accepted first one.
    # Apply the same minimum-length filter used during capture acceptance.
    result['routes'] = [r for r in result['routes'] if r['outward_frames'] >= policy['min_outward']]
    route_ids = {r['id'] for r in result['routes']}
    result['cases'] = [c for c in result['cases'] if c['route_id'] in route_ids]
    for i, case in enumerate(result['cases']):
        case['id'] = f'case_{i:03d}'
    selected = [result['frames'][k] for c in result['cases'] for k in c['frames']]
    result['config']['max_cases'] = len(result['cases'])
    result['counts'].update(
        candidate_tasks=len(result['cases']), selected_routes=len(result['routes']),
        selected_tasks=len(result['cases']), total_steps=len(selected),
        scoring_steps=len(selected)-len(result['cases']),
        selected_unique_rgb=len({f['image_sha256'] for f in selected}),
        selected_unique_route_frames=len({k for c in result['cases'] for k in c['frames']}),
        route_frame_counts=[len(r['states']) for r in result['routes']],
        outward_frame_counts=[r['outward_frames'] for r in result['routes']],
        by_target=dict(Counter(c['target'] for c in result['cases'])),
        selected_tags=dict(Counter(t for c in result['cases'] for t in c['tags'])),
        both_visible_steps=len(selected),
        both_unclipped_steps=sum(not any(o['boundary'] for o in f['objects'].values()) for f in selected),
        stationary_yaw_steps=2*sum(r['stationary_yaw_steps'] for r in result['routes']),
        translation_steps=2*sum(r['translation_steps'] for r in result['routes']))
    metadata = {c['scene_id']: c for c in manifest['cases']}
    if set(metadata) != {r['scene'] for r in result['routes']}:
        raise ValueError('some accepted scenes have no exported route')
    result['multiscene_code'] = provenance()
    result['collection'] = dict(validation=report, collection_status=manifest['collection_status'],
                                requested_backgrounds=manifest['config']['sampling']['backgrounds'],
                                requested_scenes=len(manifest['slots']))
    for frame in result['frames'].values():
        source = metadata[frame['scene']]
        frame.update(group=source['group'], split=source['split'])
    for case in result['cases']:
        source = metadata[result['frames'][case['frames'][0]]['scene']]
        case.update(group=source['group'], split=source['split'])
    result['config'].update(min_outward=policy['min_outward'], multiscene=True)
    result['counts'].update(accepted_scenes=report['accepted_scenes'],
                            accepted_backgrounds=report['accepted_backgrounds'],
                            scenes_by_split=report['by_split'])
    spacing = manifest['config']['grid']['spacing_m']
    result['limitations'] = (f'{report["accepted_scenes"]} accepted layouts in '
        f'{report["accepted_backgrounds"]} background groups. Backgrounds are separated in world XY; '
        'this does not prove visual independence or new physical-vehicle identity generalization. '
        f'{spacing:g} m / 45-degree discrete observations, static cars, some edge truncation. '
        'Return reuses the same outward photographs. GT-based offline route selection; '
        'no collision-free flight or continuous captured return-video claim. '
        'The bounded route search is not guaranteed globally optimal.')
    output = Path(args.output)
    atomic_json(output/'plan.json', result)
    scenes = {c['scene_id']: read_json(root/c['path']/'scene.json') for c in manifest['cases']}
    roundtrip.write_overview(output, result, scenes)
    page = output/'routes.html'
    text = page.read_text(encoding='utf-8').replace('灰点相隔 4 米', f'灰点相隔 {spacing:g} 米')
    cards = ['<h2>采集分组</h2><p>同一背景的所有摆车布局使用同一分组；训练和验证按背景划分。'
             '车辆仍为配置中的两种资产，不能据此声称新车辆身份泛化。</p><ul>']
    for capture in manifest['cases']:
        cards.append('<li>'+html.escape(f'{capture["scene_id"]} · {capture["group"]} · '
                                       f'{capture["split"]} · 第 {capture["randomization"]["attempt"]} 次摆车')+'</li>')
    text = text.replace('</html>', ''.join(cards)+'</ul></html>')
    page.write_text(text, encoding='utf-8')
    return result


def load_plan(args):
    result = roundtrip.load_roundtrip_plan(args)
    if not result['config'].get('multiscene') or result.get('multiscene_code') != provenance():
        raise ValueError('multi-scene implementation changed or wrong plan type')
    return result


def render_report(args, preview=False):
    plan = load_plan(args)
    roundtrip.render_report(args, preview=preview)
    directory = Path(args.output)/('preview' if preview else 'report')
    payload = read_json(directory/'data.json')
    cases = {c['id']: c for c in plan['cases']}
    for case in payload['cases']:
        case.update(group=cases[case['id']]['group'], split=cases[case['id']]['split'])
    atomic_json(directory/'data.json', payload)
    page = directory/'index.html'
    text = page.read_text(encoding='utf-8')
    info = plan['collection']['validation']
    note = (f'<p>本批次：{info["accepted_scenes"]} 个布局，{info["accepted_backgrounds"]} 个背景；'
            f'训练/验证布局数 {html.escape(str(info["by_split"]))}。'
            '同一背景内的布局属于同一个数据分组。此划分不能证明未见车辆身份泛化。</p>')
    text = text.replace('<h1>', note+'<h1>', 1)
    page.write_text(text, encoding='utf-8')
    load_plan(args)
