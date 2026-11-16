"""Prepare short translation routes that keep both vehicles in view. No torch."""
import contextlib
import html
import io
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from .livos_vehicle_data import ROOT, load_plan, plan as audit_capture, tags_for
from .region_ab_data import read_json, sha, write_json
from .region_sequence_routes import validate_route

MOVES = ('row_plus', 'row_minus', 'col_plus', 'col_minus')


def choose_routes(scene, graph, frames, *, min_pixels=8192, max_routes=6):
    """One route per heading; no yaw, no revisits, at most one change in travel direction."""
    if min_pixels < 1 or max_routes < 1:
        raise ValueError('min pixels and max routes must be positive')
    local = {f['state']: f for f in frames.values() if f['scene'] == scene}
    identities = {tuple(sorted(f['objects'])) for f in local.values()}
    if len(identities) != 1 or len(next(iter(identities), ())) != 2:
        raise ValueError('follow routes require exactly two consistent vehicle identities')
    eligible = {s for s, f in local.items()
                if all(o['pixels'] >= min_pixels for o in f['objects'].values())}
    starts = sorted(s for s in eligible
                    if not any(o['boundary'] for o in local[s]['objects'].values()))
    best = {}

    def visit(path, actions):
        if len(path) >= 3:
            sequence = [local[s] for s in path]
            turns = sum(a != b for a, b in zip(actions[:-1], actions[1:], strict=True))
            clear = sum(not any(o['boundary'] for o in f['objects'].values()) for f in sequence)
            smallest = min(o['pixels'] for f in sequence for o in f['objects'].values())
            total = sum(min(o['pixels'] for o in f['objects'].values()) for f in sequence)
            rank = (-len(path), -clear, turns, -smallest, -total, tuple(path))
            heading = tuple(graph['states'][path[0]]['pose'][3:])
            if heading not in best or rank < best[heading][0]:
                best[heading] = (rank, list(path), list(actions))
        if len(path) == 6:
            return
        for action in MOVES:
            edge = graph['states'][path[-1]]['edges'].get(action)
            if not edge or edge['to'] not in eligible or edge['to'] in path:
                continue
            end = edge['to']
            a = graph['states'][path[-1]]['pose']
            b = graph['states'][end]['pose']
            if a[2:] != b[2:] or tuple(b[:3]) in {tuple(graph['states'][s]['pose'][:3]) for s in path}:
                continue
            next_actions = actions+[action]
            if sum(x != y for x, y in zip(next_actions[:-1], next_actions[1:], strict=True)) > 1:
                continue
            visit(path+[end], next_actions)

    for start in starts:
        visit([start], [])
    chosen = sorted(best.values(), key=lambda item: item[0])[:max_routes]
    chosen.sort(key=lambda item: (graph['states'][item[1][0]]['pose'][4] % 360, item[1]))
    routes = []
    for i, (_, states, actions) in enumerate(chosen):
        edges = [graph['states'][s]['edges'][a]
                 for s, a in zip(states[:-1], actions, strict=True)]
        sequence = [local[s] for s in states]
        route = dict(id=f'{scene}/follow_{i:02d}', scene=scene, states=states, actions=actions,
                     costs_s=[e['cost_s'] for e in edges],
                     poses=[graph['states'][s]['pose'] for s in states],
                     yaw_deg=graph['states'][states[0]]['pose'][4],
                     distance_m=sum(math.dist(graph['states'][a]['pose'][:3],
                                              graph['states'][b]['pose'][:3])
                                    for a, b in zip(states[:-1], states[1:], strict=True)),
                     turns=sum(a != b for a, b in zip(actions[:-1], actions[1:], strict=True)),
                     both_unclipped_steps=sum(not any(o['boundary'] for o in f['objects'].values())
                                              for f in sequence),
                     min_vehicle_pixels=min(o['pixels'] for f in sequence for o in f['objects'].values()))
        validate_route(route, graph, local)
        routes.append(route)
    return routes


def prepare(args):
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('output exists; choose a new directory to preserve results')
    if args.min_view_pixels < 1 or args.max_routes < 1:
        raise ValueError('min-view-pixels and max-routes must be positive')
    # Reuse the unchanged capture auditor; its temporary patrol is never evaluated.
    # Keeping the common runner unchanged also preserves checks of existing runs.
    with TemporaryDirectory(prefix='livos-follow-audit-') as temporary:
        audit_args = SimpleNamespace(data=str(root), output=str(Path(temporary)/'audit'),
                                     length=2, stride=16, min_init_pixels=1,
                                     size=args.size, max_cases=1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = audit_capture(audit_args)
    manifest = read_json(root/'dataset_manifest.json')
    routes = []
    scenes = {}
    for capture in manifest['cases']:
        scene = capture['scene_id']
        graph = read_json(root/capture['path']/'graph.json')
        scenes[scene] = read_json(root/capture['path']/'scene.json')
        routes.extend(choose_routes(scene, graph, result['frames'],
                                    min_pixels=args.min_view_pixels, max_routes=args.max_routes))
    if not routes:
        raise ValueError('no three-frame translation route keeps both vehicles sufficiently visible')
    cases = []
    for route in routes:
        keys = [route['scene']+'/'+s for s in route['states']]
        sequence = [result['frames'][k] for k in keys]
        for uid in sorted(sequence[0]['objects']):
            cases.append(dict(id=f'case_{len(cases):03d}', source_id=route['id']+'/'+uid,
                              route_id=route['id'], target=uid, frames=keys,
                              actions=route['actions'], costs_s=route['costs_s'],
                              tags=tags_for(sequence, uid)))
    result['cases'] = cases
    result['routes'] = routes
    result['config'].update(length=None, stride=None, max_cases=len(cases),
                            route_policy='two_vehicle_translation', max_routes=args.max_routes,
                            min_init_pixels=args.min_view_pixels,
                            min_view_pixels=args.min_view_pixels)
    selected = [result['frames'][k] for c in cases for k in c['frames']]
    unique = {k: result['frames'][k] for c in cases for k in c['frames']}
    eligible = [f for f in result['frames'].values()
                if all(o['pixels'] >= args.min_view_pixels for o in f['objects'].values())]
    result['counts'].update(candidate_tasks=len(cases), selected_routes=len(routes),
                            selected_tasks=len(cases), total_steps=len(selected),
                            scoring_steps=sum(len(c['frames'])-1 for c in cases),
                            eligible_both_vehicle_states=len(eligible),
                            selected_unique_route_frames=len(unique),
                            selected_unique_unclipped_frames=sum(
                                not any(o['boundary'] for o in f['objects'].values())
                                for f in unique.values()),
                            selected_unique_rgb=len({f['image_sha256'] for f in selected}),
                            route_frame_counts=[len(r['states']) for r in routes],
                            by_target=dict(Counter(c['target'] for c in cases)),
                            selected_tags=dict(Counter(t for c in cases for t in c['tags'])),
                            both_visible_steps=len(selected),
                            both_unclipped_steps=sum(not any(o['boundary'] for o in f['objects'].values())
                                                     for f in selected),
                            stationary_yaw_steps=0, translation_steps=sum(len(c['actions']) for c in cases))
    result['sampling'] = ('Offline annotation-selected two-vehicle following segments. Every frame keeps '
                          'both vehicles above the declared pixel threshold. Fixed camera orientation, '
                          'translation only, no repeated position, at most one bend, both cars unclipped '
                          'at initialization. Prefer longer paths, then more unclipped views; one path '
                          'per orientation. No predictions used. A/B are tested on the same frames in '
                          'separate runs. Selected routes are not autonomous navigation or independent samples.')
    result['limitations'] = ('One layout/two static cars; this approximates camera translation while '
                             'keeping acquired vehicles in view, not following moving cars. Partial image-edge '
                             'truncation remains. Existing captures are 4 m apart: there are no intermediate '
                             'video frames. Short routes retain their actual length; no cycling or padding. '
                             'No collision-free flight certification or new identity generalization claim.')
    result['route_selection_code'] = {p: sha(ROOT/p) for p in
                                      ['run_livos_follow_test.py', 'dinoexp/livos_follow_routes.py']}
    output.mkdir(parents=True)
    write_json(output/'plan.json', result)
    write_overview(output, result, scenes)
    print('Follow plan:', result['counts'], flush=True)
    return result


def write_overview(output, result, scenes):
    cards = []
    for route in result['routes']:
        fs = [result['frames'][route['scene']+'/'+s] for s in route['states']]
        all_poses = [f['pose'] for f in result['frames'].values() if f['scene'] == route['scene']]
        low_x, low_y = min(p[0] for p in all_poses), min(p[1] for p in all_poses)
        def xy(p, origin_x=low_x, origin_y=low_y):
            return 38+(p[0]-origin_x)*15, 290-(p[1]-origin_y)*15
        shapes = []
        for pose in sorted({tuple(p) for p in all_poses}):
            x, y = xy(pose)
            shapes.append(f'<circle cx="{x}" cy="{y}" r="2" fill="#ccd5df"/>')
        for uid, obj in scenes[route['scene']].get('objects', {}).items():
            x, y = xy(obj['center'])
            shapes.append(f'<rect x="{x-7}" y="{y-4}" width="14" height="8" fill="#c6612b"/>'
                          f'<text x="{x+10}" y="{y+5}">{html.escape(uid.rsplit(":", 1)[-1])}</text>')
        points = ' '.join(f'{x},{y}' for x, y in map(xy, route['poses']))
        shapes.append(f'<polyline points="{points}" fill="none" stroke="#1767a0" stroke-width="3"/>')
        for i, pose in enumerate(route['poses']):
            x, y = xy(pose)
            shapes.append(f'<circle cx="{x}" cy="{y}" r="10" fill="#1767a0"/>'
                          f'<text x="{x}" y="{y+4}" text-anchor="middle" fill="white">{i+1}</text>')
        x, y = xy(route['poses'][0])
        yaw = math.radians(route['yaw_deg'])
        shapes.append(f'<path d="M{x},{y} l{26*math.cos(yaw)},{-26*math.sin(yaw)}" '
                      f'stroke="#15805c" stroke-width="3" marker-end="url(#arrow_{len(cards)})"/>')
        rows = []
        for i, f in enumerate(fs):
            fields = ' / '.join(f'{uid.rsplit(":", 1)[-1]}: {o["pixels"]}'
                                + ('（截边）' if o['boundary'] else '')
                                for uid, o in sorted(f['objects'].items()))
            rows.append(f'<tr><td>{i+1}</td><td>{f["state"]}</td><td>{fields}</td></tr>')
        cards.append(f'<section><h2>{html.escape(route["id"])} · 朝向 {route["yaw_deg"]}°</h2>'
                     f'<p>{len(fs)} 张，平移 {route["distance_m"]:g} 米，'
                     f'移动方向改变 {route["turns"]} 次，原地旋转 0 次。</p>'
                     '<svg viewBox="0 0 330 330" role="img" aria-label="相机平移路线与车辆位置">'
                     f'<defs><marker id="arrow_{len(cards)}" markerWidth="7" markerHeight="7" refX="5" refY="3" '
                     'orient="auto"><path d="M0,0 L6,3 L0,6" fill="#15805c"/></marker></defs>'
                     + ''.join(shapes)+'</svg><table><tr><th>顺序</th><th>相机状态</th><th>两车像素</th></tr>'
                     + ''.join(rows)+'</table></section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双车持续同框路线</title>'
            '<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f9;color:#17283a;'
            'margin:28px auto;max-width:1250px;padding:0 20px}.cards{display:grid;'
            'grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}'
            'section{background:white;padding:22px;border-radius:10px}h2{font-size:20px}'
            'svg{width:300px;max-width:100%}td,th{padding:6px;border-bottom:1px solid #dde4eb;'
            'text-align:left;font-size:14px}table{border-collapse:collapse}a{color:#1767a0}</style>'
            '<h1>发现两车后的持续观察路线</h1><p>相机沿蓝线移动；编号是照片顺序；橙色 A/B 是两车位置；'
            '绿色短箭头是固定拍摄朝向。横轴为世界 X，纵轴为世界 Y，灰点间距 4 米。</p>'
            '<p>所有帧两车都达到预设可见像素数；首帧两车均不截边。后续截边仍会出现，表中逐项标出。'
            '每条路线分别以 A、B 为目标测试；此页没有模型分数。</p>'
            '<p><a href="preview/index.html">打开真实照片逐帧预览</a> · '
            '<a href="report/index.html">打开运行后的模型结果（需服务器完成测试）</a></p>'
            '<div class="cards">'+''.join(cards)+'</div></html>')
    (output/'routes.html').write_text(page, encoding='utf-8')



def load_follow_plan(args):
    """Check common inputs and follow-route source hashes before any model work."""
    result = load_plan(Path(args.output), Path(args.data))
    if result['config'].get('route_policy') != 'two_vehicle_translation':
        raise ValueError('need a follow plan from run_livos_follow_test.py --stage audit')
    expected = {p: sha(ROOT/p) for p in
                ['run_livos_follow_test.py', 'dinoexp/livos_follow_routes.py']}
    if result.get('route_selection_code') != expected:
        raise ValueError('follow route implementation changed; create a new output')
    return result


def render_report(args, preview=False):
    """Reuse saved-mask verification and viewer, with the correct flight description."""
    from .livos_vehicle_report import report
    load_follow_plan(args)
    report(args, preview=preview)
    page = Path(args.output)/('preview' if preview else 'report')/'index.html'
    source = page.read_text(encoding='utf-8')
    old = ('这是离散相机路线：相邻拍摄位置相差 4 米或方向相差 45 度。路线有重复与交叠，'
           '统计不代表独立车辆泛化；目标消失可能是离开画面，不能直接称为遮挡。')
    new = ('这是发现两车后的短平移路线：每步移动 4 米，拍摄朝向固定，不原地旋转。'
           '每帧两车均在画面内，部分车身仍有截边；同一路线分别以 A、B 为目标测试。'
           '这是两辆静止车辆的离散拍摄，统计不代表长视频或新车辆泛化。'
           '<a href="../routes.html">查看路线图和完整路线表</a>')
    if source.count(old) != 1:
        raise ValueError('common viewer description changed; inspect before adapting')
    page.write_text(source.replace(old, new), encoding='utf-8')
    load_follow_plan(args)
