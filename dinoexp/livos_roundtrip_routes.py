"""One outward camera walk and its observed return. Pure data; no model imports."""
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
YAWS = ('yaw_left', 'yaw_right')
REVERSE = dict(row_plus='row_minus', row_minus='row_plus', col_plus='col_minus',
               col_minus='col_plus', yaw_left='yaw_right', yaw_right='yaw_left')
SOURCES = ['run_livos_roundtrip_test.py', 'dinoexp/livos_roundtrip_routes.py']
PHASE_NAMES = dict(outbound='去程', turnaround='折返点', returning='返程')


def choose_routes(scene, graph, frames, *, min_pixels=8192, max_routes=1,
                  max_outward=17, beam_width=128):
    """Bounded deterministic search, at most four separated 45-degree yaw steps outward."""
    if min(min_pixels, max_routes, beam_width) < 1 or max_outward < 3:
        raise ValueError('positive limits and max-outward >= 3 required')
    local = {f['state']: f for f in frames.values() if f['scene'] == scene}
    identities = {tuple(sorted(f['objects'])) for f in local.values()}
    if len(identities) != 1 or len(next(iter(identities), ())) != 2:
        raise ValueError('roundtrip requires exactly two consistent vehicle identities')
    eligible = {s for s, f in local.items()
                if all(o['pixels'] >= min_pixels for o in f['objects'].values())}
    starts = sorted(s for s in eligible
                    if not any(o['boundary'] for o in local[s]['objects'].values()))

    def rank(item):
        path, actions = item
        positions = {tuple(graph['states'][s]['pose'][:3]) for s in path}
        movement = [a for a in actions if a in MOVES]
        bends = sum(a != b for a, b in zip(movement[:-1], movement[1:], strict=True))
        clear = sum(not any(o['boundary'] for o in local[s]['objects'].values()) for s in path)
        minimum = min(o['pixels'] for s in path for o in local[s]['objects'].values())
        return (-len(path), -len(positions), -clear, bends, -minimum, tuple(path))

    candidates = []
    for start in starts:
        beam, complete = [([start], [])], []
        for _ in range(max_outward-1):
            expanded = []
            for path, actions in beam:
                current = path[-1]
                positions = {tuple(graph['states'][s]['pose'][:3]) for s in path}
                turns = [a for a in actions if a in YAWS]
                for action in MOVES+YAWS:
                    edge = graph['states'][current]['edges'].get(action)
                    if not edge or edge['to'] not in eligible or edge['to'] in path:
                        continue
                    end = edge['to']
                    back = graph['states'][end]['edges'].get(REVERSE[action])
                    if not back or back['to'] != current:
                        continue
                    a, b = graph['states'][current]['pose'], graph['states'][end]['pose']
                    if (a[2], a[3], a[5]) != (b[2], b[3], b[5]):
                        continue
                    if action in MOVES:
                        if a[3:] != b[3:] or tuple(b[:3]) in positions:
                            continue
                        movement = [x for x in actions+[action] if x in MOVES]
                        if sum(x != y for x, y in zip(movement[:-1], movement[1:], strict=True)) > 2:
                            continue
                    else:
                        delta = (b[4]-a[4]+180) % 360-180
                        expected = 45 if action == 'yaw_right' else -45
                        if (a[:3] != b[:3] or not actions or actions[-1] in YAWS
                                or not math.isclose(delta, expected, abs_tol=1e-6)
                                or len(turns) >= 4 or (turns and turns[0] != action)):
                            continue
                    nxt = (path+[end], actions+[action])
                    expanded.append(nxt)
                    if action in MOVES and len(nxt[0]) >= 3:
                        complete.append(nxt)
            beam = sorted(expanded, key=rank)[:beam_width]
            if not beam:
                break
        if complete:
            candidates.append(min(complete, key=rank))
    routes = []
    used = set()
    for outward, actions_out in sorted(candidates, key=rank):
        # The same geometric outward path in reverse is not a second primary route.
        key = min(tuple(outward), tuple(reversed(outward)))
        if key in used:
            continue
        used.add(key)
        states = outward+outward[-2::-1]
        actions = actions_out+[REVERSE[a] for a in reversed(actions_out)]
        poses = [graph['states'][s]['pose'] for s in states]
        sequence = [local[s] for s in states]
        n = len(outward)
        route = dict(id=f'{scene}/roundtrip_{len(routes):02d}', scene=scene, states=states,
                     actions=actions, poses=poses,
                     costs_s=[graph['states'][s]['edges'][a]['cost_s']
                              for s, a in zip(states[:-1], actions, strict=True)],
                     outward_frames=n, turnaround_step=n-1,
                     phase_by_step=['outbound']*(n-1)+['turnaround']+['returning']*(n-1),
                     replay_of_step=[None]*n+list(range(n-2, -1, -1)),
                     unique_frames=n, unique_positions=len({tuple(p[:3]) for p in poses}),
                     translation_steps=sum(a in MOVES for a in actions),
                     stationary_yaw_steps=sum(a in YAWS for a in actions),
                     outward_yaw_steps=sum(a in YAWS for a in actions_out),
                     distance_m=sum(math.dist(a[:3], b[:3])
                                    for a, b in zip(poses[:-1], poses[1:], strict=True)),
                     both_unclipped_steps=sum(not any(o['boundary'] for o in f['objects'].values())
                                              for f in sequence),
                     min_vehicle_pixels=min(o['pixels'] for f in sequence for o in f['objects'].values()))
        validate_route(route, graph, local)
        routes.append(route)
        if len(routes) == max_routes:
            break
    return routes


def prepare(args):
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('output exists; choose a new directory to preserve results')
    with TemporaryDirectory(prefix='livos-roundtrip-audit-') as temporary:
        audit_args = SimpleNamespace(data=str(root), output=str(Path(temporary)/'audit'),
                                     length=2, stride=16, min_init_pixels=1,
                                     size=args.size, max_cases=1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = audit_capture(audit_args)
    routes, scenes = [], {}
    for capture in read_json(root/'dataset_manifest.json')['cases']:
        scene = capture['scene_id']
        graph = read_json(root/capture['path']/'graph.json')
        scenes[scene] = read_json(root/capture['path']/'scene.json')
        routes.extend(choose_routes(scene, graph, result['frames'], min_pixels=args.min_view_pixels,
                                     max_routes=args.max_routes, max_outward=args.max_outward,
                                     beam_width=args.beam_width))
    if not routes:
        raise ValueError('no reversible three-frame outward route keeps both vehicles visible')
    cases = []
    for route in routes:
        keys = [route['scene']+'/'+s for s in route['states']]
        sequence = [result['frames'][k] for k in keys]
        for uid in sorted(sequence[0]['objects']):
            cases.append(dict(id=f'case_{len(cases):03d}', source_id=route['id']+'/'+uid,
                              route_id=route['id'], target=uid, frames=keys,
                              actions=route['actions'], costs_s=route['costs_s'], tags=tags_for(sequence, uid)))
    result['routes'], result['cases'] = routes, cases
    result['config'].update(length=None, stride=None, max_cases=len(cases),
                            route_policy='two_vehicle_roundtrip', min_init_pixels=args.min_view_pixels,
                            min_view_pixels=args.min_view_pixels, max_routes=args.max_routes,
                            max_outward=args.max_outward, beam_width=args.beam_width)
    selected = [result['frames'][k] for c in cases for k in c['frames']]
    unique = {k for c in cases for k in c['frames']}
    result['counts'].update(
        candidate_tasks=len(cases), selected_routes=len(routes), selected_tasks=len(cases),
        total_steps=len(selected), scoring_steps=len(selected)-len(cases),
        selected_unique_rgb=len({f['image_sha256'] for f in selected}),
        selected_unique_route_frames=len(unique), route_frame_counts=[len(r['states']) for r in routes],
        outward_frame_counts=[r['outward_frames'] for r in routes],
        by_target=dict(Counter(c['target'] for c in cases)),
        selected_tags=dict(Counter(t for c in cases for t in c['tags'])),
        both_visible_steps=len(selected),
        both_unclipped_steps=sum(not any(o['boundary'] for o in f['objects'].values()) for f in selected),
        stationary_yaw_steps=2*sum(r['stationary_yaw_steps'] for r in routes),
        translation_steps=2*sum(r['translation_steps'] for r in routes))
    result['sampling'] = ('Offline GT-selected routes, not learned navigation. Bounded beam search prefers '
                          'longer outward paths, more unique positions and unclipped views. Both cars meet '
                          'the pixel threshold at every step; both unclipped initially. No outward state '
                          'revisits or translation backtracking, at most two travel bends and four same-direction '
                          '45-degree yaw steps, separated by translations. Return retraces captured reciprocal '
                          'edges exactly once. A and B use identical frames in independent runs.')
    result['limitations'] = ('Return reuses the exact static outward RGBs; total steps are not independent '
                             'new images. Movement reverses at the endpoint while the camera keeps facing '
                             'the cars; no captured 180-degree body/camera-turn video or interpolated frames. '
                             'One layout/two static cars, some image-edge truncation, 4 m/45-degree capture '
                             'steps, no collision-free flight certification, no new identity generalization. '
                             'The search is bounded, not a proof of a globally longest possible route.')
    result['route_selection_code'] = {p: sha(ROOT/p) for p in SOURCES}
    output.mkdir(parents=True)
    write_json(output/'plan.json', result)
    write_overview(output, result, scenes)
    print('Roundtrip plan:', result['counts'], flush=True)
    return result


def write_overview(output, plan, scenes):
    cards = []
    for route in plan['routes']:
        fs = [plan['frames'][route['scene']+'/'+s] for s in route['states']]
        all_poses = [f['pose'] for f in plan['frames'].values() if f['scene'] == route['scene']]
        low_x, low_y = min(p[0] for p in all_poses), min(p[1] for p in all_poses)

        def xy(p, ox=low_x, oy=low_y):
            return 35+(p[0]-ox)*15, 300-(p[1]-oy)*15

        shapes = []
        for pose in sorted({tuple(p[:3]) for p in all_poses}):
            x, y = xy(pose)
            shapes.append(f'<circle cx="{x}" cy="{y}" r="2" fill="#c4ced8"/>')
        for uid, obj in scenes[route['scene']].get('objects', {}).items():
            x, y = xy(obj['center'])
            shapes.append(f'<text x="{x}" y="{y}" fill="#a44e1c">车{html.escape(uid.rsplit(":", 1)[-1])}</text>')
        outward = route['poses'][:route['outward_frames']]
        points = ' '.join(f'{x},{y}' for x, y in map(xy, outward))
        shapes.append(f'<polyline points="{points}" fill="none" stroke="#236ba0" stroke-width="3"/>')
        for i, pose in enumerate(outward):
            x, y = xy(pose)
            yaw = math.radians(pose[4])
            shapes.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#236ba0"/>'
                          f'<path d="M{x},{y} l{17*math.cos(yaw)},{-17*math.sin(yaw)}" stroke="#16805a"/>'
                          f'<text x="{x+6}" y="{y-7}" font-size="11">{i+1}</text>')
        rows = []
        for i, f in enumerate(fs):
            pixels = ' / '.join(uid.rsplit(':', 1)[-1]+': '+str(o['pixels'])+
                                ('（截边）' if o['boundary'] else '') for uid, o in sorted(f['objects'].items()))
            replay = route['replay_of_step'][i]
            note = '' if replay is None else f'；重访第 {replay+1} 张（相同原图）'
            action = '首帧' if i == 0 else route['actions'][i-1]
            rows.append(f'<tr><td>{i+1}</td><td>{PHASE_NAMES[route["phase_by_step"][i]]}{note}</td>'
                        f'<td>{f["state"]}</td><td>{action}</td><td>{pixels}</td></tr>')
        cards.append(f'<section><h2>{html.escape(route["id"])}</h2><p>共 {len(fs)} 步，'
                     f'{route["unique_frames"]} 张不同照片，{route["unique_positions"]} 个不同位置；'
                     f'平移 {route["distance_m"]:g} 米，分散转向 {route["stationary_yaw_steps"]} 次。</p>'
                     '<svg viewBox="0 0 350 340" role="img" aria-label="去程位置和拍摄朝向；返程原路返回">'
                     + ''.join(shapes)+'</svg><table><tr><th>步</th><th>阶段</th><th>状态</th>'
                     '<th>上一动作</th><th>两车可见像素</th></tr>'+''.join(rows)+'</table></section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双车往返观察路线</title>'
            '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;'
            'background:#f4f6f9;color:#17283a}section{background:white;padding:20px;margin:20px 0;'
            'border-radius:10px}svg{width:350px;max-width:100%}table{border-collapse:collapse}'
            'td,th{padding:7px;border-bottom:1px solid #dde4eb;text-align:left;font-size:14px}</style>'
            '<h1>发现两车后的去程与返程</h1><p>蓝线为去程，绿色短线为相机拍摄朝向；返程沿原线反向行进。'
            '横轴世界 X，纵轴世界 Y，灰点相隔 4 米。转向仅为相邻的 45 度拍摄状态，之间有平移。'
            '折返点改变行进方向，相机继续看车辆。返程重用静态照片，不是新拍视频或 180 度转头画面。</p>'
            '<p>首帧两车完整；每步两车均达到可见像素门槛，后续截边在表内标出。'
            '同一路线分别以 A、B 为目标；此页没有模型分数。</p>'
            '<p><a href="preview/index.html">真实照片逐帧预览</a> · '
            '<a href="report/index.html">服务器运行后的模型结果</a></p>'+''.join(cards)+'</html>')
    (output/'routes.html').write_text(page, encoding='utf-8')


def load_roundtrip_plan(args):
    result = load_plan(Path(args.output), Path(args.data))
    if result['config'].get('route_policy') != 'two_vehicle_roundtrip':
        raise ValueError('need a roundtrip plan from run_livos_roundtrip_test.py --stage audit')
    if result.get('route_selection_code') != {p: sha(ROOT/p) for p in SOURCES}:
        raise ValueError('roundtrip route implementation changed; create a new output')
    return result


def render_report(args, preview=False):
    from .livos_vehicle_report import report
    plan = load_roundtrip_plan(args)
    report(args, preview=preview)
    directory = Path(args.output)/('preview' if preview else 'report')
    page = directory/'index.html'
    source = page.read_text(encoding='utf-8')
    old = ('这是离散相机路线：相邻拍摄位置相差 4 米或方向相差 45 度。路线有重复与交叠，'
           '统计不代表独立车辆泛化；目标消失可能是离开画面，不能直接称为遮挡。')
    new = ('这是双车持续同框的往返观察：去程平移为主，夹少量分散的 45 度转向；'
           '折返点改变行进方向，返程按原路重访，使用与去程完全相同的静态照片。'
           '部分车身仍会截边；首帧之后不再给真值。步数不等于不同照片数，统计不代表新车辆泛化。'
           '<a href="../routes.html">查看每一步的去程、折返点和返程表</a>')
    info = "(si===0?'首帧指定目标，不计效果':'上一动作 '+f.action)"
    if source.count(old) != 1 or source.count(info) != 1:
        raise ValueError('common viewer changed; inspect before adapting')
    routes = {r['id']: r for r in plan['routes']}
    phases = {c['id']: routes[c['route_id']]['phase_by_step'] for c in plan['cases']}
    import json
    phase_js = json.dumps(phases, ensure_ascii=False).replace('<', '\\u003c')
    source = source.replace(old, new).replace('const DATA=', f'const PHASES={phase_js};const DATA=', 1)
    source = source.replace(info, info+"+'；阶段 '+({outbound:'去程',turnaround:'折返点',returning:'返程'})[PHASES[c.id][si]]")
    page.write_text(source, encoding='utf-8')
    payload = read_json(directory/'data.json')
    for case in payload['cases']:
        for step, phase in zip(case['steps'], phases[case['id']], strict=True):
            step['phase'] = phase
    write_json(directory/'data.json', payload)
    load_roundtrip_plan(args)
