"""Capture six fixed-height similar-car inspection groups, with resumable previews."""

import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import subprocess
import sys

from collect_match_photos import save_json
from dinoexp.core import checked_path
from dinoexp.dataset import load_dataset
from dinoexp.fixed_height_photos import build_plan
from dinoexp.more_photos import select_anchors


def verify_case(root, case, map_name):
    run = json.loads((root/'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'complete' or run.get('cleanup_errors') or run.get('config') != case['config'] or run.get('map') != map_name:
        raise ValueError('completed capture metadata does not match saved plan')
    reference, queries, rejected = load_dataset(root)
    expected = {(c['identity'], v['name']) for c in case['config']['candidates'] for v in case['config']['views']}
    if (rejected or len(queries) != len(expected) or
            {(q['identity'], q['view']) for q in queries} != expected or
            reference['identity'] != case['config']['target_identity']):
        raise ValueError('capture contains rejected or missing views')
    expected_z = case['anchor_z'] + case['config']['fixed_camera_height_m']
    for row in [reference, *queries]:
        z = json.loads(row['camera_pose'])[2]
        if not math.isfinite(z) or abs(z-expected_z) > .03:
            raise ValueError('recorded camera height differs from the fixed-height plan')


def write_index(root, manifest):
    cards = []
    for case in manifest['plan']['cases']:
        relative = manifest['completed'].get(case['name'])
        if relative:
            view = case['config']['views'][0]['name']
            cards.append(f'<section><h2>{html.escape(case["name"])}</h2>'
                         f'<p>{html.escape(case["pair_kind"])}</p><a href="{relative}/preview.html">'
                         f'<img loading="lazy" src="{relative}/images/{view}.png">查看全部角度与两车裁剪</a></section>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>固定高度车辆预览</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#fafafa}main{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}section{background:white;'
        'padding:16px}img{width:100%}</style><h1>固定高度 · 双车环绕拍摄</h1>'
        '<p>先检查车身差异、遮挡、裁剪完整性；此集合用于看图，未划分训练/测试。</p><main>'
        + ''.join(cards)+'</main>', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    destinations = p.add_mutually_exclusive_group(required=True)
    destinations.add_argument('--output', type=Path)
    destinations.add_argument('--resume', type=Path)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--models', nargs=2, default=['vehicle.dodge.charger', 'vehicle.lincoln.mkz'])
    p.add_argument('--anchors', nargs='+', type=int, help='optional actual spawn indices')
    p.add_argument('--height-m', type=float, default=8.)
    p.add_argument('--radius-m', type=float, default=18.)
    p.add_argument('--angle-step', type=int, default=15)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--take-tick-ownership', action='store_true')
    args = p.parse_args()
    root = (args.resume or args.output).resolve()
    if not args.resume and root.exists():
        p.error('output exists; choose --resume or a new output')
    if args.resume:
        manifest = json.loads((root/'collection.json').read_text(encoding='utf-8'))
        plan = manifest['plan']
        for case in plan['cases']:
            if case['name'] in manifest['completed']:
                verify_case(checked_path(root, manifest['completed'][case['name']]), case, plan['map'])
        if args.dry_run:
            print(json.dumps(dict(completed=len(manifest['completed']), total=len(plan['cases']),
                                  plan=plan), ensure_ascii=False, indent=2))
            return
        base = plan['cases'][0]['config']
        if (args.host, args.port) != (base['host'], base['port']):
            p.error('resume host/port must match the saved collection')
    else:
        base = json.loads((Path(__file__).parent/'configs/two_candidates.json').read_text())
        base.update(host=args.host, port=args.port)
    import carla
    from capture import connect, inventory
    _, world, versions = connect(carla, base)
    catalog = inventory(world, versions)
    library = world.get_blueprint_library()
    eligible = {bp.id for bp in library.filter('vehicle.*') if bp.has_attribute('color')
                and bp.has_attribute('number_of_wheels') and bp.get_attribute('number_of_wheels').as_int() == 4}
    if args.resume:
        if plan['map'] != catalog['map'] or plan['carla_versions'] != versions:
            raise ValueError('resume map/CARLA version differs')
        required = {c['blueprint'] for case in plan['cases'] for c in case['config']['candidates']}
    else:
        points = catalog['spawn_points']
        if args.anchors:
            by_index = {a['index']: a for a in points}
            if len(set(args.anchors)) != len(args.anchors) or set(args.anchors)-by_index.keys():
                raise ValueError('anchors must be distinct actual spawn indices')
            anchors = [by_index[i] for i in args.anchors]
        else:
            anchors = select_anchors(points, count=2)
        plan = build_plan(base, args.models, anchors, args.height_m, args.radius_m, args.angle_step)
        plan.update(map=catalog['map'], carla_versions=versions, anchors=anchors)
        required = set(args.models)
        manifest = dict(status='incomplete', plan=plan, completed={}, attempts=[])
    if required-eligible:
        raise ValueError(f'unavailable requested vehicles: {sorted(required-eligible)}; available: {sorted(eligible)}')
    print(json.dumps({k: v for k, v in plan.items() if k != 'cases'}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if catalog['existing_vehicles']:
        raise RuntimeError('use an empty controlled world; existing vehicles are not removed')
    if world.get_settings().synchronous_mode and not args.take_tick_ownership:
        raise RuntimeError('stop other tick clients before capture')
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    (root/'_configs').mkdir(exist_ok=True)
    manifest.update(status='incomplete')
    manifest.pop('error', None)
    save_json(root/'collection.json', manifest)
    try:
        for case in plan['cases']:
            if case['name'] in manifest['completed']:
                continue
            relative = case['name']
            if checked_path(root, relative).exists():
                relative += '_retry_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            destination = checked_path(root, relative)
            cfg_path = root/'_configs'/(case['name']+'.json')
            save_json(cfg_path, case['config'])
            manifest['attempts'].append(dict(case=case['name'], directory=relative))
            save_json(root/'collection.json', manifest)
            cmd = [sys.executable, '-u', str(Path(__file__).with_name('capture_scene.py')),
                   '--config', str(cfg_path), '--output', str(destination)]
            if args.take_tick_ownership:
                cmd.append('--take-tick-ownership')
            print(f'[{len(manifest["completed"])+1}/{len(plan["cases"])}] {case["name"]}', flush=True)
            subprocess.run(cmd, check=True)
            verify_case(destination, case, plan['map'])
            manifest['completed'][case['name']] = relative
            save_json(root/'collection.json', manifest)
            write_index(root, manifest)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['error'] = str(error)
        raise
    finally:
        save_json(root/'collection.json', manifest)
        write_index(root, manifest)
    print('Preview: '+str(root/'index.html'), flush=True)


if __name__ == '__main__':
    main()
