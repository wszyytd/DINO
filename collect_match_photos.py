"""Collect about 1000 CARLA vehicle crops with varied references and held-out assets."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys

from dinoexp.dataset import load_dataset
from dinoexp.local_pairs import build_manifest, check_split
from dinoexp.more_photos import build_plan, choose_models, select_anchors


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def capture_plan(plan, output, runner=subprocess.run, take_tick=False):
    root = Path(output).resolve()
    if root.exists():
        raise ValueError('output already exists; keep old data and use a new run directory')
    root.mkdir(parents=True, exist_ok=False)
    (root / '_configs').mkdir()
    manifest = dict(status='incomplete', completed_scenes=0, plan=plan)
    suites = {s: dict(status='incomplete', conditions=[],
                     note='Fresh reference per scene; no shared global reference')
              for s in ('train', 'val', 'test')}
    for split in suites:
        (root / split).mkdir()
        save_json(root / split / 'suite.json', suites[split])
    save_json(root / 'collection.json', manifest)
    preview_links = []
    try:
        for case in plan['cases']:
            split, name, cfg = case['split'], case['name'], case['config']
            config = root / '_configs' / (name + '.json')
            save_json(config, cfg)
            destination = root / split / name
            command = [sys.executable, '-u', str(Path(__file__).with_name('capture_scene.py')),
                       '--config', str(config), '--output', str(destination)]
            if take_tick:
                command.append('--take-tick-ownership')
            print(f'[{manifest["completed_scenes"] + 1}/{len(plan["cases"])}] '
                  f'{name}: reference {cfg["target_identity"]}', flush=True)
            runner(command, check=True)
            _, queries, rejected = load_dataset(destination)
            if rejected or len(queries) != len(cfg['views']) * 2:
                raise ValueError(f'{name}: incomplete or rejected observations; inspect saved photos')
            suites[split]['conditions'].append(dict(name=name, status='complete', config=cfg,
                                                    capture_group=case['capture_group']))
            save_json(root / split / 'suite.json', suites[split])
            manifest['completed_scenes'] += 1
            save_json(root / 'collection.json', manifest)
            preview_links.append(f'<li><a href="{split}/{name}/preview.html">'
                                 f'{html.escape(name)}</a></li>')
            (root / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
                '<h1>新增车辆照片</h1><p>检查真实外观、清晰度和遮挡，真值框不代表部位可见。</p><ul>'
                + ''.join(preview_links) + '</ul>', encoding='utf-8')
        for split, suite in suites.items():
            suite['status'] = 'complete'
            save_json(root / split / 'suite.json', suite)
        pairs = {s: build_manifest(root, s, 'expanded-photos-' + s) for s in suites}
        check_split(pairs['train']['pairs'], pairs['val']['pairs'])
        check_split(pairs['train']['pairs'], pairs['test']['pairs'])
        check_split(pairs['val']['pairs'], pairs['test']['pairs'])
        for split, data in pairs.items():
            save_json(root / f'pairs_{split}.json', data)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['error'] = str(error)
        raise
    finally:
        save_json(root / 'collection.json', manifest)
    print(f'Saved {plan["expected_query_crops"]} query crops and '
          f'{plan["expected_reference_crops"]} references: {root}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', type=Path, default=Path(__file__).parent
                   / 'configs/two_candidates.json')
    p.add_argument('--output', type=Path, default=Path('data') / (
        'more_photos_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--models', nargs='+', help='optional six to eight IDs, checked against inventory')
    p.add_argument('--min-anchor-distance', type=float, default=80)
    p.add_argument('--dry-run', action='store_true', help='read CARLA inventory, print plan, no capture')
    p.add_argument('--take-tick-ownership', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output exists; choose a new directory')
    base = json.loads(args.base_config.read_text(encoding='utf-8-sig'))
    base.update(host=args.host, port=args.port)
    try:
        import carla
        from capture import connect, inventory
        _, world, versions = connect(carla, base)
        catalog = inventory(world, versions)
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError('Cannot read CARLA. Activate carla10 and start the simulator first. '
                           'No photos/output directory created. ' + str(error)) from error
    eligible = []
    for bp in world.get_blueprint_library().filter('vehicle.*'):
        if (bp.has_attribute('color') and bp.has_attribute('number_of_wheels')
                and bp.get_attribute('number_of_wheels').as_int() == 4):
            eligible.append(bp.id)
    models = choose_models(eligible, args.models)
    anchors = select_anchors(catalog['spawn_points'], minimum_distance=args.min_anchor_distance)
    plan = build_plan(base, models, anchors, args.seed)
    plan['map'], plan['carla_versions'] = catalog['map'], versions
    print(json.dumps({k: v for k, v in plan.items() if k != 'cases'}, indent=2), flush=True)
    if args.dry_run:
        print('Read-only plan. No photos captured; no output directory created.')
        return
    if catalog['existing_vehicles']:
        raise RuntimeError('Use an empty controlled scene; existing vehicles will not be removed')
    if world.get_settings().synchronous_mode and not args.take_tick_ownership:
        raise RuntimeError('World already synchronous; stop the other tick client first')
    capture_plan(plan, args.output, take_tick=args.take_tick_ownership)


if __name__ == '__main__':
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
