"""Collect about 1000 CARLA vehicle crops with varied references and held-out assets."""

import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import subprocess
import sys

from dinoexp.dataset import load_dataset
from dinoexp.core import checked_path
from dinoexp.local_pairs import build_manifest, check_split
from dinoexp.more_photos import (AUTO_MIN_ANCHOR_DISTANCE, build_plan, choose_models,
                                select_anchors)
from dinoexp.more_resume import check_resume_environment, prepare_resume, resume_summary
from dinoexp.more_placement import (apply_camera_radius, apply_placement, placement_options,
                                  probe_placement)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def capture_plan(plan, output, runner=subprocess.run, take_tick=False, *, resume=False, pilot=False,
                 test_placement=None, test_camera_radius=None):
    root = Path(output).resolve()
    if resume:
        manifest, completed, _ = prepare_resume(root, pilot)
        if test_placement is not None:
            manifest = apply_placement(manifest, completed, test_placement)
        if test_camera_radius is not None:
            manifest = apply_camera_radius(manifest, completed, test_camera_radius)
        plan = manifest['plan']
        if manifest.get('error'):
            manifest.setdefault('previous_errors', []).append(manifest.pop('error'))
    else:
        if pilot:
            raise ValueError('--pilot requires an existing --resume collection')
        if root.exists():
            raise ValueError('output already exists; keep old data and use a new run directory')
        root.mkdir(parents=True, exist_ok=False)
        (root / '_configs').mkdir()
        manifest, completed = dict(plan=plan), set()
    manifest.update(status='incomplete', completed_scenes=len(completed))
    suites = {s: dict(status='incomplete', conditions=[],
                     note='Fresh reference per scene; no shared global reference')
              for s in ('train', 'val', 'test')}
    preview_links = []
    def add_completed(case):
        split, name = case['split'], case['name']
        suites[split]['conditions'].append(dict(name=name, status='complete', config=case['config'],
                                                capture_group=case['capture_group']))
        preview_links.append(f'<li><a href="{split}/{name}/preview.html">'
                             f'{html.escape(name)}</a></li>')
    def save_progress():
        for split in suites:
            save_json(root / split / 'suite.json', suites[split])
        manifest['completed_scenes'] = len(completed)
        save_json(root / 'collection.json', manifest)
        scope = ('小规模试验：保留已完成采集，缩减验证和测试条件。'
                 if manifest.get('scope') == 'reduced_pilot' else '')
        (root / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
            '<h1>新增车辆照片</h1><p>' + scope
            + '检查真实外观、清晰度和遮挡，真值框不代表部位可见。</p><ul>'
            + ''.join(preview_links) + '</ul>', encoding='utf-8')
    for case in plan['cases']:
        if case['name'] in completed:
            add_completed(case)
    for split in suites:
        (root / split).mkdir(exist_ok=True)
    save_progress()
    try:
        for case in plan['cases']:
            split, name, cfg = case['split'], case['name'], case['config']
            if name in completed:
                continue
            config = root / '_configs' / (name + '.json')
            save_json(config, cfg)
            destination = checked_path(root, f'{split}/{name}')
            if destination.exists():
                # Preserve every partial attempt. Resolve both targets inside the
                # explicitly selected collection before moving on Windows/Linux.
                archive_root = checked_path(root, '_failed')
                archive_root.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                archive = checked_path(root, f'_failed/{split}_{name}_{stamp}')
                destination.rename(archive)
                manifest.setdefault('archived_attempts', []).append(str(archive.relative_to(root)))
                save_json(root / 'collection.json', manifest)
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
            completed.add(name)
            add_completed(case)
            save_progress()
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
    destination = p.add_mutually_exclusive_group()
    destination.add_argument('--output', type=Path, default=Path('data') / (
        'more_photos_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    destination.add_argument('--resume', type=Path, help='verify and continue a saved collection')
    p.add_argument('--pilot', action='store_true', help='with --resume: retain completed groups, '
                   'add only the first missing A/B reference group per split')
    p.add_argument('--retry-test-placement', action='store_true', help='with --resume --pilot: '
                   'probe reserved test positions and spacing before capturing either reference')
    p.add_argument('--test-camera-radius', type=float, help='with --resume --pilot: raise pending '
                   'test query elevations to cap horizontal offset in metres; retain current placement')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--models', nargs='+', help='optional six to eight IDs, checked against inventory')
    p.add_argument('--min-anchor-distance', type=float,
                   help='strict minimum spacing in metres; default auto maximizes spacing '
                        'with a 40m acquisition floor')
    p.add_argument('--dry-run', action='store_true', help='read CARLA inventory, print plan, no capture')
    p.add_argument('--take-tick-ownership', action='store_true')
    args = p.parse_args()
    if args.pilot and not args.resume:
        raise ValueError('--pilot requires --resume')
    if args.retry_test_placement and not (args.resume and args.pilot):
        raise ValueError('--retry-test-placement requires --resume --pilot')
    if args.test_camera_radius is not None:
        if not (args.resume and args.pilot):
            raise ValueError('--test-camera-radius requires --resume --pilot')
        if args.retry_test_placement:
            raise ValueError('adjust cameras at the existing placement; omit --retry-test-placement')
    if args.resume:
        saved, completed, remaining = prepare_resume(args.resume, args.pilot)
        if args.test_camera_radius is not None:
            saved = apply_camera_radius(saved, completed, args.test_camera_radius)
        options = placement_options(saved, completed) if args.retry_test_placement else None
        if args.dry_run:
            summary = resume_summary(saved, completed, remaining)
            if options is not None:
                summary['test_placement_options'] = options
                summary['placement_warning'] = 'Live two-vehicle probe required; no position confirmed yet.'
            if args.test_camera_radius is not None:
                summary['test_view_scope'] = saved['plan'].get('test_view_scope')
                summary['test_query_views'] = next(c['config']['views']
                    for c in saved['plan']['cases'] if c['split'] == 'test')
            print(json.dumps(summary, indent=2))
            return
        plan = saved['plan']
        if not remaining:
            capture_plan(None, args.resume, resume=True, pilot=args.pilot)
            return
        base = dict(plan['cases'][0]['config'])
    else:
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
    if args.resume:
        check_resume_environment(plan, catalog, eligible)
        # Host/port locate the current server; saved case configs stay unchanged.
        for case in remaining:
            if case['config']['host'] != args.host or case['config']['port'] != args.port:
                raise ValueError('resume host/port must match the saved capture plan')
    else:
        models = choose_models(eligible, args.models)
        anchors = select_anchors(catalog['spawn_points'], minimum_distance=args.min_anchor_distance)
        plan = build_plan(base, models, anchors, args.seed)
        plan['map'], plan['carla_versions'] = catalog['map'], versions
        plan['anchor_selection'] = dict(
            mode='auto' if args.min_anchor_distance is None else 'explicit',
            method='deterministic_multi_start_farthest_point',
            requested_min_distance_m=args.min_anchor_distance,
            auto_min_distance_m=AUTO_MIN_ANCHOR_DISTANCE if args.min_anchor_distance is None else None,
            actual_min_distance_m=min(math.dist(a['pose'][:2], b['pose'][:2])
                                      for i, a in enumerate(anchors) for b in anchors[i + 1:]),
            available_spawn_points=len(catalog['spawn_points']), selected_spawn_points=anchors,
            note='Distinct capture positions within one map; camera footprints/backgrounds may overlap.')
    print(json.dumps({k: v for k, v in plan.items() if k != 'cases'}, indent=2), flush=True)
    if args.dry_run:
        print('Read-only plan. No photos captured; no output directory created.')
        return
    if catalog['existing_vehicles']:
        raise RuntimeError('Use an empty controlled scene; existing vehicles will not be removed')
    if world.get_settings().synchronous_mode and not args.take_tick_ownership:
        raise RuntimeError('World already synchronous; stop the other tick client first')
    placement = (probe_placement(carla, world, saved, completed, catalog,
                                 take_tick=args.take_tick_ownership)
                 if args.retry_test_placement else None)
    capture_plan(plan, args.resume or args.output, take_tick=args.take_tick_ownership,
                 resume=bool(args.resume), pilot=args.pilot, test_placement=placement,
                 test_camera_radius=args.test_camera_radius)


if __name__ == '__main__':
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
