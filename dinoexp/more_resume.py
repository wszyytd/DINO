"""Read-only recovery of completed capture cases and explicit smaller pilot plans."""

from copy import deepcopy
import json
from pathlib import Path
import re

from .core import checked_path
from .dataset import load_dataset


SPLITS = ('train', 'val', 'test')


def prepare_resume(root, pilot=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'collection.json').read_text(encoding='utf-8-sig'))
    plan = manifest['plan']
    cases = plan['cases']
    seen, completed = set(), set()
    for case in cases:
        split, name = case['split'], case['name']
        if split not in SPLITS or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen:
            raise ValueError('invalid or duplicate saved capture case')
        seen.add(name)
        path = checked_path(root, f'{split}/{name}')
        marker = path / 'run.json'
        if not marker.is_file():
            continue
        try:
            run = json.loads(marker.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            # The child may be interrupted during its final marker write. An
            # uncheckpointed attempt stays pending; suite claims are checked
            # below so corruption of a recorded success still stops recovery.
            continue
        if not isinstance(run, dict):
            continue
        if run.get('status') != 'complete' or run.get('cleanup_errors'):
            continue
        if run.get('config') != case['config'] or run.get('map') != plan['map']:
            raise ValueError(f'{name}: completed scene config/map differs from saved plan')
        ref, queries, rejected = load_dataset(path)
        expected = {(c['identity'], v['name']) for c in case['config']['candidates']
                    for v in case['config']['views']}
        if (ref['identity'] != case['config']['target_identity'] or rejected
                or len(queries) != len(expected)
                or {(q['identity'], q['view']) for q in queries} != expected):
            raise ValueError(f'{name}: completed scene has incomplete or inconsistent observations')
        completed.add(name)
    # Never silently discard a scene the old controller said was completed.
    for split in SPLITS:
        suite = json.loads((root / split / 'suite.json').read_text(encoding='utf-8-sig'))
        for item in suite['conditions']:
            if item['name'] in seen and item['name'] not in completed:
                raise ValueError(f'{item["name"]}: previously completed scene is now invalid')
    if pilot:
        groups = {}
        for case in cases:
            groups.setdefault((case['split'], case['capture_group']), []).append(case)
        chosen = {key for key, members in groups.items()
                  if any(c['name'] in completed for c in members)}
        for split in SPLITS:
            if not any(key[0] == split for key in chosen):
                key = next((key for key in groups if key[0] == split), None)
                if key is None:
                    raise ValueError(f'saved plan has no {split} reference group')
                chosen.add(key)
        for key in chosen:
            members = groups[key]
            if len(members) != 2 or {c['config']['target_identity'] for c in members} != {'A', 'B'}:
                raise ValueError('pilot needs complete A/B reference groups')
        manifest.setdefault('original_plan', deepcopy(plan))
        plan = deepcopy(plan)
        plan['cases'] = [c for c in cases if (c['split'], c['capture_group']) in chosen]
        plan['split_anchors'] = {s: sorted({c['config']['anchor_spawn_index']
            for c in plan['cases'] if c['split'] == s}) for s in SPLITS}
        plan['expected_query_crops'] = sum(len(c['config']['views']) * 2 for c in plan['cases'])
        plan['expected_reference_crops'] = len(plan['cases'])
        if manifest.get('scope') != 'reduced_pilot':
            plan['note'] += (' Reduced pilot: retain verified completed groups and at least one '
                             'A/B reference group per split; fewer held-out locations/conditions.')
        plan['split_models'] = {s: [model for model in plan['split_models'][s]
            if any(v['blueprint'] == model for c in plan['cases'] if c['split'] == s
                   for v in c['config']['candidates'])] for s in SPLITS}
        manifest['plan'], manifest['scope'] = plan, 'reduced_pilot'
    models = {s: {v['blueprint'] for c in plan['cases'] if c['split'] == s
                   for v in c['config']['candidates']} for s in SPLITS}
    anchors = {s: {c['config']['anchor_spawn_index'] for c in plan['cases'] if c['split'] == s}
               for s in SPLITS}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if models[a] & models[b] or anchors[a] & anchors[b]:
            raise ValueError('saved plan overlaps models or locations across splits')
    remaining = [c for c in plan['cases'] if c['name'] not in completed]
    return manifest, completed, remaining


def resume_summary(manifest, completed, remaining):
    plan = manifest['plan']
    return dict(scope=manifest.get('scope', 'full_plan'),
                completed_scenes=len(completed), remaining_scenes=len(remaining),
                planned_scenes=len(plan['cases']),
                expected_query_crops=plan['expected_query_crops'],
                expected_reference_crops=plan['expected_reference_crops'],
                next_cases=[c['name'] for c in remaining],
                split_models=plan['split_models'], split_anchors=plan['split_anchors'],
                warning='Read-only preview; no files changed. Reduced pilots have limited coverage.')


def check_resume_environment(plan, catalog, eligible):
    if catalog['map'] != plan['map']:
        raise ValueError(f'resume requires original map {plan["map"]}; current {catalog["map"]}')
    required = {v['blueprint'] for c in plan['cases'] for v in c['config']['candidates']}
    if required - set(eligible):
        raise ValueError(f'resume assets unavailable: {sorted(required - set(eligible))}')
    recorded = plan.get('anchor_selection', {}).get('selected_spawn_points', [])
    actual = {p['index']: p['pose'] for p in catalog['spawn_points']}
    for point in recorded:
        current = actual.get(point['index'])
        if current is None or any(abs(a - b) > 0.1 for a, b in zip(point['pose'], current, strict=True)):
            raise ValueError('resume spawn-point layout differs from original capture')
