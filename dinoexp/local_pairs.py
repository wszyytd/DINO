"""Portable identity-pair manifests and conservative experiment metrics (no torch)."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import checked_path
from .dataset import load_dataset
from .parts import discover_scenes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(data_root, dataset, group):
    """All conditions within a capture share one user-declared independence group."""
    data_root = Path(data_root).resolve()
    root = checked_path(data_root, dataset)
    if not group or not str(group).strip():
        raise ValueError('a nonempty capture group is required')
    records, sources = [], {}
    for scene in discover_scenes(root):
        scene_name, scene_root = scene['name'], scene['root']
        ref, queries, _ = load_dataset(scene_root)
        sources[(scene_root / 'metadata.csv').relative_to(data_root).as_posix()] = sha256(
            scene_root / 'metadata.csv')
        ref_path = checked_path(scene_root, ref['crop'])
        for row in queries:
            q = checked_path(scene_root, row['crop'])
            view = row.get('view') or str(row.get('sequence', 'unknown'))
            condition = {k: row.get(k, '') for k in
                         ('position', 'weather', 'sequence', 'azimuth', 'elevation', 'distance')}
            records.append({
                'reference': ref_path.relative_to(data_root).as_posix(),
                'query': q.relative_to(data_root).as_posix(),
                'reference_sha256': sha256(ref_path), 'query_sha256': sha256(q),
                'label': int(row['identity'] == ref['identity']),
                'scene': scene_name, 'candidate': row['identity'],
                'sample_id': row['sample_id'], 'view': view,
                'condition': condition,
                'group': str(group),
                'comparison_group': f'{dataset}/{scene_name}/{view}/'
                                    + json.dumps(condition, sort_keys=True, separators=(',', ':')),
            })
    manifest = {'format_version': 1, 'task': 'reference_candidate_identity_pair',
                'dataset': str(dataset), 'source_metadata': sources, 'pairs': records,
                'warning': 'Pilot pairs do not establish cross-scene generalization or calibration.'}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get('format_version') != 1 or not manifest.get('pairs'):
        raise ValueError('need a nonempty version 1 pair manifest')
    required = ('reference', 'query', 'reference_sha256', 'query_sha256',
                'group', 'comparison_group', 'view', 'sample_id')
    seen = set()
    for row in manifest['pairs']:
        if any(not isinstance(row.get(k), str) or not row[k] for k in required):
            raise ValueError('pair fields missing or invalid')
        if type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise ValueError('labels must be confirmed 0 or 1; unknowns cannot be negatives')
        for k in ('reference_sha256', 'query_sha256'):
            if len(row[k]) != 64 or any(c not in '0123456789abcdef' for c in row[k]):
                raise ValueError('invalid image hash')
        key = (row['reference_sha256'], row['query_sha256'])
        if key in seen:
            raise ValueError('duplicate reference/query image pair')
        seen.add(key)


def read_manifest(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_manifest(value)
    return value


def verify_images(manifest, data_root):
    validate_manifest(manifest)
    checked = set()
    for row in manifest['pairs']:
        for role in ('reference', 'query'):
            rel, expected = row[role], row[role + '_sha256']
            if (rel, expected) not in checked:
                if sha256(checked_path(data_root, rel)) != expected:
                    raise ValueError(f'image hash mismatch: {rel}')
                checked.add((rel, expected))


def check_split(train, evaluation):
    if {r['group'] for r in train} & {r['group'] for r in evaluation}:
        raise ValueError('capture group overlap across train/evaluation splits')
    def images(rows):
        return {r[k] for r in rows for k in ('reference_sha256', 'query_sha256')}
    if images(train) & images(evaluation):
        raise ValueError('image overlap across train/evaluation splits (including references)')


def ranking_metrics(rows, scores):
    """Top-1 only where both labels occur; ties never count as a correct ranking."""
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('invalid scores')
    groups = {}
    for row, score in zip(rows, scores, strict=True):
        groups.setdefault(row['comparison_group'], []).append((row, float(score)))
    margins, absent, no_negative = [], 0, 0
    for values in groups.values():
        positive = [s for r, s in values if r['label'] == 1]
        negative = [s for r, s in values if r['label'] == 0]
        if not positive:
            absent += 1
        elif not negative:
            no_negative += 1
        else:
            margins.append(max(positive) - max(negative))
    result = {'pairs': len(rows), 'ranking_groups': len(margins),
              'target_top1': float(np.mean(np.array(margins) > 0)) if margins else None,
              'mean_margin': float(np.mean(margins)) if margins else None,
              'target_absent_groups': absent, 'groups_without_negative': no_negative}
    result['by_view'] = {}
    for view in sorted({r['view'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['view'] == view]
        # Avoid recursive view aggregation.
        subset_groups = {}
        for i in indices:
            subset_groups.setdefault(rows[i]['comparison_group'], []).append(i)
        vm = []
        for indices_in_group in subset_groups.values():
            pos = [scores[i] for i in indices_in_group if rows[i]['label'] == 1]
            neg = [scores[i] for i in indices_in_group if rows[i]['label'] == 0]
            if pos and neg:
                vm.append(max(pos) - max(neg))
        result['by_view'][view] = {'ranking_groups': len(vm),
                                  'target_top1': float(np.mean(np.array(vm) > 0)) if vm else None}
    return result
