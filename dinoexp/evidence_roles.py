"""Explicit grouped roles and nested training sets; never select on evaluation scores."""
import math
from pathlib import Path
import random

from .evidence_data import load_episodes, episode_hashes


def load_corpus(root, cache):
    root = Path(root)
    if not (root/'train').is_dir():
        return load_episodes(root, cache)
    episodes, audits = [], {}
    for split in ('train', 'val', 'test'):
        values, audit = load_episodes(root/split, cache)
        episodes.extend(values)
        audits[split] = audit
    return episodes, dict(episodes=len(episodes), sources=audits,
                         cache_manifest_sha256=audits['train']['cache_manifest_sha256'],
                         skipped=[dict(item, split=s) for s, a in audits.items()
                                  for item in a['skipped']],
                         scope='explicit_grouped_crop_experiment')


def select_roles(episodes, manifest):
    if manifest.get('format_version') != 1 or set(manifest.get('roles', {})) != {'repr', 'value', 'eval'}:
        raise ValueError('role manifest needs version 1 and repr/value/eval roles')
    by_id = {e.id: e for e in episodes}
    if len(by_id) != len(episodes):
        raise ValueError('duplicate episode IDs in corpus')
    assigned, positions, groups, hashes, result = set(), {}, {}, {}, {}
    all_roles = dict(manifest['roles'], reserved_test=manifest.get('reserved_test', []))
    for role, ids in all_roles.items():
        if not isinstance(ids, list) or (role != 'reserved_test' and not ids):
            raise ValueError('data roles must be nonempty episode lists')
        if len(set(ids)) != len(ids) or assigned & set(ids):
            raise ValueError('episode overlap between data roles')
        if set(ids)-set(by_id):
            raise ValueError(f'unknown episode IDs: {sorted(set(ids)-set(by_id))}')
        assigned.update(ids)
        result[role] = [by_id[i] for i in ids]
        for e in result[role]:
            for value, seen, kind in ((e.position, positions, 'position'),
                                      (e.capture_group or e.id, groups, 'capture group')):
                if value in seen and seen[value] != role:
                    raise ValueError(f'{kind} overlap between roles')
                seen[value] = role
            for digest in episode_hashes([e]):
                if digest in hashes and hashes[digest] != role:
                    raise ValueError('image overlap between roles')
                hashes[digest] = role
    if manifest.get('require_eval_identity_disjoint', False):
        assets = {role: {i for e in values for i in e.identities if i}
                  for role, values in result.items()}
        if any(not e.identities or any(not i for i in e.identities)
               for values in result.values() for e in values):
            raise ValueError('identity metadata is missing')
        fitted = assets['repr'] | assets['value']
        if fitted & (assets['eval'] | assets['reserved_test']) or assets['eval'] & assets['reserved_test']:
            raise ValueError('evaluation/test identity overlap')
    reserved = result.pop('reserved_test')
    return result, dict(reserved_test=[e.id for e in reserved],
                        role_manifest=manifest, scope='explicit_grouped_crop_experiment')


def make_manifests(episodes, value_positions, fractions=(.25, .5, 1.), seed=7):
    if not fractions or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError('fractions must be finite in (0,1]')
    if len(set(fractions)) != len(fractions):
        raise ValueError('fractions must be unique')
    value_positions = {str(p) for p in value_positions}
    train = [e for e in episodes if e.source_split == 'train']
    value = [e.id for e in train if e.position in value_positions]
    pool = [e for e in train if e.position not in value_positions]
    groups = {}
    for e in pool:
        groups.setdefault(e.capture_group or e.id, []).append(e)
    for members in groups.values():
        if len(members) != 2 or {e.target for e in members} != {0, 1}:
            raise ValueError('nested representation sets require complete A/B reference groups')
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    manifests = []
    for fraction in sorted(fractions):
        chosen = order[:max(1, math.ceil(len(order)*fraction))]
        m = dict(format_version=1, seed=seed, fraction=fraction,
                 require_eval_identity_disjoint=True,
                 roles=dict(repr=sorted(e.id for g in chosen for e in groups[g]),
                            value=sorted(value),
                            eval=sorted(e.id for e in episodes if e.source_split == 'val')),
                 reserved_test=sorted(e.id for e in episodes if e.source_split == 'test'),
                 note='Nested whole A/B groups; fixed value and validation sets; test reserved.')
        if not m['reserved_test']:
            raise ValueError('a reserved test split is required')
        select_roles(episodes, m)
        manifests.append(m)
    return manifests
