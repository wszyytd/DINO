"""Read recorded multi-view captures without exposing future pixels to policies."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .core import checked_path
from .local_features import FeatureCache
from .local_pairs import sha256
from .evidence_state import Action, Observation, ReplayState


@dataclass(frozen=True)
class Episode:
    id: str
    group: str
    position: str
    reference_hash: str
    candidate_ids: tuple
    target: int
    initial: ReplayState
    actions: tuple
    outcomes: dict
    identities: tuple = ()
    source_split: str = ''
    capture_group: str = ''


def relative_pose(camera, target):
    """CARLA world coordinates -> vehicle coordinates, including pitch/roll."""
    camera, target = np.asarray(camera, float), np.asarray(target, float)
    if camera.shape != (6,) or target.shape != (6,) or not np.isfinite(
            np.concatenate((camera, target))).all():
        raise ValueError('camera and target need six finite pose values')
    pitch, yaw, roll = np.radians(target[3:])
    cp, sp, cy, sy, cr, sr = (math.cos(pitch), math.sin(pitch), math.cos(yaw),
                            math.sin(yaw), math.cos(roll), math.sin(roll))
    rotation = np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                         [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                         [sp, -cp*sr, cp*cr]])
    x, y, z = rotation.T @ (camera[:3] - target[:3])
    radius = float(np.linalg.norm([x, y, z]))
    if radius <= 1e-6:
        raise ValueError('camera coincides with candidate')
    azimuth, elevation = math.atan2(y, x), math.atan2(z, math.hypot(x, y))
    return (math.sin(azimuth), math.cos(azimuth), math.sin(elevation),
            math.cos(elevation), math.log(radius / 10.))


def split_by_position(episodes, mapping):
    assigned = set()
    result = {}
    for role, positions in mapping.items():
        positions = {str(p) for p in positions}
        if assigned & positions:
            raise ValueError('position overlap between data roles')
        assigned |= positions
        result[role] = [e for e in episodes if e.position in positions]
    hash_roles = {}
    for role, values in result.items():
        for episode in values:
            hashes = {episode.reference_hash}
            hashes.update(o.image_hash for h in episode.initial.histories for o in h)
            hashes.update(o.image_hash for o in episode.outcomes.values() if o.valid)
            for digest in hashes - {None}:
                if digest in hash_roles and hash_roles[digest] != role:
                    raise ValueError('image overlap between data roles')
                hash_roles[digest] = role
    return result


def _read_observation(root, row, cache):
    pose = relative_pose(json.loads(row['camera_pose']), json.loads(row['target_pose']))
    if row['status'] != 'captured':
        return Observation(row['sample_id'], row['identity'], None, pose, valid=False,
                           reason=row.get('reason', 'rejected'))
    path = checked_path(root, row['crop'])
    digest = sha256(path)
    with Image.open(path) as image:
        gray = np.asarray(image.convert('L'), dtype=np.float32)
        if gray.size == 0 or float((gray < 5).mean()) > .98:
            return Observation(row['sample_id'], row['identity'], None, pose, valid=False,
                               source=str(path), reason='blank_crop')
        blur = float(np.mean(np.diff(gray, axis=0) ** 2)) if gray.shape[0] > 1 else 0.
        quality = (1., math.log1p(gray.size) / 15., math.log1p(blur) / 10.)
    if digest not in cache.images:
        raise ValueError(f'feature cache missing captured image: {path}')
    return Observation(row['sample_id'], row['identity'], digest, pose, quality,
                       source=str(path))


def load_episodes(root, cache_root):
    """Return complete captures and an audit. Failed rows remain real action results."""
    root = Path(root)
    cache = FeatureCache(cache_root)
    episodes, skipped, sources = [], [], {}
    for metadata in sorted(root.glob('*/metadata.csv')):
        scene = metadata.parent
        run_path = scene / 'run.json'
        if not run_path.exists():
            raise ValueError(f'missing run metadata: {scene}')
        run = json.loads(run_path.read_text(encoding='utf-8-sig'))
        if run.get('status') != 'complete' or run.get('cleanup_errors'):
            skipped.append({'scene': scene.name, 'reason': 'incomplete_or_cleanup_failed'})
            continue
        with metadata.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        refs = [r for r in rows if r['role'] == 'reference']
        queries = [r for r in rows if r['role'] == 'query']
        if len(refs) != 1 or not queries:
            raise ValueError(f'invalid reference/query records: {scene}')
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError(f'duplicate sample identifiers: {scene}')
        reference = _read_observation(scene, refs[0], cache)
        if not reference.valid:
            raise ValueError(f'invalid reference: {scene}')
        candidates = tuple(sorted({r['identity'] for r in queries}))
        if refs[0]['identity'] not in candidates:
            raise ValueError('reference target absent from candidate set')
        records = [(_read_observation(scene, r, cache), r) for r in queries]
        initial, actions, outcomes = [], [], {}
        for candidate in candidates:
            ordered = sorted([(o, r) for o, r in records if o.candidate_id == candidate],
                             key=lambda pair: int(pair[1]['sequence']))
            first, _ = ordered[0]
            if not first.valid:
                raise ValueError('pilot needs a valid initial observation for every candidate')
            initial.append((first,))
            for obs, row in ordered[1:]:
                action = Action(obs.id, candidates.index(candidate), obs.pose, row['view'])
                actions.append(action)
                outcomes[action.id] = obs
        if any(reference.image_hash == o.image_hash for o, _ in records if o.valid):
            raise ValueError('reference is an exact replay of a candidate observation')
        position = str(refs[0]['position'])
        identities = tuple(sorted({r.get('blueprint', '') for r in queries}))
        episode = Episode(scene.name, f'position:{position}', position, reference.image_hash,
                          candidates, candidates.index(refs[0]['identity']),
                          ReplayState(candidates, tuple(initial)),
                          tuple(sorted(actions, key=lambda a: a.id)), outcomes, identities,
                          root.name if root.name in ('train', 'val', 'test') else '',
                          scene.name.rsplit('_ref', 1)[0])
        episodes.append(episode)
        sources[str(metadata)] = sha256(metadata)
        sources[str(run_path)] = sha256(run_path)
    if not episodes:
        raise ValueError('no complete replay episodes')
    audit = {'episodes': len(episodes), 'skipped': skipped, 'source_hashes': sources,
             'cache_manifest_sha256': sha256(Path(cache_root) / 'cache.json'),
             'feature_config': cache.config,
             'scope': 'pilot_location_split_identity_overlap_same_background_reference',
             'warning': 'This dataset does not establish identity-disjoint generalization, '
                        'cross-background reference matching, or calibrated stopping.'}
    return episodes, audit


class FeatureBank:
    """Verified frozen tensors, optionally spatially pooled for CPU experiments."""
    def __init__(self, cache_root, digests, grid=8):
        import torch
        from torch.nn import functional as F
        from .evidence_model import teacher_targets
        self.cache = FeatureCache(cache_root)
        self.config = dict(self.cache.config, pooled_grid=grid)
        self.items = {}
        for digest in sorted(set(digests)):
            item = self.cache.get(digest)
            tokens = torch.from_numpy(item['tokens']).float()
            valid = torch.from_numpy(item['valid']).bool()
            cls = torch.from_numpy(item['cls']).float()
            target, target_valid = teacher_targets(tokens[None], valid[None], cls[None])
            side = math.isqrt(tokens.shape[1])
            if grid < 2 or grid > side or side % grid:
                raise ValueError('pooling grid must divide native patch grid and be >=2')
            if grid != side:
                masks = valid.reshape(1, 1, side, side).float()
                coverage = F.adaptive_avg_pool2d(masks, grid)
                clean = tokens.masked_fill(~valid[None, :, None], 0)
                spatial = clean.permute(0, 2, 1).reshape(tokens.shape[0], -1, side, side)
                spatial = F.adaptive_avg_pool2d(spatial, grid) / coverage.clamp_min(1e-6)
                tokens = spatial.flatten(2).transpose(1, 2).contiguous()
                valid = coverage.flatten() > 0
            self.items[digest] = {'tokens': tokens, 'valid': valid, 'cls': cls,
                                  'teacher': target[0], 'teacher_valid': target_valid[0]}

    def get(self, digest):
        return self.items[digest]


def episode_hashes(episodes):
    result = set()
    for episode in episodes:
        result.add(episode.reference_hash)
        result.update(o.image_hash for h in episode.initial.histories for o in h)
        result.update(o.image_hash for o in episode.outcomes.values() if o.valid)
    return result - {None}
