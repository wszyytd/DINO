"""Source-backed, human-reviewed part evidence. No probabilities or auto-exclusion."""

import copy
import hashlib
import json
from collections import defaultdict

from .core import checked_path


CHECKS = ('reference_correct', 'query_correct', 'same_part_side',
          'identity_continuity', 'clear_enough', 'appearance_stable')
STATES = ('pending', 'uncertain', 'compatible', 'contradictory')
ASSETS = ('source', 'overlay', 'crop', 'masked_crop', 'mask')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def describe_sample(root, sample, part):
    matches = [r for r in sample['records']
               if r['status'] == 'detected' and r['class_name'] == part]
    # Keep the full crop/overlay even when there is no usable part prediction.
    record = sample['records'][0]
    result = dict(sample_id=sample['sample_id'], metadata=sample['capture_metadata'],
                  source=record['source'], overlay=record['overlay'], instances=[])
    for match in matches:
        instance = {k: match.get(k) for k in ASSETS}
        instance.update(confidence=match['confidence'], bbox_xyxy=match['bbox_xyxy'],
                        native_size=[match['native_width'], match['native_height']],
                        roi_size=[match['crop_width'], match['crop_height']],
                        mask_pixels=match['mask_pixels'])
        result['instances'].append(instance)
    paths = {result['source'], result['overlay']}
    for instance in result['instances']:
        paths.update(instance[k] for k in ASSETS if instance.get(k))
    result['asset_sha256'] = {
        path: hashlib.sha256(checked_path(root, path).read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return result


def build_cards(root, parts=None):
    """Pair exact class labels; a unique model label is only a review proposal."""
    parts = parts or ['right_front-door', 'right_back-door']
    if len(set(parts)) != len(parts) or not all(isinstance(p, str) and p for p in parts):
        raise ValueError('parts must be unique nonempty exact class names')
    run = json.loads((root / 'parts_run.json').read_text(encoding='utf-8-sig'))
    if run.get('status') != 'complete':
        raise ValueError('parts run must be complete')
    samples = json.loads((root / 'parts.json').read_text(encoding='utf-8-sig'))
    scenes = defaultdict(list)
    seen = set()
    for sample in samples:
        key = (sample['scene'], sample['sample_id'])
        if key in seen or not sample['records'] or sample['role'] not in ('reference', 'query'):
            raise ValueError('invalid or duplicate parts sample')
        seen.add(key)
        scenes[sample['scene']].append(sample)
    if not scenes:
        raise ValueError('parts run has no samples')
    cards = []
    for scene, entries in sorted(scenes.items()):
        references = [s for s in entries if s['role'] == 'reference']
        if len(references) != 1:
            raise ValueError('each scene needs exactly one reference')
        queries = [s for s in entries if s['role'] == 'query']
        if not queries:
            raise ValueError('each scene needs queries')
        for query in queries:
            for part in parts:
                ref = describe_sample(root, references[0], part)
                obs = describe_sample(root, query, part)
                availability = 'needs_review'
                for label, sample in [('reference', ref), ('query', obs)]:
                    count = len(sample['instances'])
                    if count != 1:
                        availability = label + ('_missing' if count == 0 else '_ambiguous')
                        break
                    if min(sample['instances'][0]['roi_size']) < 2:
                        availability = label + '_roi_too_small'
                        break
                metadata = query['capture_metadata']
                card = dict(card_id=digest([scene, query['sample_id'], part])[:24],
                            scene=scene, candidate=query['identity'], part=part,
                            sample_id=query['sample_id'], sequence=int(metadata.get('sequence') or 0),
                            view=metadata.get('view', ''), frame=metadata.get('frame', ''),
                            reference=ref, observation=obs, availability=availability,
                            status='unknown', cosine=None, review=None)
                # Bind reviews to actual pixels, geometry, part and detection output, not scores.
                card['fingerprint'] = digest(card)
                cards.append(card)
    return sorted(cards, key=lambda c: (c['scene'], c['candidate'], c['sequence'], c['part']))


def review_template(cards):
    return dict(schema_version=1, reviews=[dict(
        card_id=c['card_id'], fingerprint=c['fingerprint'], status='pending',
        reviewer='', basis='', checks={key: False for key in CHECKS}
    ) for c in cards])


def apply_reviews(cards, document):
    result = copy.deepcopy(cards)
    by_id = {c['card_id']: c for c in result}
    seen = set()
    for review in document['reviews']:
        key = review['card_id']
        if key in seen or key not in by_id:
            raise ValueError('duplicate or unknown review card_id')
        seen.add(key)
        card = by_id[key]
        if review['fingerprint'] != card['fingerprint']:
            raise ValueError('review fingerprint does not match this observation')
        status = review['status']
        if status not in STATES:
            raise ValueError('invalid review status')
        if status == 'pending':
            continue
        if not str(review.get('reviewer', '')).strip() or not str(review.get('basis', '')).strip():
            raise ValueError('reviewer and basis are required')
        if status in ('compatible', 'contradictory'):
            if card['availability'] != 'needs_review':
                raise ValueError('unavailable part cannot become evidence without a new localization run')
            if not all(review.get('checks', {}).get(k) is True for k in CHECKS):
                raise ValueError('all correspondence/quality checks must be true')
        card.update(status=status, review=copy.deepcopy(review))
    return result


def fuse_cards(cards):
    """Set-based history: repetitions never add votes; unresolved contradictions stay explicit."""
    candidates = defaultdict(list)
    for card in cards:
        candidates[(card['scene'], card['candidate'])].append(card)
    result = []
    for (scene, candidate), history in sorted(candidates.items()):
        by_part = defaultdict(list)
        for card in history:
            by_part[card['part']].append(card)
        states = {}
        for part, entries in sorted(by_part.items()):
            reviewed = {c['status'] for c in entries} - {'unknown', 'uncertain'}
            if {'compatible', 'contradictory'} <= reviewed:
                status = 'conflict'
            elif reviewed:
                status = next(iter(reviewed))
            else:
                status = 'uncertain' if any(c['status'] == 'uncertain' for c in entries) else 'unknown'
            states[part] = dict(status=status,
                               observation_groups=len({c['view'] or c['sample_id'] for c in entries}),
                               card_ids=list(dict.fromkeys(c['card_id'] for c in entries)))
        values = {s['status'] for s in states.values()}
        state = ('conflict' if 'conflict' in values else
                 'contradiction_recorded' if 'contradictory' in values else
                 'compatible_observed' if 'compatible' in values else 'insufficient')
        action = ('resolve_conflict' if state == 'conflict' else
                  'review_contradiction' if state == 'contradiction_recorded' else
                  'review_localization' if any(c['status'] == 'unknown' and
                  c['availability'] == 'needs_review' for c in history) else
                  'acquire_clear_corresponding_part')
        result.append(dict(scene=scene, candidate=candidate, state=state, parts=states,
                           decision='retain', suggested_next_step=action,
                           reason='pilot: no calibrated exclusion rule or identity probability'))
    return result
