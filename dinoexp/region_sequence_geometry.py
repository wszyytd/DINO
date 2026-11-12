"""Teacher-seeded mask assistance: fragments/extent/IDs, not autonomous detection."""
import math

import numpy as np

from .region_geometry import clip_box


DEFAULT_HINTS = dict(min_seed_pixels=16, min_seed_fraction=.01, min_visible_pixels=32,
                     min_area_518=196., padding_fraction=.04)


def group_regions(record, masks, cfg=None):
    cfg = dict(DEFAULT_HINTS if cfg is None else cfg)
    if set(cfg) != set(DEFAULT_HINTS) or any(not math.isfinite(v) or v < 0 for v in cfg.values()):
        raise ValueError('invalid mask hint configuration')
    if (min(cfg['min_seed_pixels'], cfg['min_visible_pixels'], cfg['min_area_518']) <= 0
            or not 0 < cfg['min_seed_fraction'] <= 1 or cfg['padding_fraction'] > .5):
        raise ValueError('invalid mask hint thresholds')
    width, height = record['size']
    spatial = np.asarray(record['spatial'])
    if spatial.ndim != 2 or min(spatial.shape) < 1 or not np.isfinite(spatial).all():
        raise ValueError('need teacher spatial grid')
    if any(m.shape != (height, width) or m.dtype != np.bool_ for m in masks.values()):
        raise ValueError('mask must be a boolean full-RGB array')
    components = {c['component_id']: c for c in record['before_merge']}
    gy, gx = spatial.shape
    candidates = record['candidates']
    support = {}
    for candidate in candidates:
        active = np.zeros((gy, gx), bool)
        for cid in candidate['merged_from']:
            if cid not in components:
                raise ValueError('candidate references unknown teacher component')
            for y, x in components[cid]['cells']:
                if not 0 <= y < gy or not 0 <= x < gx:
                    raise ValueError('teacher cell outside heatmap')
                active[y, x] = True
        support[candidate['candidate_id']] = active
    observations, used = [], set()
    for uid, mask in sorted(masks.items()):
        yy, xx = np.where(mask)
        total = len(xx)
        if total == 0:
            continue
        rows = np.minimum(yy*gy//height, gy-1)
        cols = np.minimum(xx*gx//width, gx-1)
        ids, ids_components = [], set()
        union = np.zeros(total, bool)
        for candidate in candidates:
            cid = candidate['candidate_id']
            hit = support[cid][rows, cols]
            # A single component can be a tiny fragment: combine support per car
            # before applying the minimum, rather than discarding each fragment.
            if hit.any():
                ids.append(cid); ids_components.update(candidate['merged_from']); union |= hit
        pixels = int(union.sum())
        if pixels < cfg['min_seed_pixels'] or pixels/total < cfg['min_seed_fraction']:
            continue
        box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
        dx = (box[2]-box[0])*cfg['padding_fraction']
        dy = (box[3]-box[1])*cfg['padding_fraction']
        padded = clip_box([box[0]-dx, box[1]-dy, box[2]+dx, box[3]+dy], (width, height))
        area = total*(518/max(width, height))**2
        writable = total >= cfg['min_visible_pixels'] and area >= cfg['min_area_518']
        observations.append(dict(identity=uid, box=padded, source_candidate_ids=ids,
                                 component_ids=sorted(ids_components), seed_pixels=pixels,
                                 seed_fraction=pixels/total, visible_pixels=total,
                                 effective_area_518=area, coverage=1., mask_extent_used=True,
                                 write_allowed=bool(writable),
                                 rejection=None if writable else 'visible vehicle area too small'))
        used.update(ids)
    split_ids = [c['candidate_id'] for c in candidates
                 if sum(c['candidate_id'] in o['source_candidate_ids'] for o in observations) > 1]
    return dict(observations=observations, raw_candidate_count=len(candidates),
                unassigned_candidate_ids=[c['candidate_id'] for c in candidates if c['candidate_id'] not in used],
                mask_split_candidate_ids=split_ids,
                grouping_assistance='GT masks group teacher-supported fragments, supply visible extent '
                                    'and stable identity; no vehicle without teacher active support is added')
