"""Reference-conditioned proposals and independent supervision. No torch dependency."""
import math
import numpy as np


DEFAULTS = dict(
    proposal=dict(relative_peak=.45, uniform_multiple=2., expansion=1.5,
                  min_side=96, merge_iou=.3, merge_containment=.6, max_candidates=8),
    label=dict(min_coverage=.5, min_pixels=32, min_occupancy=.02,
               min_area_518=196, secondary_pixels=16, secondary_coverage=.1,
               secondary_occupancy=.02, giant_area_fraction=.5),
    training=dict(seeds=[7, 17, 27], epochs=100, lr=1e-4, batch_size=8,
                  weight_decay=.01, heads=4, slots=8),
    format_version=1)


def validate_config(config):
    if set(config) != set(DEFAULTS) or config['format_version'] != 1:
        raise ValueError('expected full region A/B v1 config')
    for section in ('proposal', 'label', 'training'):
        if not isinstance(config[section], dict) or set(config[section]) != set(DEFAULTS[section]):
            raise ValueError('config fields differ: '+section)
    p, label, train = (config[k] for k in ('proposal', 'label', 'training'))
    for section in (p, label, {k: v for k, v in train.items() if k != 'seeds'}):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in section.values()):
            raise ValueError('finite nonnegative numeric settings required')
    if (not 0 < p['relative_peak'] <= 1 or p['uniform_multiple'] <= 1 or p['expansion'] < 1 or
            p['min_side'] <= 0 or not isinstance(p['max_candidates'], int) or p['max_candidates'] < 1 or
            not 0 < p['merge_iou'] <= 1 or not 0 < p['merge_containment'] <= 1):
        raise ValueError('invalid proposal thresholds or candidate cap')
    for key in ('min_coverage', 'min_occupancy', 'secondary_coverage', 'secondary_occupancy', 'giant_area_fraction'):
        if not 0 < label[key] <= 1:
            raise ValueError('label fractions must be in (0,1]')
    if min(label['min_pixels'], label['min_area_518'], label['secondary_pixels']) <= 0:
        raise ValueError('positive label sizes required')
    if (any(not isinstance(train[k], int) or train[k] < 1 for k in ('epochs', 'batch_size', 'heads', 'slots')) or
            train['lr'] <= 0 or 256 % train['heads'] or not isinstance(train['seeds'], list) or not train['seeds'] or
            any(not isinstance(s, int) or s < 0 or s >= 2**32 for s in train['seeds']) or
            len(set(train['seeds'])) != len(train['seeds'])):
        raise ValueError('invalid training budget/seeds')


def clip_box(box, size):
    """Integer half-open RGB bounds; A/B must use the same returned integers."""
    if len(box) != 4 or not np.isfinite(box).all() or min(size) <= 0:
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = size
    result = [max(0, min(w, math.floor(x0))), max(0, min(h, math.floor(y0))),
              max(0, min(w, math.ceil(x1))), max(0, min(h, math.ceil(y1)))]
    return result if result[2] > result[0] and result[3] > result[1] else None


def _area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def _mergeable(a, b, cfg):
    overlap = _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return (overlap / max(1, _area(a)+_area(b)-overlap) >= cfg['merge_iou'] or
            overlap / max(1, min(_area(a), _area(b))) >= cfg['merge_containment'])


def propose(spatial, size, cfg):
    """8-connected components of raw numerical conditional spatial probabilities."""
    p = np.asarray(spatial, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('expected finite nonnegative teacher spatial grid')
    total = p.sum()
    conditional = p/total if total > 0 else np.zeros_like(p)
    threshold = max(float(conditional.max())*cfg['relative_peak'], cfg['uniform_multiple']/p.size)
    active = conditional >= threshold
    gy, gx = p.shape
    visited = np.zeros_like(active)
    before = []
    for y, x in zip(*np.where(active), strict=True):
        if visited[y, x]:
            continue
        stack, cells = [(int(y), int(x))], []
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop(); cells.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < gy and 0 <= nx < gx and active[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True; stack.append((ny, nx))
        yy, xx = np.array(cells).T
        box = [float(xx.min()*size[0]/gx), float(yy.min()*size[1]/gy),
               float((xx.max()+1)*size[0]/gx), float((yy.max()+1)*size[1]/gy)]
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        w = max(cfg['min_side'], (box[2]-box[0])*cfg['expansion'])
        h = max(cfg['min_side'], (box[3]-box[1])*cfg['expansion'])
        before.append(dict(component_id=len(before), cells=[list(c) for c in cells],
                           component_box=box, box=clip_box([cx-w/2, cy-h/2, cx+w/2, cy+h/2], size),
                           peak=float(conditional[yy, xx].max())))
    merged = [dict(box=r['box'], merged_from=[r['component_id']], peak=r['peak']) for r in before]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i+1, len(merged)):
                a, b = merged[i], merged[j]
                if _mergeable(a['box'], b['box'], cfg):
                    a['box'] = [min(a['box'][0], b['box'][0]), min(a['box'][1], b['box'][1]),
                                max(a['box'][2], b['box'][2]), max(a['box'][3], b['box'][3])]
                    a['merged_from'] = sorted(a['merged_from']+b['merged_from'])
                    a['peak'] = max(a['peak'], b['peak'])
                    merged.pop(j); changed = True; break
            if changed:
                break
    ranked = sorted(merged, key=lambda r: (-r['peak'], r['box']))
    kept = sorted(ranked[:cfg['max_candidates']], key=lambda r: r['box'])
    return dict(threshold=threshold, spatial_mass=float(total), before_merge=before,
                discarded_after_limit=ranked[cfg['max_candidates']:],
                candidates=[dict(candidate_id=i, **r) for i, r in enumerate(kept)])


def pool_weights(size, box, image_size=518, output_side=4):
    """Area of each 4x4 ROI bin intersecting each letterboxed 14px feature patch.

    No annotation/mask input. Partially padded patches contribute only content area;
    their receptive fields can still contain padding, as in the original encoder.
    """
    side = image_size//14
    weights = np.zeros((output_side**2, side**2), np.float32)
    box = clip_box(box, size)
    if box is None:
        return weights
    w, h = size
    scale = min(image_size/w, image_size/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    left, top = (image_size-rw)//2, (image_size-rh)//2
    xs = left + np.linspace(box[0], box[2], output_side+1)*rw/w
    ys = top + np.linspace(box[1], box[3], output_side+1)*rh/h
    yy, xx = np.meshgrid(np.arange(side)*14, np.arange(side)*14, indexing='ij')
    for row in range(output_side):
        for col in range(output_side):
            wx = np.maximum(0, np.minimum(xx+14, xs[col+1])-np.maximum(xx, xs[col]))
            wy = np.maximum(0, np.minimum(yy+14, ys[row+1])-np.maximum(yy, ys[row]))
            weights[row*output_side+col] = (wx*wy).reshape(-1)
    return weights


def label_regions(boxes, masks, target, cfg, ignore=None):
    """GT is read only here, after final proposals; never changes their coordinates."""
    if target not in masks or not masks:
        raise ValueError('reference must have a physical object label')
    height, width = masks[target].shape
    if any(m.shape != (height, width) for m in masks.values()):
        raise ValueError('mask geometry mismatch')
    if ignore is not None and ignore.shape != (height, width):
        raise ValueError('ignore geometry mismatch')
    totals = {uid: int(np.count_nonzero(m)) for uid, m in masks.items()}
    rows, positives = [], []
    for i, original in enumerate(boxes):
        box = clip_box(original, (width, height))
        row = dict(valid_box=box is not None, vehicles={}, ambiguous=False, background=True,
                   fragment=False, confirmed_identity=None, area_fraction=0., giant=False,
                   ignore_pixels=0)
        if box is not None:
            x0, y0, x1, y1 = box
            area = _area(box)
            significant, qualified = [], []
            row['area_fraction'] = area/(width*height)
            row['giant'] = row['area_fraction'] >= cfg['giant_area_fraction']
            for uid, m in masks.items():
                pixels = int(np.count_nonzero(m[y0:y1, x0:x1]))
                coverage, occupancy = pixels/max(1, totals[uid]), pixels/area
                projected = pixels*(518/max(width, height))**2
                row['vehicles'][uid] = dict(pixels=pixels, coverage=coverage,
                                            occupancy=occupancy, effective_area_518=projected)
                if pixels > 0:
                    row['background'] = False
                if pixels >= cfg['secondary_pixels'] and (coverage >= cfg['secondary_coverage'] or
                                                           occupancy >= cfg['secondary_occupancy']):
                    significant.append(uid)
                if (pixels >= cfg['min_pixels'] and coverage >= cfg['min_coverage'] and
                        occupancy >= cfg['min_occupancy'] and projected >= cfg['min_area_518']):
                    qualified.append(uid)
            row['ignore_pixels'] = int(np.count_nonzero(ignore[y0:y1, x0:x1])) if ignore is not None else 0
            unknown = (row['ignore_pixels'] >= cfg['secondary_pixels'] and
                       row['ignore_pixels']/area >= cfg['secondary_occupancy'])
            row['ambiguous'] = len(significant) > 1 or unknown
            row['fragment'] = not row['background'] and not qualified and not row['ambiguous']
            if len(qualified) == 1 and not row['ambiguous']:
                row['confirmed_identity'] = qualified[0]
                if qualified[0] == target:
                    positives.append(i)
        rows.append(row)
    coverage = [r['vehicles'].get(target, {}).get('coverage', 0.) for r in rows]
    return dict(physical_visible=totals[target] > 0, visible_pixels=totals[target],
                positives=positives, candidates=rows, max_target_coverage=max(coverage, default=0.),
                any_target_intersection=any(c > 0 for c in coverage),
                coverage_at_threshold=any(c >= cfg['min_coverage'] for c in coverage))


def remap_positives(positives, permutation):
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError('candidate permutation must be a bijection')
    return [j for j, original in enumerate(permutation) if original in positives]
