"""Mask and identity-overlap diagnostics. Pure numpy; not DAVIS J&F."""
from collections import defaultdict

import numpy as np


def mean(values):
    return float(np.mean(values)) if values else None


def score_frame(prediction, masks, target, ignore, *, success_iou=.5, wrong_iou=.1,
                wrong_margin=.05, absent_pixels=32):
    prediction, ignore = np.asarray(prediction, bool), np.asarray(ignore, bool)
    if target not in masks or any(np.asarray(m).shape != prediction.shape for m in masks.values()):
        raise ValueError('prediction/GT shape or target mismatch')
    if ignore.shape != prediction.shape:
        raise ValueError('ignore shape mismatch')
    valid = ~ignore
    pred, gt = prediction & valid, np.asarray(masks[target], bool) & valid
    area, pred_area = int(gt.sum()), int(pred.sum())
    intersection = int((pred & gt).sum())
    union = int((pred | gt).sum())
    iou = intersection/union if area else None
    overlaps = {}
    for uid, raw in masks.items():
        mask = np.asarray(raw, bool) & valid
        denom = int((pred | mask).sum())
        overlaps[uid] = float((pred & mask).sum()/denom) if denom else 0.
    other = {uid: val for uid, val in overlaps.items() if uid != target}
    best_other = max(other, key=other.get) if other else None
    other_iou = other[best_other] if best_other else 0.
    wrong = other_iou >= wrong_iou and other_iou > overlaps[target]+wrong_margin
    ys, xs = np.nonzero(gt)
    boundary = bool(area and (xs.min() == 0 or ys.min() == 0 or
                             xs.max() == gt.shape[1]-1 or ys.max() == gt.shape[0]-1))
    return dict(target_pixels=area, prediction_pixels=pred_area, intersection=intersection,
                union=union, iou=iou, success=bool(area and iou >= success_iou),
                visible=bool(area), boundary=boundary, tiny=0 < area < 32,
                two_visible=bool(area and any(np.any(np.asarray(m, bool) & valid)
                                             for uid, m in masks.items() if uid != target)),
                absent_false_positive=bool(not area and pred_area >= absent_pixels),
                wrong_vehicle_overlap=bool(wrong), other_identity=best_other,
                other_iou=other_iou, overlaps=overlaps,
                ignored_prediction_pixels=int((prediction & ignore).sum()))


def recovery_events(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['case']].append(row)
    events = []
    for case, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['step'])
        for i in range(1, len(rows)):
            if not rows[i]['visible'] or rows[i-1]['visible']:
                continue
            visible_run = []
            for row in rows[i:]:
                if not row['visible']:
                    break
                visible_run.append(row)
            successes = [r for r in visible_run if r['success']]
            first = successes[0]['step'] if successes else None
            events.append(dict(case=case, return_step=rows[i]['step'],
                               visible_steps=len(visible_run), recovered=first is not None,
                               delay_steps=None if first is None else first-rows[i]['step'],
                               recovered_within_3_steps=any(r['success'] for r in visible_run[:3])))
    return events


def summarize(records):
    rows = [r for r in records if r['step'] > 0]
    visible = [r for r in rows if r['visible']]
    absent = [r for r in rows if not r['visible']]
    two = [r for r in rows if r['two_visible']]
    recovery = recovery_events(records)
    return dict(scored_steps=len(rows), initial_steps_excluded=sum(r['step'] == 0 for r in records),
                visible_steps=len(visible), absent_steps=len(absent),
                mean_visible_iou=mean([r['iou'] for r in visible]),
                visible_success_rate=mean([r['success'] for r in visible]),
                absent_false_positive_rate=mean([r['absent_false_positive'] for r in absent]),
                wrong_vehicle_steps=sum(r['wrong_vehicle_overlap'] for r in rows),
                two_visible_steps=len(two),
                two_visible_wrong_rate=mean([r['wrong_vehicle_overlap'] for r in two]),
                boundary_steps=sum(r['boundary'] for r in visible),
                boundary_mean_iou=mean([r['iou'] for r in visible if r['boundary']]),
                tiny_steps=sum(r['tiny'] for r in visible),
                tiny_mean_iou=mean([r['iou'] for r in visible if r['tiny']]),
                recovery_events=len(recovery),
                recovery_rate=mean([e['recovered'] for e in recovery]),
                recovery_within_3_steps_rate=mean([e['recovered_within_3_steps'] for e in recovery]),
                mean_recovery_delay_successful=mean([e['delay_steps'] for e in recovery if e['recovered']]))


def build_summary(records):
    return dict(overall=summarize(records),
                by_case={key: summarize([r for r in records if r['case'] == key])
                         for key in sorted({r['case'] for r in records})},
                by_target={key: summarize([r for r in records if r['target'] == key])
                           for key in sorted({r['target'] for r in records})},
                recovery=recovery_events(records))
