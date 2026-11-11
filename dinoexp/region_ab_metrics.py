"""Pure evaluation logic: physical visibility is independent from proposal Null."""
import numpy as np


def teacher_metrics(rows, labels):
    ls = [labels[r['id']] for r in rows]
    visible = [l for l in ls if l['physical_visible']]
    candidates = [c for l in ls for c in l['candidates']]
    return dict(pairs=len(rows), unique_rgb=len({r['image_sha256'] for r in rows}),
                physical_visible=len(visible), physically_absent=len(ls)-len(visible),
                covered_at_least_one_pixel=sum(l['any_target_intersection'] for l in visible),
                covered_50_percent=sum(l['coverage_at_threshold'] for l in visible),
                confirmable_covered=sum(bool(l['positives']) for l in visible),
                not_confirmably_covered=sum(not l['positives'] for l in visible),
                complete_spatial_miss=sum(not l['any_target_intersection'] for l in visible),
                candidates=len(candidates), mean_candidates=len(candidates)/max(1, len(rows)),
                empty_candidate_pairs=sum(not r['candidates'] for r in rows),
                multi_positive_pairs=sum(len(l['positives']) > 1 for l in ls),
                ambiguous=sum(c['ambiguous'] for c in candidates), fragments=sum(c['fragment'] for c in candidates),
                background=sum(c['background'] for c in candidates), giant=sum(c['giant'] for c in candidates),
                area_fraction_quantiles=np.quantile([c['area_fraction'] for c in candidates], [0, .25, .5, .75, 1]).tolist() if candidates else [],
                max_target_coverage_mean=float(np.mean([l['max_target_coverage'] for l in visible])) if visible else None)


def metric_counts(predictions, labels):
    result = dict(decisions=len(predictions), covered=0, correct_region=0, wrong_region=0, false_null=0,
                  ranking_correct=0, ranking_evaluable=0, ranking_with_distractors=0, ranking_correct_with_distractors=0,
                  downstream_correct=0, physical_visible=0, visible_found=0,
                  absent=0, absent_rejected=0, absent_false_alarm=0, end_to_end_correct=0,
                  teacher_not_confirmable=0, extraction_failure=0, ambiguous_selected=0)
    for p in predictions:
        l = labels[p['id']]; positives = l['positives']; choice = p['choice']; failure = p['extraction_failure']
        null = choice is None
        region_correct = not null and choice in positives
        downstream = region_correct or (not positives and null)
        result['extraction_failure'] += failure
        result['downstream_correct'] += downstream and not failure
        if positives:
            result['covered'] += 1
            result['correct_region'] += region_correct and not failure
            result['false_null'] += null
            result['wrong_region'] += not null and (not region_correct or failure)
            result['ranking_evaluable'] += 1
            rank_ok = p['rank_choice'] in positives and not failure
            result['ranking_correct'] += rank_ok
            if len(p['scores'])-1 > len(positives):
                result['ranking_with_distractors'] += 1
                result['ranking_correct_with_distractors'] += rank_ok
        if l['physical_visible']:
            result['physical_visible'] += 1
            result['visible_found'] += region_correct and not failure
            result['teacher_not_confirmable'] += not positives
            success = region_correct and not failure
        else:
            result['absent'] += 1
            result['absent_rejected'] += null and not failure
            result['absent_false_alarm'] += not null
            success = null and not failure
        result['end_to_end_correct'] += success
        if not null:
            result['ambiguous_selected'] += l['candidates'][choice]['ambiguous']
    rates = {'conditional_accuracy': ('correct_region', 'covered'), 'ranking_accuracy': ('ranking_correct', 'ranking_evaluable'),
             'ranking_with_distractors_accuracy': ('ranking_correct_with_distractors', 'ranking_with_distractors'),
             'downstream_accuracy': ('downstream_correct', 'decisions'), 'visible_recall': ('visible_found', 'physical_visible'),
             'absent_rejection': ('absent_rejected', 'absent'), 'end_to_end_accuracy': ('end_to_end_correct', 'decisions')}
    for name, (numerator, denominator) in rates.items():
        result[name] = result[numerator]/result[denominator] if result[denominator] else None
    return result
