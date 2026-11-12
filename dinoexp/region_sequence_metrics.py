"""Pure-data metrics: accepted evidence and physical appearances stay separate."""
from collections import Counter

import numpy as np


def summarize(predictions):
    counts = Counter(prefixes=len(predictions))
    per_target = {}
    for p in predictions:
        truth, choice = p['accepted_truth'], p['choice_identity']
        correct = choice == truth
        positive, physical = truth is not None, p['physical_truth'] is not None
        counts['correct'] += int(correct)
        counts['physical_correct'] += int(choice == p['physical_truth'])
        counts['physical_target_prefixes'] += int(physical)
        counts['physical_target_correct'] += int(physical and choice == p['physical_truth'])
        counts['no_physical_target_prefixes'] += int(not physical)
        counts['accepted_target_prefixes'] += int(positive)
        counts['accepted_target_correct'] += int(positive and correct)
        counts['false_null'] += int(positive and choice is None)
        counts['wrong_vehicle'] += int(positive and choice is not None and not correct)
        counts['no_accepted_target_prefixes'] += int(not positive)
        counts['null_rejections'] += int(not positive and choice is None)
        counts['physical_seen_without_accepted_target'] += int(physical and not positive)
        counts['two_seen_target_prefixes'] += int(positive and len(p['seen_identities']) >= 2)
        counts['two_seen_target_correct'] += int(positive and len(p['seen_identities']) >= 2 and correct)
        counts['two_seen_target_rank_correct'] += int(positive and len(p['seen_identities']) >= 2
                                                       and p['rank_identity'] == truth)
        history_only = positive and truth not in p['current_identities']
        counts['history_only_target_prefixes'] += int(history_only)
        counts['history_only_target_correct'] += int(history_only and correct)
        target = per_target.setdefault(p['target_identity'], dict(positive=0, correct=0))
        target['positive'] += int(positive); target['correct'] += int(positive and correct)
    ratio = lambda a, b: counts[a]/counts[b] if counts[b] else None
    return dict(counts=dict(counts), accepted_accuracy=ratio('correct', 'prefixes'),
                physical_accuracy=ratio('physical_correct', 'prefixes'),
                physical_target_accuracy=ratio('physical_target_correct', 'physical_target_prefixes'),
                accepted_target_accuracy=ratio('accepted_target_correct', 'accepted_target_prefixes'),
                null_rejection=ratio('null_rejections', 'no_accepted_target_prefixes'),
                two_seen_target_accuracy=ratio('two_seen_target_correct', 'two_seen_target_prefixes'),
                two_seen_target_ranking=ratio('two_seen_target_rank_correct', 'two_seen_target_prefixes'),
                history_only_target_accuracy=ratio('history_only_target_correct', 'history_only_target_prefixes'),
                always_null_accepted_accuracy=ratio('no_accepted_target_prefixes', 'prefixes'),
                always_null_physical_accuracy=ratio('no_physical_target_prefixes', 'prefixes'),
                per_target={uid: dict(v, accuracy=v['correct']/v['positive'] if v['positive'] else None)
                            for uid, v in per_target.items()})


def control_summary(rows):
    result = {}
    for name in sorted({r['control'] for r in rows}):
        selected = [r for r in rows if r['control'] == name and r['eligible']]
        result[name] = dict(
            pairs=len(selected),
            normal_correct=sum(r['normal_correct'] for r in selected),
            control_correct=sum(r['control_correct'] for r in selected),
            normal_wrong_control_right=sum(not r['normal_correct'] and r['control_correct'] for r in selected),
            normal_right_control_wrong=sum(r['normal_correct'] and not r['control_correct'] for r in selected),
            choice_flips=sum(r['normal_choice_identity'] != r['choice_identity'] for r in selected),
            max_target_probability_change=max((abs(r['target_probability']-r['normal_target_probability'])
                                               for r in selected), default=None))
    return result


def aggregate(run):
    keys = ('accepted_accuracy', 'physical_accuracy', 'physical_target_accuracy',
            'accepted_target_accuracy', 'two_seen_target_accuracy', 'two_seen_target_ranking',
            'history_only_target_accuracy', 'always_null_accepted_accuracy', 'always_null_physical_accuracy')
    output = {}
    for split in sorted({s for r in run['results'] for s in r['splits']}):
        output[split] = {}
        for method in ('last', 'mean', 'memory'):
            entries = [r for r in run['results'] if r['method'] == method]
            output[split][method] = {}
            for key in keys:
                values = {str(r['seed']): r['splits'][split]['metrics'][key] for r in entries}
                numbers = [v for v in values.values() if v is not None]
                output[split][method][key] = dict(per_seed=values,
                    mean=float(np.mean(numbers)) if numbers else None,
                    population_std=float(np.std(numbers)) if numbers else None)
    return output
