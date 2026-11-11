"""Small, inspectable diagnostic statistics and offline reports; no model execution."""
import csv
import hashlib
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def score_description(logits, target, ever_seen):
    scores = [float(x) for x in logits]
    k = len(ever_seen)
    if len(scores) != k + 1 or not 0 <= target < k or not all(map(math.isfinite, scores)):
        raise ValueError('invalid candidate scores or target')
    truth = target if ever_seen[target] else k
    predicted = max(range(k + 1), key=scores.__getitem__)
    values = [math.exp(x - max(scores)) for x in scores]
    probabilities = [x / sum(values) for x in values]
    others = [scores[i] for i in range(k) if i != target and ever_seen[i]]
    available = [scores[i] for i in range(k) if ever_seen[i]]
    if predicted == truth:
        error = 'correct'
    elif truth == k:
        error = 'candidate_before_target_seen'
    elif predicted == k:
        error = 'null_instead_of_target'
    else:
        error = 'wrong_candidate'
    return dict(logits=scores, probabilities=probabilities, truth_index=truth,
                predicted_index=predicted, correct=predicted == truth, error_type=error,
                target_probability=probabilities[target],
                identity_margin=scores[target] - max(others) if ever_seen[target] and others else None,
                candidate_null_margin=max(available) - scores[k] if available else None)


def metric_counts(records):
    counts = {key: [0, 0] for key in ('all_prefix', 'current_visible', 'both_visible',
              'target_hidden_after_seen', 'target_not_seen', 'final')}
    for row in records:
        conditions = dict(all_prefix=True, current_visible=row['target_visible'],
                          both_visible=row['both_visible'],
                          target_hidden_after_seen=row['target_seen'] and not row['target_visible'],
                          target_not_seen=not row['target_seen'], final=row['is_final'])
        for key, keep in conditions.items():
            if keep:
                counts[key][0] += int(row['correct']); counts[key][1] += 1
    return {key: dict(correct=a, count=b, accuracy=a / b if b else None)
            for key, (a, b) in counts.items()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ['no_records']
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [(dict(zip(keys, key, strict=True)), values) for key, values in groups.items()]


def population(rows):
    return dict(prefixes=len(rows), unique_frames=len({(r['scene_id'], r['frame_id']) for r in rows}),
                unique_prefixes=len({r['prefix_id'] for r in rows}),
                routes=len({r['path_id'] for r in rows}),
                layouts=len({r['scene_id'] for r in rows}))


def error_table(rows, extra=()):
    result = []
    for group, values in grouped(rows, ('split', 'mode', 'seed', 'target_id', *extra)):
        def rate(kind, values=values):
            return sum(r['error_type'] == kind for r in values) / len(values)
        margins = [r['identity_margin'] for r in values if r['identity_margin'] is not None]
        result.append(dict(**group, **population(values), correct=rate('correct'),
                           wrong_candidate=rate('wrong_candidate'), null=rate('null_instead_of_target'),
                           candidate_before_seen=rate('candidate_before_target_seen'),
                           pairwise_correct=sum(m > 0 for m in margins) / len(margins) if margins else None,
                           pairwise_ties=sum(m == 0 for m in margins),
                           pairwise_count=len(margins)))
    return result


def control_table(rows):
    result = []
    for subset in ('all_eligible', 'both_visible'):
        selected = [r for r in rows if subset == 'all_eligible' or r['both_visible']]
        for group, values in grouped(selected, ('split', 'mode', 'seed', 'condition', 'stage')):
            base = mean(float(r['baseline']['correct']) for r in values)
            changed = mean(float(r['intervention']['correct']) for r in values)
            result.append(dict(**group, subset=subset, **population(values), baseline_accuracy=base,
                               intervention_accuracy=changed, accuracy_change=changed - base,
                               correct_to_wrong=sum(r['baseline']['correct'] and not r['intervention']['correct']
                                                    for r in values),
                               wrong_to_correct=sum(not r['baseline']['correct'] and r['intervention']['correct']
                                                    for r in values)))
    return result


def repeat_table(rows):
    result = []
    for group, values in grouped(rows, ('split', 'mode', 'seed', 'history_type', 'extra_reads')):
        changes = [r['max_probability_change'] for r in values]
        result.append(dict(**group, samples=len(values), mean_change=mean(changes),
                           median_change=median(changes), max_change=max(changes),
                           choice_flips=sum(r['choice_flipped'] for r in values),
                           with_next=sum(r.get('after_next') is not None for r in values)))
    return result


def reference_pairs(rows):
    result = []
    for group, values in grouped(rows, ('split', 'mode', 'seed', 'task_id', 'step')):
        if len(values) < 2:
            continue
        result.append(dict(**group, scene_id=values[0]['scene_id'], frame_id=values[0]['frame_id'],
                           query_ids=[r['target_id'] for r in values],
                           predicted_ids=[r['predicted_id'] for r in values],
                           all_queries_correct=all(r['correct'] for r in values),
                           prediction_changes=len({r['predicted_index'] for r in values}) > 1,
                           all_queries_currently_visible=all(r['target_visible'] for r in values)))
    return result


def make_examples(output, root, rows, limit):
    """Round-robin across declared error/quality groups; retain full prefix context."""
    folder = output / 'examples'; assets = folder / 'assets'; assets.mkdir(parents=True)
    bins = defaultdict(list)
    for row in rows:
        bins[(row['split'], row['error_type'], row['quality_group'], row['mode'], row['seed'])].append(row)
    selected = []
    while bins and len(selected) < limit:
        for key in list(bins):
            if len(selected) == limit:
                break
            selected.append(bins[key].pop(0))
            if not bins[key]:
                del bins[key]
    prefixes = defaultdict(list)
    for row in rows:
        prefixes[(row['split'], row['mode'], row['seed'], row['task_id'])].append(row)

    def copy_asset(relative):
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError('example asset outside dataset')
        name = hashlib.sha256(relative.encode()).hexdigest()[:20] + source.suffix
        target = assets / name
        if not target.exists():
            shutil.copyfile(source, target)
        return 'assets/' + name

    cards = []
    for row in selected:
        scene_src, ref_src = copy_asset(row['image_path']), copy_asset(row['reference_path'])
        candidates = []
        for candidate in row['candidate_metadata']:
            mask = candidate['mask_path']
            thumb = (f'<img class="mask" src="{copy_asset(mask)}" alt="instance mask">' if mask else '')
            candidates.append(f'<div>{html.escape(candidate["candidate_id"])} '
                              f'{html.escape(candidate["quality"])} {thumb}</div>')
        history = [p for p in prefixes[(row['split'], row['mode'], row['seed'], row['task_id'])]
                   if p['step'] <= row['step']]
        history_html = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            p['step'], p['frame_id'], p['predicted_id'], p['error_type'],
            [round(v, 3) for v in p['probabilities']])) + '</tr>' for p in history)
        cards.append(f'<article><h2>{html.escape(row["mode"])} seed={row["seed"]} '
                     f'{html.escape(row["split"])} · {html.escape(row["error_type"])}</h2>'
                     f'<p>目标 {html.escape(row["target_id"])}；候选顺序 '
                     f'{html.escape(str(row["candidate_ids"]))}，最后一列为 Null。'
                     f'原始帧 {html.escape(row["frame_id"])}</p>'
                     f'<div class="images"><img src="{ref_src}" alt="reference">'
                     f'<img src="{scene_src}" alt="aerial"></div>'
                     f'<details><summary>独立实例掩码与质量</summary>{"".join(candidates)}</details>'
                     f'<table><tr><th>步</th><th>帧</th><th>预测</th><th>错误类型</th><th>概率</th></tr>'
                     f'{history_html}</table></article>')
    page = ('<!doctype html><meta charset="utf-8"><title>车辆记忆诊断样例</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}'
            'article{border-top:1px solid #bbb;padding:16px 0}.images{display:flex;gap:16px}'
            '.images img{width:46%;object-fit:contain}.mask{width:240px;display:block}'
            'td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse;font-size:13px}</style>'
            '<h1>车辆记忆诊断样例</h1><p>按数据分组、错误类型、质量、模式和种子轮流取样；'
            '显示完整观察前缀，不只展示支持某个假设的错误。原图与实例掩码分开显示。</p>'
            + ''.join(cards))
    (folder / 'index.html').write_text(page, encoding='utf-8')
    write_json(output / 'example_selection.json', dict(policy='round_robin_split_error_quality_mode_seed',
                limit=limit, selected=[dict(mode=r['mode'], seed=r['seed'], split=r['split'],
                                          task_id=r['task_id'], step=r['step']) for r in selected]))


def write_reports(output, root, normal, controls, switches, repeats, checks, example_limit):
    write_jsonl(output / 'steps.jsonl', normal)
    write_jsonl(output / 'history_controls.jsonl', controls)
    write_jsonl(output / 'reference_switch.jsonl', switches)
    write_jsonl(output / 'repeat_checks.jsonl', repeats)
    both = [r for r in normal if r['both_visible']]
    write_csv(output / 'error_types.csv', error_table(both))
    write_csv(output / 'position_quality.csv', error_table(both, ('quality_group', 'target_horizontal')))
    write_csv(output / 'reference_switch.csv', error_table(switches, ('both_visible',)))
    write_csv(output / 'reference_switch_pairs.csv', reference_pairs(switches))
    write_csv(output / 'history_controls.csv', control_table(controls))
    write_csv(output / 'repeat_checks.csv', repeat_table(repeats))
    write_json(output / 'replay_checks.json', checks)
    lines = ['# 车辆证据记忆诊断', '', '本报告由固定权重推理生成，没有训练或修改旧结果。', '',
             '| 划分 | 模式 | 种子 | 双车正确 | 选另一候选 | 选 Null |', '|---|---|---:|---:|---:|---:|']
    for group, values in grouped(both, ('split', 'mode', 'seed')):
        counts = [sum(r['error_type'] == name for r in values)
                  for name in ('correct', 'wrong_candidate', 'null_instead_of_target')]
        lines.append(f'| {group["split"]} | {group["mode"]} | {group["seed"]} | '
                     + ' | '.join(f'{n}/{len(values)} ({n/len(values):.1%})' for n in counts) + ' |')
    lines += ['', '## 口径与限制', '',
              '- steps 中 truth_index 保留缓存 valid 累积的旧口径；physical_truth_index 单独记录原始掩码口径。',
              '- error_types 只统计目标与干扰当前同框；pairwise_correct 是忽略 Null 的两车排序诊断。',
              '- history_controls 中每一行使用同一检查点、相同当前输入，真值不随干预改动。',
              '- swap_before 在新图输入前比较；swap_after 再喂相同当前图；只交换已见候选 slots。',
              '- pause 仅在目标此前已见时报告；reset 包含全部前缀。不同条件样本数可能不同。',
              '- any_single_frame_correct_so_far 仅检查目标可见的单帧身份判断，不计早期正确 Null；这是离线诊断。',
              '- repeat 按空历史/一张可用观察/可用与弱观察混合，以及参考身份限额取样；缺失类别不补造。',
              '- 后续新图选本路线中首个不同 RGB 且目标标注为 usable 的观察；标签仅作诊断选样，不是在线策略。',
              '- usable 是数据集既有临时质量规则，不保证该图确实提供新线索。',
              '- 布局、帧和前缀重复相关；多种子不替代独立场景。不能仅凭位置相关性认定位置捷径。',
              '- 查看 replay_checks.json 中 verified；partial smoke 或缺少原 fit 记录时不得声称完整重放通过。',
              '', '更多字段见 docs/evidence-memory-diagnostics.md。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    make_examples(output, root, normal, example_limit)
