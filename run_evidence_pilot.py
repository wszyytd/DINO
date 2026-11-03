"""Run the location-separated, equal-cost active-evidence pilot on real cached photographs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch

from dinoexp.evidence_data import load_episodes, split_by_position, episode_hashes, FeatureBank
from dinoexp.evidence_labels import FrozenRuntime, CLSRuntime, generate_labels
from dinoexp.evidence_model import EvidenceModel
from dinoexp.evidence_policy import rollout, summarize_rollouts
from dinoexp.evidence_training import (train_memory, train_value, load_memory, load_value,
                                       evaluate_value, save_json, seed_all)
from dinoexp.local_features import resolve_device
from dinoexp.local_pairs import sha256


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='directory containing capture scenes')
    p.add_argument('--cache', type=Path, required=True, help='existing complete frozen feature cache')
    p.add_argument('--output', type=Path, default=Path('out') / (
        'active_evidence_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--allow-pilot-location-split', action='store_true',
                   help='acknowledge identity overlap / same-background reference limitations')
    p.add_argument('--repr-position', default='42')
    p.add_argument('--value-position', default='64')
    p.add_argument('--eval-position', default='105')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--grid', type=int, default=8)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--memory-tokens', type=int, default=4)
    p.add_argument('--blocks', type=int, default=1)
    p.add_argument('--memory-steps', type=int, default=120)
    p.add_argument('--value-steps', type=int, default=300)
    p.add_argument('--states-per-episode', type=int, default=12)
    p.add_argument('--budget', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--random-repeats', type=int, default=3)
    p.add_argument('--alphas', type=float, nargs='+', default=[0., .1])
    p.add_argument('--score-temperature', type=float, default=1.)
    p.add_argument('--stop-gap', type=float,
                   help='optional uncalibrated stopping diagnostic; default fixed-budget ranks')
    p.add_argument('--dry-run', action='store_true')
    return p


def _policies(runtime, episodes, output, args, value=None):
    reports, traces = {}, []
    policies = ['random', 'fixed', 'coverage', 'oracle']
    if value is not None:
        policies.append('learned')
    for policy in policies:
        policy_traces = []
        repeats = args.random_repeats if policy == 'random' else 1
        for repeat in range(repeats):
            for index, episode in enumerate(episodes):
                policy_traces.append(rollout(runtime, episode, policy, args.budget,
                                             seed=args.seed + 1000*repeat + index,
                                             value=value, stop_gap=args.stop_gap))
        reports[policy] = summarize_rollouts(policy_traces, args.budget)
        traces.extend(policy_traces)
        print(f'{output.name} / {policy}: '
              f'{reports[policy]["rank_accuracy_by_budget"]}', flush=True)
    save_json(output/'policies.json', reports)
    save_json(output/'traces.json', traces)
    return reports


def _write_report(output, result):
    args = result['arguments']
    lines = ['# 主动证据收集：首次离线实验', '',
             '本次使用真实 CARLA 照片的冻结 DINOv2 特征。代码实现和流程实验已执行；'
             '结果不能作为身份互斥泛化、跨背景参考匹配或实际飞行效率的结论。', '',
             '## 实验范围', '',
             f'- 表征训练位置：{args["repr_position"]}；价值训练位置：{args["value_position"]}；'
             f'评测位置：{args["eval_position"]}。',
             '- 位置角色互斥，车型身份重叠；同一位置内的不同参考和光照并非独立场景。',
             f'- 骨干冻结；内部维度 {args["dim"]}；特征空间池化到 {args["grid"]}×{args["grid"]}；'
             f'记忆 token {args["memory_tokens"]}；种子 {args["seed"]}。',
             f'- 每任务最多新增 {args["budget"]} 次单候选观察，失败尝试同样计数。',
             '- 选择固定训练步数的最后检查点；没有根据评测位置选择模型或调参。',
             '- 默认关闭提前确认，下面报告固定预算的正确排序率，不是已校准的确认成功率。', '',
             '## 固定观察预算结果', '',
             '| 表征 | 选视角策略 | 初始 | 1 次 | 2 次 | 4 次 | 最终预算 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, report in result['policies'].items():
        for policy, values in report.items():
            curve = values['rank_accuracy_by_budget']
            points = [curve[min(i, len(curve)-1)] for i in (0, 1, 2, 4, args['budget'])]
            lines.append(f'| {name} | {policy} | ' + ' | '.join(f'{100*x:.1f}%' for x in points) + ' |')
    lines.extend(['', '## 价值头诊断', '',
                  '| 分支 | 价值 MSE | 全零预测 MSE | 平均动作选择损失 | 非并列状态选中最优动作 |',
                  '|---|---:|---:|---:|---:|'])
    for name, values in result['value_diagnostics'].items():
        hit = values['best_action_rate_nonconstant']
        hit_text = f'{100*hit:.1f}%' if hit is not None else '无非并列状态'
        lines.append(f'| {name} | {values["mse"]:.6f} | {values["zero_predictor_mse"]:.6f} | '
                     f'{values["mean_action_regret"]:.6f} | {hit_text} |')
    lines.extend(['', '## 解读边界', '',
                  'oracle 使用真实未来观察与目标身份，只用于分析；它是单步贪心参考，'
                  '不代表全局最短路径。其他策略不读取未来特征或目标标签。', '',
                  '随机策略重复次数不会增加独立场景数。单个评测位置的小样本结果只用于'
                  '定位记忆、匹配与动作价值问题；不能证明流形假设或预测辅助一定有效。', '',
                  '若学习策略没有稳定优于随机／覆盖策略，应按实际结果报告，先检查上游匹配'
                  '与标签分布；不能根据这个评测位置反复选超参数后继续称其为独立测试。', '',
                  '完整数据审计、配置、逐步损失、标签、检查点和动作轨迹位于本目录。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(args):
    if not args.allow_pilot_location_split:
        raise ValueError('pass --allow-pilot-location-split for this explicitly limited experiment')
    if min(args.threads, args.memory_steps, args.value_steps, args.states_per_episode,
           args.random_repeats) < 1 or not 0 <= args.budget <= 10:
        raise ValueError('positive training counts required; budget must be in [0,10]')
    if len(set(args.alphas)) != len(args.alphas) or any(a < 0 for a in args.alphas):
        raise ValueError('auxiliary weights must be unique and nonnegative')
    torch.set_num_threads(args.threads)
    episodes, audit = load_episodes(args.data, args.cache)
    roles = split_by_position(episodes, {'repr': [args.repr_position],
                                        'value': [args.value_position],
                                        'eval': [args.eval_position]})
    if any(not items for items in roles.values()):
        raise ValueError('each data role must contain at least one complete episode')
    audit['roles'] = {name: {'episodes': [e.id for e in items],
                            'groups': sorted({e.group for e in items}),
                            'identities': sorted({i for e in items for i in e.identities})}
                      for name, items in roles.items()}
    print(json.dumps({'roles': {k: len(v) for k, v in roles.items()},
                      'skipped': audit['skipped'], 'scope': audit['scope']}, ensure_ascii=False),
          flush=True)
    if args.dry_run:
        return audit
    args.output.mkdir(parents=True, exist_ok=False)
    arguments = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()}
    result = {'status': 'running', 'arguments': arguments, 'audit': audit,
              'policies': {}, 'value_diagnostics': {}}
    source_root = Path(__file__).resolve().parent
    result['code'] = {'git_head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(),
        'sha256': {str(p.relative_to(source_root)): sha256(p) for p in
                   [Path(__file__).resolve(), *sorted((source_root/'dinoexp').glob('evidence_*.py'))]}}
    save_json(args.output/'run.json', result)
    save_json(args.output/'data_audit.json', audit)
    try:
        device = resolve_device(args.device, torch)
        bank = FeatureBank(args.cache, episode_hashes([e for values in roles.values() for e in values]),
                           grid=args.grid)
        result['feature_config'] = bank.config
        baseline = args.output/'cls_mean'
        baseline.mkdir()
        result['policies']['cls_mean'] = _policies(CLSRuntime(bank), roles['eval'],
                                                  baseline, args)
        for alpha in args.alphas:
            name = f'alpha_{alpha:g}'
            folder = args.output/name
            folder.mkdir()
            seed_all(args.seed)
            model = EvidenceModel(input_dim=bank.config['input_dim'], dim=args.dim,
                                  layers=len(bank.config['layers']), heads=4,
                                  memory_tokens=args.memory_tokens, blocks=args.blocks).to(device)
            train_memory(model, bank, roles['repr'], folder/'memory', steps=args.memory_steps,
                         alpha=alpha, seed=args.seed)
            model, _ = load_memory(folder/'memory'/'last.pt', bank.config, device=device)
            runtime = FrozenRuntime(model, bank, args.score_temperature)
            training_labels = generate_labels(runtime, roles['value'],
                                              args.states_per_episode, args.seed)
            train_value(runtime, training_labels, folder/'value',
                        steps=args.value_steps, seed=args.seed)
            value = load_value(folder/'value'/'last.pt', runtime.fingerprint, device=device)
            evaluation_labels = generate_labels(runtime, roles['eval'],
                                                max(2, args.states_per_episode//2), args.seed+11)
            result['value_diagnostics'][name] = evaluate_value(value, evaluation_labels, device)
            result['policies'][name] = _policies(runtime, roles['eval'], folder, args, value)
            save_json(args.output/'run.json', result)
        result['status'] = 'complete'
        save_json(args.output/'run.json', result)
        _write_report(args.output, result)
        print(f'Report: {(args.output / "report.md").resolve()}', flush=True)
        return result
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_json(args.output/'run.json', result)
        raise


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
