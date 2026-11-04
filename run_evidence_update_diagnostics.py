"""Compare single and phase-expected joint updates on the same saved 16 states."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from dinoexp.evidence_training import save_json
from dinoexp.evidence_update_diagnostics import fit_updates
from dinoexp.local_pairs import sha256


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    branches = list(result['runs'])
    for axis_key, axis_label, filename in (
            ('step', 'Optimizer updates', 'comparison_updates'),
            ('training_state_forwards', 'Training state forwards (evaluation excluded)', 'comparison_work')):
        fig, axes = plt.subplots(2, len(branches), figsize=(6*len(branches), 7), squeeze=False)
        for col, branch in enumerate(branches):
            for mode, color in [('single', '#2966b4'), ('joint', '#16815c')]:
                curves = [value[mode]['curve'] for value in result['runs'][branch].values()]
                x = [v[axis_key] for v in curves[0]]
                for row, metric in enumerate(('weighted_mse', 'mean_action_regret')):
                    values = np.array([[v[metric] for v in curve] for curve in curves])
                    for series in values:
                        axes[row, col].plot(x, series, color=color, alpha=.22, linewidth=.8)
                    axes[row, col].plot(x, values.mean(0), color=color, label=mode, linewidth=2)
            axes[0, col].set_title(branch+' | fixed 16 training states')
            axes[0, col].set_yscale('log')
            for row in range(2):
                axes[row, col].set_xlabel(axis_label)
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend()
        axes[0, 0].set_ylabel('Weighted MSE (log axis)')
        axes[1, 0].set_ylabel('Mean action regret')
        fig.suptitle('Same phase weights and initialization | thin: seeds, thick: mean | training only')
        fig.tight_layout()
        fig.savefig(output/(filename+'.png'), dpi=150)
        fig.savefig(output/(filename+'.svg'))
        plt.close(fig)
    lines = ['# 固定 16 状态的更新方式对照', '',
             '两种方式每个训练阶段的期望状态权重相同：奇数步只用困难状态，偶数步使用全部状态。'
             'single 抽一个状态；joint 求该阶段的全部状态梯度平均。网络、初始化、学习率和标签不变。', '',
             '每个 joint 更新前向处理 16 状态，每个 single 更新处理 1 状态。'
             '表中 125 次 joint 更新与 2000 次 single 更新具有相同的训练状态前向次数，'
             '但优化器更新次数、梯度计算和 AdamW 衰减次数不同，不能称完全等算力。', '',
             '| 分支 | seed | 更新方式 | 更新数 | 状态前向数 | 加权 MSE | 动作 regret |',
             '|---|---:|---|---:|---:|---:|---:|']
    for branch, seeds in result['runs'].items():
        for seed, modes in seeds.items():
            for mode, run in modes.items():
                checkpoints = [run['curve'][-1]]
                if mode == 'joint':
                    checkpoints = [v for v in run['curve'] if v['step'] == result['arguments']['steps']//16]+checkpoints
                for v in checkpoints:
                    lines.append(f'| {branch} | {seed} | {mode} | {v["step"]} | '
                                 f'{v["training_state_forwards"]} | {v["weighted_mse"]:.6g} | '
                                 f'{v["mean_action_regret"]:.6g} |')
    lines += ['', '![相同更新次数](comparison_updates.png)', '',
              '![按训练状态前向次数](comparison_work.png)', '',
              '每个 runs 子目录保留逐步各状态损失、整体损失变化、梯度夹角矩阵和最终权重。'
              '梯度方向冲突本身不能证明结构错误；需结合真实更新造成的其他状态损失变化。', '',
              '本实验仅覆盖原来按收益范围筛出的同一批 pair1 训练状态，不是独立身份或场景的泛化成绩。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-run', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    if args.steps < 16 or args.steps % 16 or args.threads < 1 or args.lr <= 0:
        parser.error('steps must be a positive multiple of 16; positive threads/lr required')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct')
    baseline = json.loads((args.diagnostic_run/'run.json').read_text(encoding='utf-8'))
    if baseline['status'] != 'complete':
        raise ValueError('previous diagnostic is incomplete')
    indices = baseline['selection']['subsets']['16']
    if len(indices) != 16 or len(set(indices)) != 16:
        raise ValueError('need the original 16 distinct state indices')
    torch.set_num_threads(args.threads)
    branches, configs = {}, {}
    for branch in ('alpha_0', 'alpha_0.1'):
        source = args.source_run/branch
        paths = {'memory': source/'memory/last.pt', 'labels': source/'value/labels.pt'}
        if {k: sha256(p) for k, p in paths.items()} != baseline['inputs_sha256'][branch]:
            raise ValueError('source checkpoint/labels changed since the previous diagnostic')
        rows = torch.load(paths['labels'], map_location='cpu', weights_only=True)
        meta = json.loads((source/'value/training.json').read_text(encoding='utf-8'))
        allowed = set(meta['training_episodes'])
        if any(r['episode_id'] not in allowed or r['upstream_fingerprint'] != meta['upstream_fingerprint']
               for r in rows):
            raise ValueError('label provenance outside its training role')
        branches[branch] = [rows[i] for i in indices]
        configs[branch] = meta['model_config']
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parent
    files = [Path(__file__), *(repository/'dinoexp').glob('evidence_*.py')]
    result = dict(status='running', scope='same 16 training states, optimizer dynamics only',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  indices=indices, source_hashes=baseline['inputs_sha256'],
                  diagnostic_manifest_sha256=sha256(args.diagnostic_run/'run.json'),
                  source_sha256={str(p): sha256(p) for p in files},
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                                   text=True).strip(), runs={})
    save_json(args.output/'run.json', result)
    try:
        for branch, records in branches.items():
            result['runs'][branch] = {}
            for seed in args.seeds:
                modes = result['runs'][branch][str(seed)] = {}
                for mode in ('single', 'joint'):
                    run = fit_updates(records, configs[branch], args.output/branch/f'seed_{seed}'/mode,
                                      mode, steps=args.steps, seed=seed, lr=args.lr,
                                      probe_steps=tuple(sorted({0, args.steps//16, 500, 1000, 1500, args.steps})))
                    modes[mode] = {k: v for k, v in run.items() if k != 'per_step'}
                    if mode == 'single' and seed == baseline['arguments']['seed'] and (
                            args.lr == baseline['arguments']['lr'] and args.steps == baseline['arguments']['steps']):
                        old = torch.load(args.diagnostic_run/branch/'n_16/last.pt', weights_only=True)['state_dict']
                        new = torch.load(args.output/branch/f'seed_{seed}'/mode/'last.pt', weights_only=True)['state_dict']
                        modes[mode]['previous_run_max_parameter_error'] = max(float((old[k]-v).abs().max())
                                                                           for k, v in new.items())
                    if mode == 'joint' and modes['single']['initial_fingerprint'] != run['initial_fingerprint']:
                        raise ValueError('paired runs have different initialization')
                    save_json(args.output/'run.json', result)
        for branch in branches:
            paths = {'memory': args.source_run/branch/'memory/last.pt',
                     'labels': args.source_run/branch/'value/labels.pt'}
            if {k: sha256(p) for k, p in paths.items()} != result['source_hashes'][branch]:
                raise RuntimeError('input files changed during experiment')
        render(result, args.output)
        result['status'] = 'complete'
        save_json(args.output/'run.json', result)
        print(f'Complete: {args.output}', flush=True)
    except Exception as error:
        result.update(status='failed', error=str(error))
        save_json(args.output/'run.json', result)
        raise


if __name__ == '__main__':
    main()
