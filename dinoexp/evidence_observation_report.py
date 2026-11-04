"""Static scientific report for the controlled observation-input experiment."""

from pathlib import Path


def render_report(result, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    output = Path(output)
    lines = ['# 全状态更新与真实观察价值诊断', '',
             '观察特征条件是特权诊断，不能作为可部署飞行策略成绩。'
             '开发数据来自以前使用过的位置，车辆身份重合、参考背景未隔离；不代表独立新车辆测试。', '',
             f'训练状态：{result["selection"]["raw_states"]} 条去重后 '
             f'{result["selection"]["unique_states"]} 条，未按收益筛选。', '',
             '| 分支 | seed | 条件 | 训练加权 MSE | 训练 regret | 开发 regret | 开发随机 regret |',
             '|---|---:|---|---:|---:|---:|---:|']
    colors = ['#848484', '#2966b4', '#a87619', '#16815c']
    branches = list(result['runs'])
    fig, axes = plt.subplots(2, len(branches), figsize=(6*len(branches), 7), squeeze=False)
    work_lines = ['', '## 相同训练状态处理数（同为偶数阶段）', '',
                  '| 分支 | seed | 条件 | 更新数 | 状态前向数 | 加权 MSE | regret |',
                  '|---|---:|---|---:|---:|---:|---:|']
    for col, branch in enumerate(branches):
        for seed, modes in result['runs'][branch].items():
            for color, (condition, fit) in zip(colors, modes.items(), strict=True):
                train, dev = fit['training'], fit['development']
                dev_text = f'{dev["mean_action_regret"]:.6g} | {dev["random_mean_regret"]:.6g}' if dev else '— | —'
                lines.append(f'| {branch} | {seed} | {condition} | {train["weighted_mse"]:.6g} | '
                             f'{train["mean_action_regret"]:.6g} | {dev_text} |')
                point = fit['matched_work']
                if point:
                    work_lines.append(f'| {branch} | {seed} | {condition} | {point["step"]} | '
                                      f'{point["training_state_forwards"]} | {point["weighted_mse"]:.6g} | '
                                      f'{point["mean_action_regret"]:.6g} |')
                for row, metric in enumerate(('weighted_mse', 'mean_action_regret')):
                    axes[row, col].plot([v['step'] for v in fit['curve']],
                                        [v[metric] for v in fit['curve']], color=color,
                                        label=condition+f' (seed {seed})', linewidth=1.3)
        axes[0, col].set_title(branch+' | all training states')
        axes[0, col].set_yscale('log')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            ax.set_xlabel('Optimizer updates (different state-processing budgets)')
            ax.legend(fontsize=7)
    axes[0, 0].set_ylabel('Weighted MSE (log axis)')
    axes[1, 0].set_ylabel('Mean action regret')
    fig.suptitle('Frozen upstream | real future inputs: privileged diagnostic only')
    fig.tight_layout()
    fig.savefig(output/'training_curves.png', dpi=150)
    fig.savefig(output/'training_curves.svg')
    plt.close(fig)
    lines += work_lines + ['', '![训练曲线](training_curves.png)', '',
                           'joint 每次前向全部训练状态，single 每次一个。等更新数不等于等算力；'
                           '相同状态前向数也不等于相同 FLOPs 或运行时间。', '',
                           '开发只评估最终固定步数的检查点。每个条件的 fit.json 保留逐状态预测、'
                           '动作标签、收益明显子集和按拍摄配置聚合的指标。失败、负收益和近零收益动作均保留。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
