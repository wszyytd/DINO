"""Portable photo audit and standalone scientific learning curves."""

from collections import Counter
import html
from pathlib import Path

import numpy as np
from PIL import Image

from .local_pairs import sha256


def audit_summary(rows):
    models = sorted(rows[0]['models'])
    result = {}
    for model in models:
        values = [r['models'][model] for r in rows]
        gains = np.array([r['gain'] for r in values])
        result[model] = dict(actions=len(values), transitions=dict(Counter(
            r['transition'] for r in values)), gain_quantiles=np.quantile(
                gains, [0, .1, .5, .9, 1]).tolist(),
            positive_over_001=int((gains > .01).sum()),
            negative_under_minus001=int((gains < -.01).sum()),
            absolute_gain_under_001=int((abs(gains) <= .01).sum()))
    return result


def render_audit(rows, states, output):
    output = Path(output)
    assets = output/'assets'
    assets.mkdir(exist_ok=True)
    copied = {}

    def picture(path, label):
        if not path:
            return '<span>无有效图片</span>'
        if path not in copied:
            digest = sha256(path)
            relative = f'assets/{digest}.jpg'
            with Image.open(path) as source:
                preview = source.convert('RGB')
                preview.thumbnail((360, 240))
                preview.save(output/relative, quality=85)
            copied[path] = relative
        return (f'<figure><img loading="lazy" src="{copied[path]}" alt="{html.escape(label)}">'
                f'<figcaption>{html.escape(label)}</figcaption></figure>')

    cards = []
    for row in rows:
        state = states[str(row['record_index'])]
        tags = ' '.join(f'{model}:{value["transition"]}' for model, value in row['models'].items())
        lines = []
        for model, value in row['models'].items():
            def pred(key, item=value, candidates=state['candidate_ids']):
                p = item[key]
                return '并列' if p is None else html.escape(candidates[p])
            lines.append(f'<tr><td>{html.escape(model)}</td>'
                         f'<td>{pred("before_prediction")} → {pred("after_prediction")}</td>'
                         f'<td>{value["before_margin"]:+.5f} → {value["after_margin"]:+.5f}</td>'
                         f'<td>{value["gain"]:+.5f}</td><td>{value["transition"]}</td></tr>')
        history = ''.join(picture(o['source'], f'{candidate}: {o["id"]}')
                          for candidate, group in zip(state['candidate_ids'], state['history'], strict=True)
                          for o in group)
        cards.append(f'<article data-tags="{html.escape(tags)}">'
                     f'<h2>{html.escape(row["episode_id"])} · 状态 {row["record_index"]} · '
                     f'{html.escape(row["action_id"])} / {html.escape(row["view"])}</h2>'
                     f'<p>正确目标：{html.escape(state["candidate_ids"][state["target"]])}；'
                     f'本次观察候选：{html.escape(row["candidate_id"])}。'
                     f'位置 64 / 训练数据；真实目标只用于审计和标签。</p>'
                     f'<div class="photos">{picture(state["reference"], "参考图")}'
                     f'{picture(row["source"], "加入的新图")}</div>'
                     '<table><thead><tr><th>匹配器</th><th>预测前后</th><th>正确目标 margin 前后</th>'
                     f'<th>收益</th><th>变化</th></tr></thead><tbody>{"".join(lines)}</tbody></table>'
                     f'<details><summary>查看已有观察（{sum(len(h) for h in state["history"])} 张）</summary>'
                     f'<div class="photos">{history}</div></details></article>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>真实观察收益审计</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f5f6f8;color:#15202b;margin:24px auto;max-width:1150px;padding:0 20px}
h1{font-size:26px}h2{font-size:17px}article{background:white;border:1px solid #dce1e7;border-radius:10px;padding:18px;margin:20px 0}
.photos{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;max-width:360px}img{max-width:100%;height:180px;object-fit:contain;background:#edf0f3}figcaption{font-size:12px;word-break:break-all}
table{width:100%;border-collapse:collapse;margin:16px 0}td,th{text-align:left;border-bottom:1px solid #e1e4e8;padding:7px}select{padding:7px;margin:8px}header{position:sticky;top:0;background:#f5f6f8;z-index:2;padding:8px 0}details{margin-top:10px}
</style><h1>真实观察收益审计</h1><p>同一批训练状态和真实后续照片，对照冻结 CLS 均值及两个学习聚合器。收益是各自分数变化，不能跨匹配器直接比较大小；车辆身份标注用于判断对错。当前照片为缩略预览。</p>
<p>corrected：纠正误判；introduced_error：引入误判；stayed_correct：仍正确；stayed_incorrect：仍未正确。</p>
<header><label>模型 <select id="model"><option value="">全部</option><option>cls_mean</option><option>alpha_0</option><option>alpha_0.1</option></select></label>
<label>变化 <select id="kind"><option value="">全部</option><option>corrected</option><option>introduced_error</option><option>stayed_correct</option><option>stayed_incorrect</option></select></label><span id="count"></span></header>'''
    page += '\n'.join(cards)
    page += '''<script>
const cards=[...document.querySelectorAll('article')],model=document.querySelector('#model'),kind=document.querySelector('#kind');
function filter(){let n=0;for(const card of cards){const ok=card.dataset.tags.split(' ').some(t=>{const [m,k]=t.split(':');return(!model.value||m===model.value)&&(!kind.value||k===kind.value)});card.hidden=!ok;if(ok)n++;}document.querySelector('#count').textContent=`显示 ${n} / ${cards.length} 条`;}
model.onchange=kind.onchange=filter;filter();</script></html>'''
    (output/'audit.html').write_text(page, encoding='utf-8')
    return len(copied)


def render_curves(fits, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    sizes = sorted({int(size) for branch in fits.values() for size in branch})
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.5*len(sizes), 7), squeeze=False)
    for column, size in enumerate(sizes):
        for name, branch in fits.items():
            curve = branch[str(size)]['curve']
            steps = [v['step'] for v in curve]
            axes[0, column].plot(steps, [max(v['mse'], 1e-12) for v in curve], label=name)
            axes[1, column].plot(steps, [v['mean_action_regret'] for v in curve], label=name)
        axes[0, column].set_title(f'{size} fixed training states')
        axes[0, column].set_yscale('log')
        for row in range(2):
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_xlabel('Optimizer updates')
            axes[row, column].legend()
    axes[0, 0].set_ylabel('MSE (unscaled, log axis)')
    axes[1, 0].set_ylabel('Mean action regret')
    fig.suptitle('Fixed-state value fitting | original inputs and architecture | training only')
    fig.tight_layout()
    fig.savefig(Path(output)/'fit_curves.png', dpi=160)
    fig.savefig(Path(output)/'fit_curves.svg')
    plt.close(fig)


def render_summary(result, output):
    selection = result['selection']
    lines = ['# 证据收益审计与固定状态拟合', '',
             f'状态：{result["status"]}。范围：位置 64 的价值训练数据，仅用于诊断。', '',
             f'原始 {selection["raw_states"]} 个状态，去重后 {selection["unique_states"]} 个。'
             f'两分支的动作收益范围都至少为 {selection["min_range"]} 的状态有 '
             f'{selection["informative_states"]} 个。拟合样本从这些状态中按固定种子和任务轮转选取。', '',
             '选择范围：'+', '.join(selection['eligible_episodes']), '',
             '[查看参考图、已有观察与新图](audit.html)', '',
             '## 逐动作真实更新', '',
             '| 匹配器 | 纠正误判 | 引入误判 | 仍正确 | 仍未正确 |',
             '|---|---:|---:|---:|---:|']
    for name, value in result['audit_summary'].items():
        t = value['transitions']
        lines.append(f'| {name} | {t.get("corrected", 0)} | {t.get("introduced_error", 0)} | '
                     f'{t.get("stayed_correct", 0)} | {t.get("stayed_incorrect", 0)} |')
    lines += ['', '每行使用相同历史及动作，但匹配器初始正确率不同；绝对纠错数量不代表公平的优劣排名。'
              '多个动作共享同一状态，不能视为独立场景。CLS 与学习匹配器分数尺度不同，不直接比较收益大小。', '',
              '重算标签的最大绝对误差：'+str(result['max_label_error']), '',
              '## 固定状态训练', '',
              '| 分支 | 状态数 | 最终 MSE | 全零 MSE | 最终 regret | 随机 regret | 明显差异排序准确率 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, sizes in result['fits'].items():
        for size, fit in sizes.items():
            row = fit['curve'][-1]
            lines.append(f'| {name} | {size} | {row["mse"]:.6g} | {row["zero_predictor_mse"]:.6g} | '
                         f'{row["mean_action_regret"]:.6g} | {row["random_mean_action_regret"]:.6g} | '
                         f'{row["meaningful_pair_accuracy"]:.1%} |')
    lines += ['', '![固定集合上的训练曲线](fit_curves.png)', '',
              f'学习率 {result["arguments"]["lr"]}；每个规模重新初始化，'
              f'训练 {result["arguments"]["steps"]} 步。采用最后检查点，不选曲线最低点。'
              '明显差异排序只比较真实收益相差超过 0.01 的动作；regret 使用原始实际差值。', '',
              '模型、损失、AdamW 权重衰减、梯度裁剪和困难状态抽样沿用原训练设定。'
              '训练标签已经参与样本筛选，这里衡量拟合能力，不衡量泛化或实际飞行效率。', '',
              '各 n_*/fit.json 保存每步梯度、状态抽样、固定集合评估、最终预测和参数更新量。']
    (Path(output)/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
