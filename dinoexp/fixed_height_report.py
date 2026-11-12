"""Read-only reports for the supplied-box orbit test; no neural-network imports."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .core import checked_path
from .fixed_height_data import load_plan
from .region_ab_data import read_json, sha, write_json


def report(args, preview=False):
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    plan = load_plan(args); run = None; values = {}
    if not preview:
        run = read_json(output/'evaluation/run.json')
        if (run.get('kind') != 'fixed_height_test_evaluation_v1' or run['status'] != 'complete'
                or run['plan_sha256'] != sha(output/'plan.json')
                or run['prepared_sha256'] != sha(output/'prepared.json') or run['source'] != plan['source']):
            raise ValueError('complete fixed-checkpoint evaluation required')
        for result in run['results']:
            file = checked_path(output, result['file'])
            if sha(file) != result['sha256']:
                raise ValueError('evaluation scores changed')
            values[f'{result["method"]}_{result["seed"]}'] = read_json(file)['predictions']
    directory = output/('preview' if preview else 'report')
    if directory.exists():
        raise ValueError('viewer exists; preserve it and choose a new output')
    (directory/'assets').mkdir(parents=True)

    def copy(relative, digest):
        target = directory/'assets'/(digest+Path(relative).suffix)
        if not target.exists():
            shutil.copyfile(checked_path(root, relative), target)
        return 'assets/'+target.name

    cases, records = [], {}
    for original in plan['cases']:
        case = dict(original)
        case['reference_asset'] = copy(case['reference'], case['reference_sha256'])
        cases.append(case)
    for original in plan['records']:
        record = dict(original)
        record['image_asset'] = copy(record['image'], record['image_sha256'])
        record['observations'] = []
        with Image.open(checked_path(root, record['image'])) as im:
            image = im.convert('RGB')
        for original_obj in original['observations']:
            obj = dict(original_obj); crop = 'assets/'+obj['key']+'.png'
            if not (directory/crop).exists():
                image.crop(obj['box']).save(directory/crop)
            obj['asset'] = crop; record['observations'].append(obj)
        records[record['id']] = record
    payload = dict(cases=cases, records=records, results=values, counts=plan['counts'], preview=preview)
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    (directory/'index.html').write_text(HTML.replace('__DATA__', encoded), encoding='utf-8')
    lines = ['# 固定高度旧采集：现有模型供框测试', '',
             '车辆区域和跨图身份由采集标注提供，查询输入从原整图按投影框重新裁出。', '',
             '测试识别与时序融合；未测试老师定位、自动跟踪或主动动作选择。没有实例掩膜，未计算物理可见率。', '',
             '每组参考图固定，按采集的 24 个角度顺序更新，每组开始清空状态。refA/refB 是两次采集，未当作同一批 RGB 配对。', '',
             f'数据：{plan["counts"]["cases"]} 组，{plan["counts"]["observations"]} 个顺序判断，'
             f'{plan["counts"]["full_images"]} 张整图，{plan["counts"]["supplied_crops"]} 张保存裁图。', '',
             '所有参考的目标均在采集候选中，因此无法衡量“目标不在场时正确拒绝”。Null 按原模型分数参与竞争。', '',
             '采集最初用于目视检查，以下测试不代表严格独立场景或未见车型的泛化结论。', '']
    summary = dict(status='preview_only' if preview else 'complete', protocol=plan['protocol'], counts=plan['counts'])
    if preview:
        lines += ['当前仅完成数据核对和图片预览；未提取特征、未运行模型，没有准确率结果。']
    else:
        lines += ['单图对照使用同一个检查点，每张图单独清空状态；它与另训的 last 方法不同。', '',
                  '| 方法 | 种子 | 判断数 | 正确率 | 两车排序正确 | 错车 | Null | 同权重单图正确率 | 历史救回/伤害 | 最后一步 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        csv_rows = []
        for r in run['results']:
            m = r['metrics']['all_prefixes']; final = r['metrics']['final_step']
            lines.append(f'| {r["method"]} | {r["seed"]} | {m["count"]} | {100*m["accuracy"]:.2f}% | '
                         f'{100*m["target_ranking"]:.2f}% | {m["wrong_vehicle"]} | {m["null"]} | '
                         f'{100*m["single_frame_accuracy"]:.2f}% | {m["history_helped"]}/{m["history_harmed"]} | '
                         f'{100*final["accuracy"]:.2f}% |')
            csv_rows.append(dict(method=r['method'], seed=r['seed'], **m, final_accuracy=final['accuracy']))
        lines += ['', '## 三个种子的平均结果', '', '| 方法 | 顺序正确率均值±标准差 | 同权重单图均值 | 最后一步均值 |',
                  '|---|---:|---:|---:|']
        summary['methods'] = {}
        for method in ('last', 'mean', 'memory'):
            selected = [r['metrics'] for r in run['results'] if r['method'] == method]
            acc = [r['all_prefixes']['accuracy'] for r in selected]
            single = float(np.mean([r['all_prefixes']['single_frame_accuracy'] for r in selected]))
            final = float(np.mean([r['final_step']['accuracy'] for r in selected]))
            summary['methods'][method] = dict(accuracy_mean=float(np.mean(acc)), accuracy_std=float(np.std(acc)),
                                             single_frame_accuracy_mean=single, final_accuracy_mean=final)
            lines.append(f'| {method} | {100*np.mean(acc):.2f}±{100*np.std(acc):.2f}% | {100*single:.2f}% | {100*final:.2f}% |')
        lines += ['', '完整分组、每车、每角度和逐条对照在 evaluation 下的 JSON。', '',
                  '一组内各角度相关，最后一步只有 12 组；不能把多步判断当作数百个独立场景。', '',
                  '编码耗时见 prepared.json，匹配耗时是缓存特征上的批量测试，不能当作在线端到端延迟。']
        with (directory/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    write_json(directory/'summary.json', summary)
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Fixed-height viewer:', directory/'index.html', flush=True)
    return summary


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>旧采集：车辆识别与逐步融合测试</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 18px;background:#f4f6f8;color:#18232d}select,button,input{font:inherit;margin:4px;padding:5px}img{object-fit:contain;background:#e9edf1;max-width:100%}.layout{display:grid;grid-template-columns:3fr 1fr;gap:18px}.card{background:white;border:1px solid #ccd5de;border-radius:8px;padding:14px;margin:12px 0}#whole{width:100%;max-height:650px}#ref{max-height:220px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #dde3e9;text-align:left}.good{color:#087442}.bad{color:#b12c30}.gallery{display:flex;overflow:auto;gap:6px}.gallery figure{margin:0;flex:0 0 100px}.gallery img{width:100px;height:70px}small{word-break:break-all}#compare{display:flex;gap:12px;flex-wrap:wrap}@media(max-width:800px){.layout{grid-template-columns:1fr}}
</style><h1>车辆识别与逐步融合</h1><p>供框与身份对应来自采集标注。查询裁图按原整图和框重新生成。当前页面不评价老师定位、自动跟踪或物理可见性；这组数据没有目标缺席样例。</p>
<div id="status" class="card"></div><label>采集组 <select id="case"></select></label><label>种子 <select id="seed"><option>7</option><option>17</option><option>27</option></select></label><label>查看状态 <select id="method"><option>memory</option><option>last</option><option>mean</option></select></label><br>
<button id="prev">上一步</button><button id="next">下一步</button><button id="play">播放</button><input id="step" type="range" min="0" value="0"><span id="caption"></span>
<div class="layout"><div><img id="whole"><div id="compare"></div></div><div class="card"><h3>固定参考图</h3><img id="ref"><div id="target"></div></div></div>
<div class="card"><h3>当前候选与状态</h3><div id="cars"></div></div><div class="card"><h3>截至当前的历史裁图</h3><div id="history"></div></div>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let pos=0,timer=null;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='播放'}
function currentCase(){return DATA.cases[Number($('case').value)||0]}
function prediction(method,c,t){return (DATA.results[method+'_'+$('seed').value]||[]).find(r=>r.case_id===c.id&&r.step===t)}
function fmt(x){return x===undefined||x===null?'—':Number(x).toFixed(4)}
function render(){const c=currentCase();pos=Math.max(0,Math.min(pos,c.records.length-1));const r=DATA.records[c.records[pos]],p=prediction($('method').value,c,pos);$('step').max=c.records.length-1;$('step').value=pos;$('caption').textContent=`第 ${pos+1}/${c.records.length} 步 · ${r.angle}°`;$('whole').src=r.image_asset;$('ref').src=c.reference_asset;$('target').textContent='参考车辆 '+c.target_identity;$('status').textContent=DATA.preview?'仅数据预览：尚未提取特征或执行模型。':'固定九个检查点，逐步更新；同权重单图对照每张图清空状态。';
$('compare').innerHTML=['last','mean','memory'].map(m=>{const v=prediction(m,c,pos);return `<div class="card"><b>${m}</b><div class="${v?(v.correct?'good':'bad'):''}">${v?(v.choice_identity??'Null'):'未执行'}</div><div>同权重单图：${v?(v.single_frame.choice_identity??'Null'):'—'}</div></div>`}).join('');
$('cars').innerHTML='<table><tr><th>车辆</th><th>当前裁图</th><th>分数/概率</th><th>写入次数</th><th>状态变化及哈希</th></tr>'+r.observations.map(o=>{const j=p?p.identities.indexOf(o.identity):-1,s=p?p.state[j]:null;return `<tr><td>${esc(o.identity)}${o.identity===c.target_identity?'（目标）':''}</td><td><img src="${o.asset}" height="100"></td><td>${p?fmt(p.scores[j])+' / '+fmt(p.probabilities[j]):'—'}</td><td>${s?s.count:'—'}</td><td>${s?fmt(s.change_l2)+'<br><small>'+s.sha256+'</small>':'—'}</td></tr>`}).join('')+`<tr><td>Null</td><td></td><td>${p?fmt(p.scores[2])+' / '+fmt(p.probabilities[2]):'—'}</td><td colspan="2">采集目标在场，选 Null 计错；不等于已测拒绝能力。</td></tr></table>`;
$('history').innerHTML=c.identities.map(uid=>'<b>'+esc(uid)+'</b><div class="gallery">'+c.records.slice(0,pos+1).map((id,t)=>{const o=DATA.records[id].observations.find(o=>o.identity===uid);return `<figure data-step="${t}"><img src="${o.asset}"><figcaption>${t+1} · ${DATA.records[id].angle}°</figcaption></figure>`}).join('')+'</div>').join('');$('prev').disabled=pos===0;$('next').disabled=pos===c.records.length-1;}
$('case').innerHTML=DATA.cases.map((c,i)=>`<option value="${i}">${esc(c.id)}</option>`).join('');$('case').onchange=()=>{stop();pos=0;render()};$('seed').onchange=$('method').onchange=render;$('step').oninput=()=>{stop();pos=Number($('step').value);render()};$('prev').onclick=()=>{stop();pos--;render()};$('next').onclick=()=>{stop();pos++;render()};$('play').onclick=()=>{if(timer!==null){stop();return}timer=setInterval(()=>{if(pos>=currentCase().records.length-1){stop();return}pos++;render()},800);$('play').textContent='暂停'};render();
</script></html>'''
