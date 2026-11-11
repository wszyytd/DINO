"""Self-contained local HTML evidence viewer and paired-seed tables (no network)."""
import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .core import checked_path
from .region_ab_data import read_json, write_json, digest_json, sha
from .region_geometry import clip_box


def proposal_preview(directory, image, rows):
    """Early geometric preview: raw RGB, final boxes and raw-grid diagnostic values."""
    directory = Path(directory)
    for r in rows:
        name = digest_json(r['id'])
        picture = image.copy(); draw = ImageDraw.Draw(picture)
        for c in r['candidates']:
            draw.rectangle(c['box'], outline='#ff3860', width=3)
            draw.text((c['box'][0]+3, c['box'][1]+3), str(c['candidate_id']), fill='#ffffff', stroke_width=2, stroke_fill='#000000')
        picture.save(directory/(name+'.png'))
        write_json(directory/(name+'.json'), r)
    files = sorted(directory.glob('*.png'))
    body = ''.join(f'<p>{html.escape(p.stem)}</p><img style="max-width:95vw" src="{p.name}">' for p in files)
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>前3帧候选坐标检查</title>'+body, encoding='utf-8')


def aggregate(run):
    result = {}
    keys = ['conditional_accuracy', 'ranking_accuracy', 'ranking_with_distractors_accuracy',
            'downstream_accuracy', 'visible_recall', 'absent_rejection', 'end_to_end_accuracy']
    splits = sorted({s for r in run['results'] for s in r['splits']})
    for split in splits:
        result[split] = {}
        for key in keys:
            values = {v: {r['seed']: r['splits'][split]['metrics'][key] for r in run['results'] if r['variant'] == v} for v in ('A', 'B')}
            row = {}
            for v in ('A', 'B'):
                x = [value for value in values[v].values() if value is not None]
                row[v] = dict(per_seed=values[v], mean=float(np.mean(x)) if x else None,
                              population_std=float(np.std(x)) if x else None,
                              minimum=min(x) if x else None, maximum=max(x) if x else None)
            differences = [values['B'][s]-values['A'][s] for s in values['A']
                           if values['A'][s] is not None and values['B'].get(s) is not None]
            row['B_minus_A'] = dict(values=differences, mean=float(np.mean(differences)) if differences else None,
                                   population_std=float(np.std(differences)) if differences else None,
                                   positive_seeds=sum(x > 0 for x in differences), seeds=len(differences))
            result[split][key] = row
    return result


def report(args):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    run, meta = read_json(output/'training/run.json'), read_json(output/'prepared.json')
    if run.get('status') != 'complete' or run['prepared_sha256'] != sha(output/'prepared.json'):
        raise ValueError('need complete training tied to this prepared cache')
    for name, digest in meta['artifacts'].items():
        if sha(output/name) != digest:
            raise ValueError('changed report input: '+name)
    directory = output/'report'
    if directory.exists():
        raise ValueError('report exists; preserve previous report')
    directory.mkdir(); (directory/'assets').mkdir()
    rows = read_json(output/'proposals/records.json')['records']; labels = read_json(output/'labels.json')
    predictions = {}
    csv_rows = []
    for result in run['results']:
        v, seed = result['variant'], result['seed']
        folder = output/'training'/f'{v}_seed{seed}'
        if sha(folder/'last.pt') != result['checkpoint_sha256']:
            raise ValueError('checkpoint changed before report')
        predictions.setdefault(str(seed), {}).setdefault(v, {})
        for split, record in result['splits'].items():
            if sha(folder/(split+'.json')) != record['evaluation_sha256']:
                raise ValueError('evaluation predictions changed before report')
            data = read_json(folder/(split+'.json'))
            if data['metrics'] != record['metrics']:
                raise ValueError('evaluation metrics changed before report')
            predictions[str(seed)][v].update({r['id']: r for r in data['predictions']})
            csv_rows.append(dict(variant=v, seed=seed, split=split, **record['metrics']))
    with (directory/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    summary = aggregate(run); write_json(directory/'summary.json', summary)
    copied, entries = {}, []
    def asset(path, digest):
        if digest not in copied:
            source = checked_path(root, path)
            if sha(source) != digest:
                raise ValueError('viewer RGB/reference hash changed')
            relative = 'assets/'+digest+source.suffix
            shutil.copyfile(source, directory/relative); copied[digest] = relative
        return copied[digest]
    for r in rows:
        item = dict(r, label=labels[r['id']], image_asset=asset(r['image'], r['image_sha256']),
                    reference_asset=asset(r['reference'], r['reference_sha256']), crops=[])
        with Image.open(directory/item['image_asset']) as im:
            for c in r['candidates']:
                if clip_box(c['box'], im.size) is None:
                    item['crops'].append(None)
                    continue
                name = 'assets/'+digest_json([r['image_sha256'], c['box']])+'_crop.png'
                if not (directory/name).exists():
                    im.convert('RGB').crop(c['box']).save(directory/name)
                item['crops'].append(name)
        entries.append(item)
    payload = dict(rows=entries, predictions=predictions, summary=summary,
                   teacher=read_json(output/'training/teacher_metrics.json'), preparation=meta['preparation'],
                   profile=read_json(output/'online_profile.json') if (output/'online_profile.json').exists() else None,
                   config=meta['config'], scope=meta['split']['scope'])
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (directory/'index.html').write_text(HTML.replace('__PAYLOAD__', serialized), encoding='utf-8')
    lines = ['# 老师区域 A/B 单帧对照', '', meta['split']['scope'], '',
             'A：整图编码后矩形池化；B：同框原始RGB裁图独立编码。以下仅比较新A/B。', '',
             '标签阈值固定；多正候选允许；任意提取失败按链路失败保留分母。标准差为三个种子的总体标准差。', '',
             '|划分|指标|A 均值±标准差|B 均值±标准差|B−A|B提升种子数|', '|---|---|---|---|---|---|']
    def fmt(value):
        return 'N/A' if value is None else f'{100*value:.2f}%'
    for split, metrics in summary.items():
        for key, m in metrics.items():
            lines.append(f'|{split}|{key}|{fmt(m["A"]["mean"])} ± {fmt(m["A"]["population_std"])}|'
                         f'{fmt(m["B"]["mean"])} ± {fmt(m["B"]["population_std"])}|{fmt(m["B_minus_A"]["mean"])}|'
                         f'{m["B_minus_A"]["positive_seeds"]}/{m["B_minus_A"]["seeds"]}|')
    lines += ['', '## 如何解释', '',
              '先看 teacher_metrics.json 的 confirmable_covered / physical_visible。它是可确认区域召回；covered_50_percent 只表示像素覆盖，要同时看 giant、ambiguous 和面积分布。',
              '再看 conditional_accuracy 及 ranking_accuracy。前者含 Null 拒绝，后者只看候选排序；有干扰候选的排序另列，避免只有一个候选导致虚高。',
              '若 B 在三个种子上的条件识别和整链指标稳定优于 A，才有理由优先接 B。若区域召回很低，应先解决定位；若排序尚可但 false_null 多，应检查拒绝边界。',
              '本轮不检验时序融合收益，也没有独立场景泛化结论。B 改变了尺度、上下文和位置分布，不能从本实验判断唯一原因。', '',
              '缓存后网络耗时见 training/run.json；完整在线单帧基准见 online_profile.json（若尚未运行，不用缓存耗时冒充在线速度）。',
              '逐帧查看：打开同目录 index.html；每一种子、划分、错误类型均可筛选，空类别显示 0。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Report:', directory/'index.html', flush=True)


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>老师区域 A/B 对照</title><style>
body{font:15px system-ui;margin:20px;background:#f6f7f9;color:#17212b}h1{font-size:24px}select,button{font:inherit;padding:6px;margin:4px}section{background:white;padding:16px;border-radius:8px;margin:12px 0} .views{display:grid;grid-template-columns:1fr 2fr 2fr;gap:12px}.views img,.views canvas{width:100%;object-fit:contain}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd2dc;padding:8px;text-align:left}.crop{width:180px;max-height:150px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere}.ok{background:#e4f4df}.chosen{font-weight:bold;color:#125b9c}.bad{background:#fce5e5}summary{cursor:pointer}@media(max-width:900px){.views{grid-template-columns:1fr}body{margin:8px}}
</style><h1>同一批老师区域：整图编码 A / 裁图编码 B</h1><p id="scope"></p>
<details><summary>总体指标、来源及耗时</summary><pre id="overview"></pre></details>
<section><label>种子<select id="seed"></select></label><label>划分<select id="split"></select></label><label>类别<select id="category"></select></label>
<button id="prev">上一张</button><button id="next">下一张</button><select id="sample"></select><p id="counts"></p></section>
<section id="detail"><h2 id="title"></h2><p id="verdict"></p><div class="views"><div>参考图<img id="ref"></div><div>原图＋老师最终框<canvas id="boxes"></canvas><a id="native" target="_blank">查看原始RGB</a></div><div>老师原始空间输出可视化<canvas id="heat"></canvas><p>仅显示时按当前峰值着色；区域生成使用保存的原始数值和固定规则。</p></div></div>
<h3>候选、真实对应及两组输出</h3><table><thead><tr><th>候选/实际RGB裁图</th><th>坐标、监督标签</th><th>A logit / 概率</th><th>B logit / 概率</th></tr></thead><tbody id="scores"></tbody></table>
<details><summary>原始空间数值、合并记录与标签</summary><pre id="raw"></pre></details></section>
<script>const DATA=__PAYLOAD__;
const $=id=>document.getElementById(id), names={all:'全部',both:'两组都对',a:'A对B错',b:'A错B对',neither:'两组都错',miss:'老师无可确认目标框',absent:'目标缺席',small:'碎片候选',multi:'多车/未知歧义'};
const option=(v,t)=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o};
Object.keys(DATA.predictions).forEach(s=>$('seed').append(option(s,s)));
[...new Set(DATA.rows.map(r=>r.split))].sort().forEach(s=>$('split').append(option(s,s)));
if([...$('split').options].some(o=>o.value==='probe'))$('split').value='probe';else if([...$('split').options].some(o=>o.value==='test'))$('split').value='test';
Object.entries(names).forEach(([k,n])=>$('category').append(option(k,n)));
$('scope').textContent=DATA.scope;
$('overview').textContent=JSON.stringify({summary:DATA.summary,teacher:DATA.teacher,preparation:DATA.preparation,online:DATA.profile,config:DATA.config},null,2);
const pred=(r,v)=>DATA.predictions[$('seed').value][v][r.id];
const correct=(r,p)=>!p.extraction_failure&&(r.label.physical_visible?r.label.positives.includes(p.choice):p.choice===null);
function category(r,k){let a=correct(r,pred(r,'A')),b=correct(r,pred(r,'B'));return {all:true,both:a&&b,a:a&&!b,b:!a&&b,neither:!a&&!b,miss:r.label.physical_visible&&!r.label.positives.length,absent:!r.label.physical_visible,small:r.label.candidates.some(c=>c.fragment),multi:r.label.candidates.some(c=>c.ambiguous)}[k]}
let visible=[];let generation=0;
function refresh(){let rows=DATA.rows.filter(r=>r.split===$('split').value);$('counts').textContent=Object.entries(names).map(([k,n])=>n+' '+rows.filter(r=>category(r,k)).length).join('；')+'。数量为0表示本次不存在该类样本。';visible=rows.filter(r=>category(r,$('category').value));$('sample').replaceChildren(...visible.map((r,i)=>option(i,r.id)));render()}
function render(){let r=visible[Number($('sample').value)];$('detail').hidden=!r;if(!r)return;let gen=++generation;let a=pred(r,'A'),b=pred(r,'B');$('title').textContent=r.id;$('ref').src=r.reference_asset;$('native').href=r.image_asset;
const choice=p=>p.choice===null?'Null':'候选'+p.choice;
$('verdict').textContent=`物理可见：${r.label.physical_visible}；可确认正候选：[${r.label.positives}]；老师Null：${r.teacher_null}（不输入匹配器）。A：${choice(a)}，链路${correct(r,a)?'正确':'失败'}；B：${choice(b)}，链路${correct(r,b)?'正确':'失败'}；提取失败 A=${a.extraction_failure}, B=${b.extraction_failure}`;
let img=new Image();img.onload=()=>{if(gen!==generation)return;for(let name of ['boxes','heat']){let c=$(name);c.width=img.width;c.height=img.height;let ctx=c.getContext('2d');ctx.drawImage(img,0,0);if(name==='boxes'){ctx.lineWidth=3;ctx.font='bold 22px sans-serif';r.candidates.forEach((c,i)=>{let [x,y,x1,y1]=c.box;ctx.strokeStyle=r.label.positives.includes(i)?'#50ff50':'#ff3860';ctx.strokeRect(x,y,x1-x,y1-y);ctx.fillStyle='#fff';ctx.strokeStyle='#111';ctx.strokeText(String(i),x+4,y+24);ctx.fillText(String(i),x+4,y+24)})}else{let mx=Math.max(...r.spatial.flat(),1e-12);r.spatial.forEach((row,y)=>row.forEach((v,x)=>{ctx.fillStyle=`rgba(255,40,0,${.75*v/mx})`;ctx.fillRect(x*c.width/row.length,y*c.height/r.spatial.length,c.width/row.length+1,c.height/r.spatial.length+1)}))}}};img.src=r.image_asset;
$('scores').replaceChildren();for(let i=0;i<=r.candidates.length;i++){let tr=document.createElement('tr');let last=i===r.candidates.length;let td=document.createElement('td');td.textContent=last?'Null':'候选 '+i;if(!last){let im=document.createElement('img');im.src=r.crops[i];im.className='crop';td.append(document.createElement('br'),im)}tr.append(td);let label=document.createElement('td');label.textContent=last?'当前候选集合中没有可确认目标框':JSON.stringify({box:r.candidates[i].box,...r.label.candidates[i]});tr.append(label);for(let p of [a,b]){let cell=document.createElement('td');cell.textContent=p.scores[i].toFixed(5)+' / '+(p.probabilities[i]*100).toFixed(2)+'%';if((last&&p.choice===null)||p.choice===i)cell.className='chosen';tr.append(cell)}if(r.label.positives.includes(i)||last&&!r.label.positives.length)tr.className='ok';$('scores').append(tr)}$('raw').textContent=JSON.stringify(r,null,2)}
for(let id of ['seed','split','category'])$(id).onchange=refresh;$('sample').onchange=render;
$('prev').onclick=()=>{if(visible.length){$('sample').selectedIndex=($('sample').selectedIndex-1+visible.length)%visible.length;render()}};
$('next').onclick=()=>{if(visible.length){$('sample').selectedIndex=($('sample').selectedIndex+1)%visible.length;render()}};refresh();
</script></html>'''
