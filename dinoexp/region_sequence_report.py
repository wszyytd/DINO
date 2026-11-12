"""Local causal route viewer. Preview is geometry-only; report uses saved predictions."""
import csv
import json
from pathlib import Path
import shutil

from PIL import Image

from .core import checked_path
from .region_ab_data import digest_json, read_json, sha, write_json
from .region_sequence_data import load_plan
from .region_sequence_metrics import aggregate


def report(args, preview=False):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    meta, records, routes = load_plan(output, root)
    run = None
    if not preview:
        run = read_json(output/'training/run.json')
        if (run.get('status') != 'complete' or run['plan_sha256'] != sha(output/'plan.json')
                or run['prepared_sha256'] != sha(output/'prepared.json')):
            raise ValueError('report needs completed sequence training tied to this plan/cache')
    directory = output/('preview' if preview else 'report')
    if directory.exists():
        raise ValueError('viewer exists; preserve it and choose a new output')
    directory.mkdir(); (directory/'assets').mkdir()
    predictions, controls, csv_rows = {}, {}, []
    if run is not None:
        for result in run['results']:
            seed, method = str(result['seed']), result['method']
            folder = output/'training'/f'{method}_seed{seed}'
            if sha(folder/'last.pt') != result['checkpoint_sha256']:
                raise ValueError('checkpoint changed before sequence report')
            table = predictions.setdefault(seed, {}).setdefault(method, {})
            control_table = controls.setdefault(seed, {})
            for split, entry in result['splits'].items():
                if sha(folder/(split+'.json')) != entry['evaluation_sha256']:
                    raise ValueError('sequence predictions changed before report')
                value = read_json(folder/(split+'.json'))
                if value['metrics'] != entry['metrics']:
                    raise ValueError('sequence metric record changed')
                for p in value['predictions']:
                    table.setdefault(p['route_id'], []).append(p)
                for c in value['controls']:
                    control_table.setdefault(c['route_id'], {}).setdefault(str(c['step']), []).append(c)
                csv_rows.append(dict(method=method, seed=seed, split=split,
                                     **{k: v for k, v in value['metrics'].items() if k not in ('counts', 'per_target')}))
            for route in routes:
                ordered = sorted(table[route['id']], key=lambda p: p['step'])
                if [p['step'] for p in ordered] != list(range(len(route['records']))):
                    raise ValueError('viewer must contain every causal route step exactly once')
                table[route['id']] = ordered
        with (directory/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
    copied = {}

    def asset(path, digest):
        if digest not in copied:
            source = checked_path(root, path)
            if sha(source) != digest:
                raise ValueError('viewer image changed')
            relative = 'assets/'+digest+source.suffix
            shutil.copyfile(source, directory/relative); copied[digest] = relative
        return copied[digest]

    entries = {}
    for rid, r in records.items():
        fields = ('id', 'scene', 'frame', 'target_identity', 'size', 'spatial', 'teacher_null',
                  'original_candidates', 'candidates', 'threshold', 'physical_visible', 'physical_pixels',
                  'unassigned_candidate_ids', 'mask_split_candidate_ids')
        item = {k: r[k] for k in fields}
        item.update(image_asset=asset(r['image'], r['image_sha256']),
                    reference_asset=asset(r['reference'], r['reference_sha256']), observations=[])
        with Image.open(directory/item['image_asset']) as im:
            for observation in r['observations']:
                crop = 'assets/'+digest_json([r['image_sha256'], observation['box']])+'_crop.png'
                if not (directory/crop).exists():
                    im.convert('RGB').crop(observation['box']).save(directory/crop)
                item['observations'].append(dict(observation, crop=crop))
        entries[rid] = item
    summary = aggregate(run) if run else None
    payload = dict(preview=preview, records=entries, routes=routes, predictions=predictions, controls=controls,
                   summary=summary, counts=meta['counts'], config=meta['config'], assistance=meta['assistance'],
                   scope=meta['split']['scope'], sampling=meta['sampling'])
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (directory/'index.html').write_text(HTML.replace('__PAYLOAD__', serialized), encoding='utf-8')
    write_json(directory/'summary.json', dict(preview_only=preview, summary=summary, counts=meta['counts'],
                                            plan_sha256=sha(output/'plan.json')))
    fmt = lambda v: '—' if v is None else f'{100*v:.2f}%'
    lines = ['# 连续路线与车辆证据融合', '', meta['split']['scope'], '',
             '**使用仿真掩膜辅助分组、补全可见范围和跨帧对应。这不是自动检测与跟踪系统。**', '',
             '老师热图只产生与参考图相关的区域种子；裁出的原始 RGB 送入冻结编码器；老师分数和身份不作为匹配特征。', '',
             '固定路线每一步都有真实相邻图边，包含中途转向和移动。不是视频采集，也不证明可以无碰撞飞行。', '',
             '所有方法每车占固定位置，车辆暂时消失后保留历史；完全相同的裁图在同一路线内默认只写一次。', '',
             'last：每车最近一次有效写入；mean：不同图片的有效写入取平均；memory：学习如何更新历史。', '',
             '匹配标签表示目标是否已被有效接收；physical 指标保留目标物理出现的真值，漏检不会从该指标中消失。', '',
             '路线窗口互有重叠，前缀不是独立样本。当前仍只有一处布局、两辆车和固定参考图，不能当成独立场景泛化。', '']
    if preview:
        lines += ['这是纯几何预览，尚未加载或运行模型；没有融合分数和效果结论。']
    else:
        lines += ['|划分|方法|已接收目标时选对|两车已有证据时选对|只靠历史时选对|物理出现过目标时选对|全体物理准确率|全体物理恒Null基线|',
                  '|---|---|---|---|---|---|---|---|']
        for split, methods in summary.items():
            for method, values in methods.items():
                fields = ('accepted_target_accuracy', 'two_seen_target_accuracy', 'history_only_target_accuracy',
                          'physical_target_accuracy', 'physical_accuracy', 'always_null_physical_accuracy')
                lines.append('|'+split+'|'+method+'|'+'|'.join(fmt(values[k]['mean']) for k in fields)+'|')
        lines += ['', '表内为种子均值；逐种子数字和标准差见 summary.json，原始分子分母见 training/*/划分.json。', '',
                  'memory 的留出记录还含同权重干预：每步清空、只交换此前历史内容、暂停当前写入、绕过去重强制重复写入。',
                  '这些干预使用正常前缀的副本，保留原始真值，不影响下一步。交换仅在至少两车已有证据时统计。',
                  '重复写入测试刻意绕过去重，检验更新器本身是否漂移，不代表默认输入流程会重复写入。', '',
                  '训练前自动执行 CUDA 前向、反向、候选排列、因果性和无写入不变检查；保存后重放全部留出序列。',
                  '查看 run.json 的 status、checks.json 与 reload_max_abs_score_error，失败时不生成完整报告。',
                  '缓存网络计时不含老师定位、编码、图片读取，不能冒充完整在线耗时。']
    lines += ['', '打开同目录 index.html。“下一步”只沿同一路线推进；换路线或参考回到起点，换种子或方法在同一步对照。',
              '页面的历史裁图只来自当前位置及之前；表格同时显示当前观察、最近观察、历史图片和写入次数。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Sequence viewer:', directory/'index.html', flush=True)
    return directory/'index.html'


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>连续路线与车辆证据融合</title><style>
body{font:15px system-ui;margin:24px;background:#f5f6f8;color:#18222e}h1{font-size:25px}p{line-height:1.65}.notice{border-left:4px solid #b97913;background:#fff3dd;padding:12px}section{background:white;padding:18px;margin:14px 0;border:1px solid #dde1e8;border-radius:8px}button,select{font:inherit;padding:6px;margin:4px}input[type=range]{width:100%;margin:14px 0}.views{display:grid;grid-template-columns:1fr 2fr 2fr;gap:18px}.views img,.views canvas{width:100%;object-fit:contain}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border-bottom:1px solid #d6dce4;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}td img{width:145px;max-width:100%;max-height:125px;object-fit:contain}.history{display:flex;gap:5px;flex-wrap:wrap}.history img{width:78px;height:62px;object-fit:contain}.history a{font-size:12px;text-align:center}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto}.muted{color:#677282}.selected{background:#e8f3ff}.good{color:#157040}.bad{color:#ae2525}#routeMap{width:100%;height:160px}summary{cursor:pointer}@media(max-width:950px){body{margin:8px}.views{grid-template-columns:1fr}table{table-layout:auto}td{padding:5px}}
</style><h1>连续路线与车辆证据融合</h1>
<p class="notice" id="notice"></p><p id="scope"></p>
<details><summary>来源、路线数量和总体结果</summary><pre id="overview"></pre></details>
<section><label>种子<select id="seed"></select></label><label>主要显示<select id="method"><option value="memory">学习融合</option><option value="mean">历史平均</option><option value="last">每车最近写入</option></select></label><label>划分<select id="split"></select></label><label>路线与固定参考<select id="route"></select></label>
<div><button id="prev">上一步</button><button id="play">播放</button><button id="next">下一步</button><span id="stepLabel"></span><input id="step" type="range" min="0" value="0"></div><p id="movement"></p><canvas id="routeMap" width="1000" height="160"></canvas></section>
<section><h2 id="title"></h2><p id="verdict"></p><div class="views"><div>固定参考图<img id="reference"></div><div>仿真辅助分组后的车辆区域<canvas id="image"></canvas><p>同色始终表示同一辆车；粗线表示本方法选中。虚线表示未获准写入。</p><a id="native" target="_blank">打开原始图片</a></div><div>老师热图与转框结果<canvas id="heat"></canvas><p>热图按本图峰值显示。细灰框是放宽后、掩膜分组前的候选；颜色不代表最终选择。</p></div></div></section>
<section><h2>每辆车的当前观察与历史</h2><table><thead><tr><th>固定身份位置／本步状态</th><th>当前裁图／最近有效观察</th><th>此前写入的不同图片</th><th>分数与记忆记录</th></tr></thead><tbody id="cars"></tbody></table><p id="null"></p></section>
<section><h2>同一步的三种方法</h2><table><thead><tr><th>方法</th><th>选择</th><th>与已接收证据标签比较</th></tr></thead><tbody id="compare"></tbody></table><p id="control"></p></section>
<details><summary>当前步的原始记录</summary><pre id="raw"></pre></details>
<script>const DATA=__PAYLOAD__;
const $=id=>document.getElementById(id),names={memory:'学习融合',mean:'历史平均',last:'每车最近写入'},colors=['#157dcb','#bb4e9b','#16866b','#b17c18'];
const option=(v,t)=>{const o=document.createElement('option');o.value=v;o.textContent=t;return o};
const seeds=Object.keys(DATA.predictions);(seeds.length?seeds:['未训练']).forEach(s=>$('seed').append(option(s,s)));
const splits=[...new Set(DATA.routes.map(r=>r.split))];splits.forEach(s=>$('split').append(option(s,s)));$('split').value=splits.includes('probe')?'probe':splits.includes('test')?'test':splits[0];
$('notice').textContent=DATA.preview?'纯几何预览：未运行模型，尚无融合结果。已使用仿真掩膜辅助合并碎片、补全可见范围和跨帧对应。':'本实验使用仿真掩膜辅助分组、补全可见范围和跨帧对应。下方分数来自保存的逐步融合结果，历史图片只显示到当前步。';
$('scope').textContent=DATA.scope+'。路线窗口互有重叠，不能把每个前缀当成独立样本。';
$('overview').textContent=JSON.stringify({counts:DATA.counts,summary:DATA.summary,config:DATA.config,sampling:DATA.sampling},null,2);
let generation=0,timer=null;
const route=()=>DATA.routes.find(r=>r.id===$('route').value),pred=(method,t)=>DATA.predictions[$('seed').value]?.[method]?.[route().id]?.[t];
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='播放'}
function routes(){stop();$('route').replaceChildren(...DATA.routes.filter(r=>r.split===$('split').value).map(r=>option(r.id,r.id)));reset()}
function reset(){stop();$('step').max=route().records.length-1;$('step').value=0;render()}
function label(uid){return uid===null?'Null':uid}
function imgCell(cell,src,text){if(!src){cell.textContent=text;return}const im=document.createElement('img');im.src=src;const p=document.createElement('div');p.textContent=text;cell.append(im,p)}
function render(){const r=route(),t=Number($('step').value),row=DATA.records[r.records[t]],p=pred($('method').value,t),g=++generation;
$('stepLabel').textContent=`第 ${t+1} / ${r.records.length} 步`;$('title').textContent=row.frame+' · 寻找 '+r.target_identity;$('reference').src=row.reference_asset;$('native').href=row.image_asset;
const elapsed=r.costs_s.slice(0,t).reduce((a,b)=>a+b,0);$('movement').textContent=(t?`从上一步执行 ${r.actions[t-1]}，标称耗时 ${r.costs_s[t-1]} 秒。`:'路线起点：记忆从空开始。')+`相机位置 ${r.poses[t].slice(0,3).map(v=>v.toFixed(2)).join(', ')}；朝向 ${r.poses[t][4]}°；累计标称动作时间 ${elapsed.toFixed(1)} 秒。`;
$('verdict').textContent=p?`本方法选择 ${label(p.choice_identity)}；已接收证据标签 ${label(p.accepted_truth)}；物理出现历史标签 ${label(p.physical_truth)}；${p.correct?'匹配正确':'匹配错误'}。当前有效观察 ${p.current_identities.join('、')||'无'}；本步写入 ${p.written_identities.join('、')||'无（缺席、过小或重复图）'}。`:'未运行模型。当前只检查路线、碎片合并、写入资格和跨帧对应。';
const image=new Image();image.onload=()=>{if(g!==generation)return;for(const id of ['image','heat']){const c=$(id);c.width=image.width;c.height=image.height;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);if(id==='heat'){const peak=Math.max(...row.spatial.flat(),1e-12);row.spatial.forEach((rr,y)=>rr.forEach((v,x)=>{ctx.fillStyle=`rgba(255,50,0,${.7*v/peak})`;ctx.fillRect(x*c.width/rr.length,y*c.height/row.spatial.length,c.width/rr.length+1,c.height/row.spatial.length+1)}));ctx.strokeStyle='#dedede';ctx.lineWidth=2;row.candidates.forEach(o=>{const[x,y,x1,y1]=o.box;ctx.strokeRect(x,y,x1-x,y1-y)})}else row.observations.forEach(o=>{const i=r.identities.indexOf(o.identity),[x,y,x1,y1]=o.box;ctx.strokeStyle=colors[i%colors.length];ctx.lineWidth=p?.choice_identity===o.identity?7:3;ctx.setLineDash(o.write_allowed?[]:[8,5]);ctx.strokeRect(x,y,x1-x,y1-y);ctx.setLineDash([]);ctx.font='bold 22px sans-serif';ctx.fillStyle=colors[i%colors.length];ctx.fillText(o.identity,x+4,Math.max(24,y-5))})}};image.src=row.image_asset;
const canvas=$('routeMap'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const poses=r.poses.slice(0,t+1),xs=r.poses.map(q=>q[0]),ys=r.poses.map(q=>q[1]),loX=Math.min(...xs),loY=Math.min(...ys),sx=Math.max(1,Math.max(...xs)-loX),sy=Math.max(1,Math.max(...ys)-loY);ctx.beginPath();poses.forEach((q,i)=>{const x=40+((q[0]-loX)/sx)*850,y=25+((q[1]-loY)/sy)*100;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle='#157dcb';ctx.lineWidth=3;ctx.stroke();const now=poses.at(-1),cx=40+(now[0]-loX)/sx*850,cy=25+(now[1]-loY)/sy*100;ctx.fillStyle='#172c43';ctx.beginPath();ctx.arc(cx,cy,7,0,Math.PI*2);ctx.fill();const angle=now[4]*Math.PI/180;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(angle)*24,cy+Math.sin(angle)*24);ctx.stroke();
$('cars').replaceChildren();for(const uid of r.identities){const i=r.identities.indexOf(uid),current=row.observations.find(o=>o.identity===uid),history=[],used=new Set();for(let s=0;s<=t;s++){const rr=DATA.records[r.records[s]],o=rr.observations.find(o=>o.identity===uid&&o.write_allowed);if(o&&(!DATA.config.exact_dedup||!used.has(o.key))){history.push({step:s,obs:o});used.add(o.key)}}const latest=history.at(-1),tr=document.createElement('tr');if(p?.choice_identity===uid)tr.className='selected';const info=document.createElement('td');info.style.borderLeft='4px solid '+colors[i%colors.length];info.textContent=uid+'；当前'+(row.physical_visible[uid]?'物理可见':'不在画面中')+'；'+(current?current.write_allowed?'区域允许写入':'区域太小，拒绝写入':'本步没有老师支持的区域');tr.append(info);const currentCell=document.createElement('td');imgCell(currentCell,current?.crop,current?'本步裁图'+(current.write_allowed?'':'（不写入）'):'本步无裁图');if(latest&&latest.obs.key!==current?.key)imgCell(currentCell,latest.obs.crop,`最近有效观察：第${latest.step+1}步`);tr.append(currentCell);const gallery=document.createElement('td');gallery.className='history';for(const h of history){const a=document.createElement('a');a.href=h.obs.crop;a.target='_blank';const im=document.createElement('img');im.src=h.obs.crop;a.append(im,document.createElement('br'),document.createTextNode('第'+(h.step+1)+'步'));gallery.append(a)}if(!history.length)gallery.textContent='尚无有效历史';tr.append(gallery);const score=document.createElement('td');if(p){const j=p.identities.indexOf(uid),m=p.memory[uid];score.textContent=m.seen?`概率 ${(100*p.probabilities[j]).toFixed(2)}%；分数 ${p.scores[j].toFixed(4)}；已写入 ${m.writes} 次；`:'模型尚未接收该车。';if(m.seen)score.textContent+=(m.change_l2===null?'首次状态':`相对上一步变化 ${m.change_l2.toFixed(4)}`)+'；状态摘要 '+m.slot_sha256.slice(0,12)}else score.textContent=`预期可用历史图片 ${history.length} 张；尚无网络状态`;tr.append(score);$('cars').append(tr)}
$('null').textContent=p?`Null 概率 ${(100*p.probabilities.at(-1)).toFixed(2)}%。Null 表示已有有效证据中尚未接收目标，不代表整片环境没有目标。`:'Null 的训练含义：截至此步，已有有效证据中尚未接收目标。';
$('compare').replaceChildren();for(const method of ['last','mean','memory']){const q=pred(method,t),tr=document.createElement('tr');[names[method],q?label(q.choice_identity):'未训练',q?(q.correct?'正确':'错误'):'—'].forEach(text=>{const td=document.createElement('td');td.textContent=text;tr.append(td)});$('compare').append(tr)}
const cc=DATA.controls[$('seed').value]?.[r.id]?.[String(t)]||[];$('control').textContent=cc.length?'学习融合的同权重干预：'+cc.map(c=>`${c.control} → ${label(c.choice_identity)}（${c.control_correct?'正确':'错误'}）`).join('；'):'本步没有满足条件的历史干预记录。';
$('raw').textContent=JSON.stringify({step:t,record:row,prediction:p,controls:cc},null,2)}
$('split').onchange=routes;$('route').onchange=reset;$('seed').onchange=()=>{stop();render()};$('method').onchange=render;$('step').oninput=()=>{stop();render()};
$('prev').onclick=()=>{stop();$('step').value=Math.max(0,Number($('step').value)-1);render()};$('next').onclick=()=>{stop();$('step').value=Math.min(Number($('step').max),Number($('step').value)+1);render()};$('play').onclick=()=>{if(timer){stop();return}timer=setInterval(()=>{if(Number($('step').value)>=Number($('step').max)){stop();return}$('step').value=Number($('step').value)+1;render()},1200);$('play').textContent='暂停'};routes();
</script></html>'''
