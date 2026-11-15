"""Static, file-openable LiVOS vehicle viewer; preview needs no model or torch."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .core import checked_path
from .livos_vehicle_data import binary_mask, load_plan, plan_digest
from .livos_vehicle_metrics import build_summary, score_frame
from .region_ab_data import read_json, sha, write_json


def overlay(image, target, others, ignored, prediction=None):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    layers = [(ignored, (150, 150, 150)), (others, (245, 155, 30)), (target, (35, 215, 110))]
    if prediction is not None:
        layers = [(prediction, (60, 140, 255))]
    for mask, color in layers:
        rgb[mask] = .55*rgb[mask]+.45*np.asarray(color)
    return Image.fromarray(rgb.astype(np.uint8))


def verify_results(output, plan, data):
    meta = read_json(output/'inference.json')
    if (meta.get('status') != 'complete' or meta.get('kind') != 'livos_vehicle_inference_v1'
            or meta['plan_sha256'] != plan_digest(plan)):
        raise ValueError('inference incomplete or different vehicle plan')
    for name, expected in meta['artifacts'].items():
        if sha(checked_path(output, name)) != expected:
            raise ValueError('saved inference artifact changed: '+name)
    records = read_json(output/'records.json')
    expected = {(c['id'], i): (c, key) for c in plan['cases'] for i, key in enumerate(c['frames'])}
    lookup = {(r['case'], r['step']): r for r in records}
    if len(lookup) != len(records) or lookup.keys() != expected.keys():
        raise ValueError('missing or repeated prediction steps')
    for key, (case, frame_key) in expected.items():
        row, frame = lookup[key], plan['frames'][frame_key]
        if row['frame'] != frame_key or row['target'] != case['target'] or row['initial'] != (key[1] == 0):
            raise ValueError('prediction identity/frame/initial mismatch')
        path = checked_path(output, row['prediction'])
        if sha(path) != row['prediction_sha256']:
            raise ValueError('prediction mask changed')
        with Image.open(path) as im:
            pred = np.asarray(im.convert('L')) > 0
        masks = {uid: binary_mask(data, obj['mask'], frame['size'])
                 for uid, obj in frame['objects'].items()}
        ignored = binary_mask(data, frame['ignore'], frame['size'])
        metric = score_frame(pred, masks, case['target'], ignored,
                             **{k: plan['config'][k] for k in ('success_iou', 'wrong_iou',
                                                             'wrong_margin', 'absent_pixels')})
        if any(row[k] != v for k, v in metric.items()):
            raise ValueError('recomputed mask scores differ from saved records')
    saved = read_json(output/'summary.json')
    if saved != build_summary(records):
        raise ValueError('summary differs from saved frame scores')
    return records, saved


def report(args, preview=False):
    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    plan = load_plan(output, data)
    directory = output/('preview' if preview else 'report')
    if directory.exists():
        raise ValueError('viewer already exists; previous files are preserved')
    records, summary = ([], None) if preview else verify_results(output, plan, data)
    lookup = {(r['case'], r['step']): r for r in records}
    assets = directory/'assets'
    assets.mkdir(parents=True)
    frames = {}
    for key in sorted({key for c in plan['cases'] for key in c['frames']}):
        f = plan['frames'][key]
        name = key.replace('/', '_')
        src = checked_path(data, f['image'])
        dest = assets/(name+src.suffix)
        shutil.copyfile(src, dest)
        frames[key] = dict(image=dest.relative_to(directory).as_posix(), state=f['state'])
    cases = []
    for case in plan['cases']:
        steps = []
        for i, key in enumerate(case['frames']):
            f = plan['frames'][key]
            masks = {uid: binary_mask(data, obj['mask'], f['size'])
                     for uid, obj in f['objects'].items()}
            gt = masks[case['target']]
            others = np.zeros_like(gt)
            for uid, mask in masks.items():
                if uid != case['target']:
                    others |= mask
            ignored = binary_mask(data, f['ignore'], f['size'])
            gt_name = f"{key.replace('/', '_')}_{case['target'].rsplit(':', 1)[-1]}_gt.png"
            with Image.open(checked_path(data, f['image'])) as im:
                gt_path = assets/gt_name
                if not gt_path.exists():
                    overlay(im, gt, others, ignored).save(gt_path)
                pred_image = None
                row = lookup.get((case['id'], i))
                if row is not None:
                    with Image.open(checked_path(output, row['prediction'])) as pred_im:
                        pred = np.asarray(pred_im.convert('L')) > 0
                    pred_path = assets/f"{case['id']}_{i:03d}_pred.png"
                    overlay(im, gt, others, ignored, pred).save(pred_path)
                    pred_image = pred_path.relative_to(directory).as_posix()
            steps.append(dict(**frames[key], gt=gt_path.relative_to(directory).as_posix(),
                              prediction=pred_image, metrics=row, target_pixels=int(gt.sum()),
                              other_pixels=int(others.sum()),
                              boundary=f['objects'][case['target']]['boundary'],
                              action=None if i == 0 else case['actions'][i-1]))
        cases.append(dict(id=case['id'], target=case['target'], tags=case['tags'], steps=steps))
    payload = dict(preview=preview, cases=cases, summary=summary, counts=plan['counts'])
    write_json(directory/'data.json', payload)
    if records:
        fields = ['case', 'step', 'frame', 'target', 'initial', 'visible', 'boundary', 'tiny',
                  'target_pixels', 'prediction_pixels', 'iou', 'success', 'wrong_vehicle_overlap',
                  'other_iou', 'absent_false_positive', 'inference_seconds']
        with (directory/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    page = HTML.replace('__DATA__', encoded)
    (directory/'index.html').write_text(page, encoding='utf-8')
    title = '车辆测试数据预览' if preview else 'LiVOS 车辆测试结果'
    note = title+'：'+str(plan['counts'])
    if summary:
        note += '\n'+json.dumps(summary['overall'], ensure_ascii=False, indent=2)
    (directory/'说明.txt').write_text(note, encoding='utf-8')
    print(title+':', directory/'index.html', flush=True)


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>LiVOS 车辆逐帧测试</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f9;color:#17283a;margin:24px}
main{max-width:1600px;margin:auto}h1{font-size:25px}section{background:white;padding:18px;margin:16px 0;border-radius:9px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.grid img{width:100%}
button,select,input{font:inherit;padding:5px;margin:4px}input[type=range]{width:55%}.history{display:flex;gap:8px;overflow:auto}.history img{width:130px;cursor:pointer}pre{white-space:pre-wrap;font-size:14px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><main><h1 id="title"></h1>
<p>每段路线只在第 1 张提供目标掩膜。绿色为目标真值，橙色为另一辆车，灰色为不评分区域，蓝色为模型预测。真值图用于核对；后续真值不输入模型。</p>
<p>这是离散相机路线：相邻拍摄位置相差 4 米或方向相差 45 度。路线有重复与交叠，统计不代表独立车辆泛化；目标消失可能是离开画面，不能直接称为遮挡。</p>
<section><select id="cases"></select><button id="prev">上一张</button><button id="next">下一张</button><button id="play">播放</button><br><input id="slider" type="range" min="0" value="0"><span id="position"></span><div id="info"></div></section>
<div class="grid"><section>原始画面<img id="rgb"></section><section>评分真值<img id="gt"></section><section>预测区域<img id="pred"><p id="noPrediction"></p></section></div>
<section><div id="metric"></div><p>已观察历史（只显示当前及以前的画面）</p><div class="history" id="history"></div></section>
<section><h2>汇总</h2><p>首帧由标注指定，已从效果统计排除。IoU 衡量预测区域与真值重合程度；成功定义为 IoU ≥ 0.5。跟错车按与另一辆车的区域重合诊断，不等于参考图识别准确率。</p><pre id="summary"></pre></section></main>
<script>const DATA=__DATA__;
const el=id=>document.getElementById(id);let ci=0,si=0,timer=null;
const pct=x=>x===null?'无样本':(100*x).toFixed(2)+'%';
el('title').textContent=DATA.preview?'车辆测试数据预览（尚无模型结果）':'LiVOS 车辆测试结果';
DATA.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.id+' / '+c.target+' / '+c.tags.join(', ');el('cases').appendChild(o)});
function show(){const c=DATA.cases[ci];si=Math.max(0,Math.min(si,c.steps.length-1));const f=c.steps[si];
el('slider').max=c.steps.length-1;el('slider').value=si;el('position').textContent=(si+1)+' / '+c.steps.length;
el('rgb').src=f.image;el('gt').src=f.gt;el('pred').hidden=!f.prediction;if(f.prediction)el('pred').src=f.prediction;
el('noPrediction').textContent=DATA.preview?'数据预览；未运行模型，没有预测或准确率。':'';
el('info').textContent='状态 '+f.state+'；目标像素 '+f.target_pixels+'；另一辆车像素 '+f.other_pixels+'；'+(f.boundary?'目标接触图边；':'')+(si===0?'首帧指定目标，不计效果':'上一动作 '+f.action);
const m=f.metrics;el('metric').textContent=m?'目标 IoU：'+(m.iou===null?'目标不在画面':m.iou.toFixed(4))+'；成功：'+m.success+'；跟错另一辆车：'+m.wrong_vehicle_overlap+'；消失时误报：'+m.absent_false_positive+'；状态张量：'+(m.state.persistent_tensor_bytes/1024).toFixed(1)+' KiB':'';
el('history').replaceChildren();c.steps.slice(0,si+1).forEach((h,j)=>{const im=document.createElement('img');im.src=h.image;im.title='第 '+(j+1)+' 张';im.onclick=()=>{si=j;show()};el('history').appendChild(im)});
}
el('cases').onchange=()=>{ci=Number(el('cases').value);si=0;show()};el('slider').oninput=()=>{si=Number(el('slider').value);show()};
el('prev').onclick=()=>{si--;show()};el('next').onclick=()=>{si++;show()};el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='播放'}else{el('play').textContent='暂停';timer=setInterval(()=>{if(si>=DATA.cases[ci].steps.length-1){clearInterval(timer);timer=null;el('play').textContent='播放';return}si++;show()},900)}};
if(DATA.summary){const s=DATA.summary.overall;el('summary').textContent='评分步数：'+s.scored_steps+'\n可见目标平均 IoU：'+pct(s.mean_visible_iou)+'\n可见目标成功率：'+pct(s.visible_success_rate)+'\n消失时误报率：'+pct(s.absent_false_positive_rate)+'\n两车同屏跟错率：'+pct(s.two_visible_wrong_rate)+'\n重现后恢复率：'+pct(s.recovery_rate)+'\n\n'+JSON.stringify(DATA.summary,null,2)}else el('summary').textContent='尚无模型结果。选样覆盖：'+JSON.stringify(DATA.counts,null,2);
show();</script></html>'''
