"""Static reports from saved predictions. No model or torch imports."""
from collections import defaultdict
import html
import json
from pathlib import Path

from .region_ab_data import read_json,write_json,digest_json,sha
from .evidence_query_data import render_sample,validate_manifest
from .evidence_query_metrics import summarize_updates

CSS='''body{font:15px system-ui,sans-serif;margin:24px;color:#17202a;background:#f6f7f9}main{max-width:1500px;margin:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:7px;border:1px solid #ccd1d8;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{display:inline-block;margin:8px;vertical-align:top}img{max-width:230px;max-height:170px;object-fit:contain}figcaption{max-width:230px;font-size:12px;overflow-wrap:anywhere}section{padding:16px;background:white;margin:16px 0}summary{cursor:pointer}a{color:#164ea1}.notice{background:#fff0d0;padding:12px}'''


def esc(value): return html.escape(str(value))


def page(title,body):
    return f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style><main><h1>{esc(title)}</h1>{body}</main></html>'


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def metric(x):
    return 'N/A (0)' if x['value'] is None else f"{x['value']:.4f} ({x['numerator']:.4g}/{x['denominator']})"


def report(args):
    output=Path(args.output).resolve(); m=read_json(output/'manifest.json'); validate_manifest(m)
    root=Path(args.data or m['data_root']); destination=output/'report'; assets=destination/'assets'
    assets.mkdir(parents=True,exist_ok=True)
    records=sorted((output/'evaluation').glob('*/*/records.json'))
    if not records: raise ValueError('no saved evaluation records; report never runs a model')
    overview=[]; bundles={}; md=['# 独立查询车辆记忆实验报告','',
        '身份由仿真辅助路由，定位参考固定。每车记忆 8×256。报告只读取已有预测，不执行神经网络。',
        f"外观模式：{m['config']['group_mode']}；身份划分：{m['config']['split_mode']}。",'',
        'query_all 下 C/D 不是有效历史消融。概率变化只是诊断，help/harm 以身份对错计算。',
        '不同预算、数据划分分别统计；窗口和查询之间相关，不能视为独立场景。','']
    source_records={}
    def picture(sid,perturb,caption,prefix):
        key=digest_json([sid,perturb]); path=assets/f'{key}.jpg'
        if not path.exists():
            crop=render_sample(root,m['samples'][sid],perturb)
            crop.save(path,quality=92)
        return f'<figure><a href="{prefix}assets/{key}.jpg"><img loading="lazy" src="{prefix}assets/{key}.jpg"></a><figcaption>{esc(caption)}</figcaption></figure>'
    for file in records:
        saved=read_json(file)
        if saved['status']!='complete': raise ValueError('evaluation incomplete: '+str(file))
        if not saved.get('synthetic') and saved.get('manifest_digest')!=digest_json(m): raise ValueError('evaluation manifest mismatch')
        scope='/'.join(file.relative_to(output/'evaluation').parts[:2]); rows=saved['rows']
        summary=summarize_updates(rows); bundles[scope]=summary; source_records[scope]=dict(path=file.relative_to(output).as_posix(),sha256=sha(file),synthetic=bool(saved.get('synthetic')))
        md.extend([f'## {scope}',f"人工测试记录：{bool(saved.get('synthetic'))}",'',
            '| 方法/种子 | Query | History | help | harm | net | repeat 1 | repeat 8 |',
            '|---|---|---|---|---|---|---|---|'])
        comparison=[]
        for v,seeds in summary['by_method_seed'].items():
            for seed,s in seeds.items():
                row=[v+'/'+seed]+[metric(s[k]) for k in ('query_accuracy','history_query_accuracy','help_rate','harm_rate','net_gain')]
                rep=summary['by_method_seed_strata'][v][seed]['repetitions']
                row += [metric(rep[n]['repeat_flip_rate']) if n in rep else 'N/A (0)' for n in ('1','8')]
                comparison.append(row); md.append('| '+' | '.join(row)+' |')
        body=f'<h2>{esc(scope)}</h2><p>人工测试记录：{bool(saved.get("synthetic"))}</p>'+table(['方法/种子','Query','History','help','harm','net','repeat 1','repeat 8'],comparison)
        # Explicit denominators and all strata are downloadable and visible without model access.
        body+='<details><summary>全部指标、分层和分母</summary><pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>'
        grouped=defaultdict(list)
        for row in rows:
            if row.get('episode_id'): grouped[row['episode_id']].append(row)
        links=[]; sub=destination/scope.replace('/','_'); sub.mkdir(exist_ok=True)
        for ep in m['episodes']:
            if ep['id'] not in grouped: continue
            ep_rows=grouped[ep['id']]; blocks=[]; history=[]
            last_recorded_step=max(r['step'] for r in ep_rows)
            for ti,st in enumerate(ep['steps']):
                if ti>last_recorded_step: break
                relevant=[r for r in ep_rows if r['step']==ti]
                current=''.join(picture(o['sample_id'],o['perturbation'],f"{o['vehicle_uid']} / {o['perturbation']['kind']} / write={o['write']}",'../') for o in st['observations'])
                queries=''.join(picture(q['sample_id'],{'kind':'clean'},f"query={m['samples'][q['sample_id']]['vehicle_uid']} / label={q['label']} / group={q['group']}",'../') for q in st['queries'])
                before=''.join(picture(o['sample_id'],o['perturbation'],f"历史 t={time_step} {o['vehicle_uid']}",'../') for time_step,o in history)
                diagnostic=[]
                for r in relevant:
                    diagnostic.append([r['variant']+'/'+str(r['seed']),r['kind'],r.get('vehicle_uid',''),r.get('query_identity',''),
                        r.get('label'),r.get('choice',r.get('full_choice')),r.get('skip_choice',r.get('repeat_choice','')),
                        r.get('repetitions',''),r.get('eligible',''),
                        json.dumps(r.get('scores',r.get('full_scores',[]))),json.dumps(r.get('skip_scores',r.get('repeat_scores',[])))])
                blocks.append(f'<section><h2>步骤 {ti}</h2><h3>当前观察</h3>{current}<h3>当前独立查询（不写入）</h3>{queries}<details><summary>此前已写入的观察：{len(history)} 张</summary>{before}</details>'+table(
                    ['方法','记录','干预车/查询车','查询身份','真值','Full','Skip / Repeat','重复次数','更新收益合格','Full分数','对照分数'],diagnostic)+'</section>')
                history.extend((ti,o) for o in st['observations'] if o['write'])
            name=digest_json(ep['id'])[:20]+'.html'
            (sub/name).write_text(page(ep['id'],f'<a href="../index.html">返回汇总</a><p>固定定位参考：{esc(ep["source_reference_id"])}；候选：{esc(ep["identities"])}</p>'+''.join(blocks)),encoding='utf-8')
            links.append(f'<li><a href="{sub.name}/{name}">{esc(ep["id"])}</a></li>')
        body+='<details><summary>逐步图片与 Full / Skip / Repeat 对照</summary><ul>'+''.join(links)+'</ul></details>'
        overview.append(body)
    metadata=dict(config=m['config'],coverage=m['coverage'],split_audit=m['split_audit'],manifest_digest=digest_json(m),
        source_hashes=m['source_hashes'],encoder_source=m['encoder_source'],evaluations=source_records,
        cuda_status=read_json(output/'checks.json') if (output/'checks.json').exists() else 'not recorded')
    write_json(destination/'summary.json',dict(scopes=bundles,metadata=metadata))
    md.extend(['','## 来源和复现信息','```json',json.dumps(metadata,ensure_ascii=False,indent=2),'```','',
               '全部分层及种子统计见 summary.json；逐步图像和原始分数对照见 index.html。'])
    (destination/'report.md').write_text('\n'.join(md),encoding='utf-8')
    header='<p class="notice">当前数据若只有两辆车、同一场景，只能检查融合机制。query_all 无历史外观监督。首次写入不进入后续更新 help/harm。Null 始终保留；N/A 表示没有有效分母。</p>'
    metadata_html='<details><summary>数据来源、配置、校验记录</summary><pre>'+esc(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></details>'
    (destination/'index.html').write_text(page('独立查询车辆记忆实验',header+''.join(overview)+metadata_html),encoding='utf-8')
    return dict(status='complete',report=str(destination/'index.html'),scopes=list(bundles),model_execution=False)
