"""Build a portable part-evidence ledger and conservatively retain every candidate."""

import argparse
import copy
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

from dinoexp.core import checked_path, cosine_scores
from dinoexp.evidence import ASSETS, apply_reviews, build_cards, fuse_cards, review_template


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def score_cards(root, cards, encoder):
    """Diagnostic bbox cosine only. Neither availability nor review status changes."""
    paths = []
    pairs = []
    for card in cards:
        if card['availability'] != 'needs_review':
            continue
        pair = []
        for sample in (card['reference'], card['observation']):
            path = sample['instances'][0]['crop']
            if path not in paths:
                paths.append(path)
            pair.append(paths.index(path))
        pairs.append((card, pair))
    if not paths:
        return None, []
    images = []
    for path in paths:
        with Image.open(checked_path(root, path)) as image:
            images.append(image.convert('RGB'))
    values = encoder(images)
    if len(values) != len(paths):
        raise ValueError('encoder output count mismatch')
    for card, (reference, query) in pairs:
        card['cosine'] = float(cosine_scores(values[reference], values[query:query + 1])[0])
    return values, paths


def compare(args, cards):
    import numpy as np
    import torch
    from match import encode

    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    if args.model_repo:
        model = torch.hub.load(str(args.model_repo), 'dinov2_vits14', source='local',
                               pretrained=not bool(args.checkpoint))
    else:
        model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True)
    checkpoint_hash = None
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True),
                              strict=True)
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = model.to(device).eval()
    values, paths = score_cards(args.parts_run, cards, lambda images: encode(
        images, model, torch, device, args.size, args.batch_size, 'cls'))
    if values is not None:
        np.savez_compressed(args.output / 'features.npz', features=values, paths=np.array(paths))
    return dict(model='dinov2_vits14', device=str(device), torch=torch.__version__,
                repo=str(args.model_repo or 'facebookresearch/dinov2:main'),
                checkpoint_sha256=checkpoint_hash, size=args.size, pooling='cls',
                input='native part bounding rectangle, not mask-pooled tokens',
                preprocess='aspect-preserving letterbox + ImageNet normalization',
                compared=sum(c['cosine'] is not None for c in cards))


def write_report(root, output, cards, run):
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(run, status='running', input=str(root.resolve()),
                    parts_sha256=hashlib.sha256((root / 'parts.json').read_bytes()).hexdigest(),
                    parts_run_sha256=hashlib.sha256((root / 'parts_run.json').read_bytes()).hexdigest(),
                    warning='No auto-exclusion. Cosine and detector confidence are not probabilities.')
    write_json(output / 'evidence_run.json', manifest)
    try:
        portable = copy.deepcopy(cards)
        assets = output / 'assets'
        assets.mkdir()
        copied = {}

        def copy_asset(path):
            if path not in copied:
                source = checked_path(root, path)
                name = hashlib.sha256(source.read_bytes()).hexdigest() + source.suffix.lower()
                destination = assets / name
                if not destination.exists():
                    shutil.copyfile(source, destination)
                copied[path] = 'assets/' + name
            return copied[path]

        for card in portable:
            for sample in (card['reference'], card['observation']):
                for key in ('source', 'overlay'):
                    sample[key] = copy_asset(sample[key])
                for instance in sample['instances']:
                    for key in ASSETS:
                        if instance.get(key):
                            instance[key] = copy_asset(instance[key])
        write_json(output / 'evidence.json', portable)
        write_json(output / 'candidates.json', fuse_cards(cards))
        template = review_template(cards)
        by_id = {c['card_id']: c for c in cards}
        for i, entry in enumerate(template['reviews']):
            if by_id[entry['card_id']]['review']:
                template['reviews'][i] = by_id[entry['card_id']]['review']
        write_json(output / 'reviews_template.json', template)
        fields = ['scene', 'candidate', 'sample_id', 'sequence', 'view', 'frame', 'part',
                  'availability', 'status', 'cosine', 'card_id', 'fingerprint']
        with (output / 'evidence.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(cards)
        write_page(output, portable, template, fuse_cards(cards))
        manifest.update(status='complete', cards=len(cards),
                        reviewed=sum(c['review'] is not None for c in cards),
                        candidates=len(fuse_cards(cards)))
        write_json(output / 'evidence_run.json', manifest)
    except BaseException as error:
        manifest.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        write_json(output / 'evidence_run.json', manifest)
        raise


def write_page(output, cards, template, candidates):
    def esc(value):
        return html.escape(str(value), quote=True)
    sections = []
    for card in cards:
        pictures = []
        for title, sample in [('参考', card['reference']), ('候选', card['observation'])]:
            images = f'<img src="{esc(sample["source"])}"><img src="{esc(sample["overlay"])}">'
            for instance in sample['instances']:
                images += (f'<img src="{esc(instance["crop"])}">'
                           f'<img src="{esc(instance["masked_crop"])}">'
                           f'<p>检测分数 {instance["confidence"]:.3f}；'
                           f'ROI {esc(instance["roi_size"])}；mask 像素 {instance["mask_pixels"]}</p>')
            pictures.append(f'<section><b>{title}：{esc(sample["sample_id"])}</b>{images}</section>')
        sections.append(
            f'<article data-id="{esc(card["card_id"])}" data-available="{str(card["availability"] == "needs_review").lower()}"><h2>{esc(card["scene"])} / '
            f'{esc(card["candidate"])} / {esc(card["view"])} / {esc(card["part"])}</h2>'
            f'<p>定位状态：{esc(card["availability"])}；审核状态：{esc(card["status"])}；'
            f'DINOv2 余弦：{esc(card["cosine"])}</p><div class="pair">'
            + ''.join(pictures) + '</div><div class="review"></div></article>')
    payload = json.dumps(template, ensure_ascii=True).replace('<', '\\u003c')
    summary = esc(json.dumps(candidates, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>车辆部位证据记录</title>
<style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:auto;padding:24px}article{border:1px solid #bbb;margin:20px 0;padding:16px}h2{font-size:19px}.pair{display:flex;gap:20px}.pair section{width:50%}img{max-width:100%;max-height:180px;object-fit:contain;display:block;margin:8px 0}aside{background:#fff2d8;padding:16px}label{display:inline-block;margin:5px 14px 5px 0}textarea{display:block;width:95%;height:65px}button,input,select{font:inherit}pre{white-space:pre-wrap;font-size:13px}@media(max-width:700px){.pair{display:block}.pair section{width:100%}}</style>
<h1>车辆部位证据记录</h1><aside>默认全部保留候选。部位类别相同只是待审核的对应关系。
未检出、重复检出、低相似度都不代表不是目标。颜色差异可能来自阴影；请核实可见细节及光照解释。
当前“相容/矛盾”由人工审核，不是自动识别结果。本页可填写审核，导出 JSON 后重新运行程序生成历史状态。
建议动作是固定规则提示，没有接入世界模型或 CARLA 移动控制。</aside>
<p><a href="evidence.csv">CSV</a> · <a href="evidence.json">证据记录</a> · <a href="candidates.json">候选历史</a> · <a href="reviews_template.json">审核模板</a></p>
<button id="export">导出审核 reviews.json</button><span id="message"></span>
<details><summary>当前融合结果（修改审核后需重新运行更新）</summary><pre>'''
    script = '''<script>
const doc=JSON.parse(document.getElementById('review-data').textContent);
const labels={reference_correct:'参考区域定位正确',query_correct:'候选区域定位正确',same_part_side:'同侧对应部位',identity_continuity:'候选身份连续',clear_enough:'足够清晰且无遮挡',appearance_stable:'任务期间该特征稳定'};
const states={pending:'待审核',uncertain:'仍不确定',compatible:'相容',contradictory:'矛盾'};
for(const article of document.querySelectorAll('article')){
 const row=doc.reviews.find(r=>r.card_id===article.dataset.id), box=article.querySelector('.review');
 const select=document.createElement('select');
 for(const [value,text] of Object.entries(states)){const opt=new Option(text,value);opt.disabled=article.dataset.available!=='true'&&['compatible','contradictory'].includes(value);select.add(opt);}
 select.value=row.status;select.onchange=()=>row.status=select.value;box.append(select);
 const reviewer=document.createElement('input');reviewer.placeholder='审核人';reviewer.value=row.reviewer;reviewer.oninput=()=>row.reviewer=reviewer.value;box.append(reviewer);
 const checks=document.createElement('div');
 for(const [key,text] of Object.entries(labels)){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=row.checks[key];input.onchange=()=>row.checks[key]=input.checked;label.append(input,document.createTextNode(text));checks.append(label);}box.append(checks);
 const basis=document.createElement('textarea');basis.placeholder='写明观察到的细节、疑点，以及是否能够排除阴影等解释。';basis.value=row.basis;basis.oninput=()=>row.basis=basis.value;box.append(basis);
}
document.getElementById('export').onclick=()=>{
 for(const r of doc.reviews){if(r.status==='pending')continue;if(!r.reviewer.trim()||!r.basis.trim()){alert('请为已审核记录填写审核人和依据。');return;}
 if(['compatible','contradictory'].includes(r.status)&&!Object.values(r.checks).every(v=>v===true)){alert('相容/矛盾记录必须完成全部对应与质量检查。');return;}}
 const url=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.getElementById('message').textContent=' 已导出；请用 --reviews reviews.json 重新运行。';
};
</script></html>'''
    (output / 'index.html').write_text(page + summary + '</pre></details>' + ''.join(sections)
        + '<script type="application/json" id="review-data">' + payload + '</script>' + script,
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-run', type=Path, required=True, help='complete segment_parts.py output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', default=['right_front-door', 'right_back-door'],
                        help='exact part labels; never substitute left/right')
    parser.add_argument('--reviews', type=Path, help='human reviews exported from report')
    parser.add_argument('--compare', action='store_true', help='also compute diagnostic DINOv2 cosines')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-repo', type=Path, help='local DINOv2 checkout for offline inference')
    parser.add_argument('--checkpoint', type=Path, help='official local weights; requires --model-repo')
    args = parser.parse_args()
    if args.size <= 0 or args.size % 14 or args.batch_size < 1:
        raise ValueError('size must be divisible by 14 and batch size positive')
    if args.checkpoint and not args.model_repo:
        raise ValueError('--checkpoint requires --model-repo')
    if args.output.resolve().is_relative_to(args.parts_run.resolve()):
        raise ValueError('output must be outside parts run')
    if args.output.exists():
        raise ValueError('output already exists; choose a new directory')
    cards = build_cards(args.parts_run, args.parts)
    if args.reviews:
        cards = apply_reviews(cards, json.loads(args.reviews.read_text(encoding='utf-8-sig')))
    # Report first so failed model downloads retain an inspectable ledger; manifest marks failure.
    run = dict(comparison=None, reviews_sha256=hashlib.sha256(args.reviews.read_bytes()).hexdigest()
               if args.reviews else None)
    write_report(args.parts_run, args.output, cards, run)
    if args.compare:
        manifest = json.loads((args.output / 'evidence_run.json').read_text(encoding='utf-8'))
        manifest['status'] = 'running'
        write_json(args.output / 'evidence_run.json', manifest)
        try:
            manifest['comparison'] = compare(args, cards)
            # Reuse the copied portable descriptors while attaching diagnostic scores.
            portable = json.loads((args.output / 'evidence.json').read_text(encoding='utf-8'))
            scores = {c['card_id']: c['cosine'] for c in cards}
            for card in portable:
                card['cosine'] = scores[card['card_id']]
            write_json(args.output / 'evidence.json', portable)
            template = json.loads((args.output / 'reviews_template.json').read_text(encoding='utf-8'))
            write_page(args.output, portable, template, fuse_cards(cards))
            fields = ['scene', 'candidate', 'sample_id', 'sequence', 'view', 'frame', 'part',
                      'availability', 'status', 'cosine', 'card_id', 'fingerprint']
            with (args.output / 'evidence.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(cards)
            manifest['status'] = 'complete'
        except BaseException as error:
            manifest.update(status='incomplete', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            write_json(args.output / 'evidence_run.json', manifest)
    print(f'Saved {len(cards)} evidence cards; all candidates retained: {args.output / "index.html"}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
