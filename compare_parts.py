"""Compare segmentation outputs on identical source crops; counts are not accuracy."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

from dinoexp.core import checked_path


def load_run(root):
    manifest = json.loads((root / 'parts_run.json').read_text(encoding='utf-8-sig'))
    if manifest.get('status') != 'complete':
        raise ValueError(f'parts run must be complete: {root}')
    rows = json.loads((root / 'parts.json').read_text(encoding='utf-8-sig'))
    samples = {}
    for row in rows:
        key = (row['scene'], row['sample_id'])
        if key in samples or not row['records']:
            raise ValueError('duplicate or empty parts sample')
        samples[key] = row
    if not samples:
        raise ValueError('empty parts run')
    return manifest, samples


def compare_runs(baseline, candidate, output):
    baseline, candidate, output = map(Path, [baseline, candidate, output])
    left_run, left = load_run(baseline)
    right_run, right = load_run(candidate)
    if left.keys() != right.keys():
        raise ValueError('comparison needs exactly the same scene/sample IDs')
    for key in left:
        a, b = left[key], right[key]
        if a['role'] != b['role'] or a['capture_metadata'] != b['capture_metadata']:
            raise ValueError(f'comparison source metadata differ: {key}')
        if a['records'][0]['source_sha256'] != b['records'][0]['source_sha256']:
            raise ValueError(f'comparison source crop hashes differ: {key}')
        # Detect corrupted downloads as well as mismatched manifests.
        with_image = []
        for root, sample in [(baseline, a), (candidate, b)]:
            from PIL import Image
            import numpy as np
            record = sample['records'][0]
            with Image.open(checked_path(root, record['source'])) as image:
                if image.size != (record['native_width'], record['native_height']):
                    raise ValueError('source image size disagrees with segmentation metadata')
                with_image.append(np.asarray(image.convert('RGB')))
        if with_image[0].shape != with_image[1].shape or not (with_image[0] == with_image[1]).all():
            raise ValueError(f'comparison source pixels differ: {key}')
    if (output.resolve().is_relative_to(baseline.resolve())
            or output.resolve().is_relative_to(candidate.resolve())):
        raise ValueError('comparison output must be outside model inputs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'assets').mkdir()

    def esc(value):
        return html.escape(str(value), quote=True)

    def image_asset(root, path):
        source = checked_path(root, path)
        name = hashlib.sha256(source.read_bytes()).hexdigest() + source.suffix
        dest = output / 'assets' / name
        if not dest.exists():
            shutil.copyfile(source, dest)
        return 'assets/' + name

    def summary(samples):
        query = [s for s in samples.values() if s['role'] == 'query']
        return dict(query_images=len(query), images_with_any_prediction=sum(
            any(r['status'] == 'detected' for r in s['records']) for s in query),
            predicted_labels=dict(Counter(r['class_name'] for s in query
                for r in s['records'] if r['status'] == 'detected')))

    result = dict(samples=len(left), baseline=summary(left), candidate=summary(right),
                  accuracy=None, warning='No ground-truth masks; counts are not accuracy. '
                  'Prompt labels and YOLO classes differ; evaluate semantic correspondence manually.',
                  baseline_checkpoint=left_run.get('checkpoint_sha256'),
                  candidate_checkpoint=right_run.get('checkpoint_sha256'))
    cards = []
    for key in sorted(left, key=lambda k: (k[0], int(left[k]['capture_metadata'].get('sequence') or 0), k[1])):
        panels = []
        for title, root, sample in [('Baseline', baseline, left[key]), ('Candidate', candidate, right[key])]:
            record = sample['records'][0]
            source = image_asset(root, record['source'])
            overlay = image_asset(root, record['overlay'])
            labels = ', '.join(f"{r['class_name']} {r['confidence']:.3f}" for r in sample['records']
                               if r['status'] == 'detected') or 'no prediction'
            panels.append(f'<section><h3>{title}</h3><img src="{esc(source)}">'
                          f'<img src="{esc(overlay)}"><p>{esc(labels)}</p></section>')
        cards.append(f'<article><h2>{esc(key[0])} / {esc(key[1])}</h2><div class="pair">'
                     + ''.join(panels) + '</div></article>')
    page = '''<!doctype html><meta charset="utf-8"><title>车辆部件模型对照</title>
<style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:24px auto;padding:0 16px}.pair{display:flex;gap:20px}section{width:50%;overflow-wrap:anywhere}img{display:block;max-width:100%;max-height:260px;object-fit:contain}article{border-top:1px solid #bbb;margin:24px 0}pre{white-space:pre-wrap}aside{background:#fff2d8;padding:16px}@media(max-width:700px){.pair{display:block}section{width:100%}}</style>
<h1>同一批 CARLA 图片：部件定位对照</h1><aside>检出数量不是准确率。两个模型的类别与提示词不同，不能直接比较模型分数。
逐张检查：可见车门是否漏检、前后门是否混淆、挡风玻璃是否误当侧窗、不可见部位是否乱分。
SAM 3 输出是文字提示下的区域建议，不证明部位命名正确。没有身份判断或自动排除。</aside>
<p><a href="comparison.json">运行汇总</a></p><pre>'''
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'index.html').write_text(page + esc(json.dumps(result, ensure_ascii=False, indent=2))
                                     + '</pre>' + ''.join(cards), encoding='utf-8')
    print(f'Saved model comparison: {output / "index.html"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare_runs(args.baseline, args.candidate, args.output)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
