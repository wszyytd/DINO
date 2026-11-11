"""Read-only source verification and physical-label metadata for memory diagnosis."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import checked_path
from .local_pairs import sha256


def load_source_metadata(root, scenes, cache_meta):
    root = Path(root).resolve()
    contract = cache_meta['contract']
    if sha256(root / 'dataset_manifest.json') != contract['manifest_sha256']:
        raise ValueError('diagnosis dataset manifest differs from feature source')
    for relative, digest in contract['sources'].items():
        if sha256(checked_path(root, relative.replace('\\', '/'))) != digest:
            raise ValueError('diagnosis source metadata changed: ' + relative)
    manifest = json.loads((root / 'dataset_manifest.json').read_text(encoding='utf-8'))
    cases = {case['scene_id']: case for case in manifest['cases']}
    contexts, quality_rows = [], []
    for cached in scenes:
        case = cases[cached['scene_id']]
        base = checked_path(root, case['path'])
        scene = json.loads((base / 'scene.json').read_text(encoding='utf-8'))
        frames = [json.loads(line) for line in (base / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        audits = {r['state_id']: r for r in map(json.loads,
                  (base / 'audit/observations.jsonl').read_text(encoding='utf-8').splitlines())}
        if ([f['state_id'] for f in frames] != cached['state_ids'] or
                sorted(scene['objects']) != cached['objects'] or case['group'] != cached['group']):
            raise ValueError('source/cached candidate or frame mapping differs')
        references = {}
        for uid in cached['objects']:
            ref = scene['references'][uid]
            if sha256(checked_path(root, ref['object'])) != ref['hashes'][ref['object']]:
                raise ValueError('reference image changed')
            references[uid] = ref['object']
        frame_meta = []
        for index, frame in enumerate(frames):
            path = checked_path(root, frame['image_path'])
            digest = sha256(path)
            if digest != cached['image_hashes'][index] or digest != audits[frame['state_id']]['rgb']['file_sha256']:
                raise ValueError('RGB differs from cache or capture audit')
            with Image.open(path) as image:
                width, height = image.size
            candidates = []
            for slot, uid in enumerate(cached['objects']):
                obj = frame['objects'][uid]; label = obj['mask']; q = obj['quality']
                bbox = center = centroid = None
                pixels = 0
                if label:
                    mask_path = checked_path(root, label['path'])
                    if sha256(mask_path) != label['sha256']:
                        raise ValueError('instance mask changed')
                    with Image.open(mask_path) as mask:
                        if mask.size != (width, height):
                            raise ValueError('mask/source image sizes differ')
                        yy, xx = np.nonzero(np.asarray(mask.convert('L')) > 0)
                    pixels = len(xx)
                    if pixels:
                        bbox = [float(xx.min()/width), float(yy.min()/height),
                                float((xx.max()+1)/width), float((yy.max()+1)/height)]
                        center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
                        centroid = [float((xx.mean()+.5)/width), float((yy.mean()+.5)/height)]
                row = dict(scene_id=cached['scene_id'], frame_id=frame['state_id'], candidate_id=uid,
                           candidate_index=slot, image_path=frame['image_path'],
                           mask_path=label['path'] if label else None, visible_gt=bool(pixels),
                           legacy_visible=bool(cached['valid'][index, slot].any()),
                           annotated_visible=bool(q.get('visibility_present', False)),
                           visible_pixels=pixels, effective_area_518=q.get('effective_area_518'),
                           quality=q.get('recognition_quality', 'unknown'),
                           quality_rule=q.get('quality_rule'), touches_boundary=q.get('touches_boundary'),
                           severely_truncated=q.get('severely_truncated'), too_small=q.get('too_small'),
                           bbox_normalized=bbox, bbox_center_normalized=center,
                           mask_centroid_normalized=centroid)
                candidates.append(row); quality_rows.append(row)
            frame_meta.append(dict(image_path=frame['image_path'], candidates=candidates))
        contexts.append(dict(frames=frame_meta, references=references))
    return contexts, quality_rows


def enrich(row, contexts):
    context = contexts[row['scene_index']]
    frame = context['frames'][row['frame_index']]
    row['image_path'] = frame['image_path']
    row['reference_path'] = context['references'][row['target_id']]
    row['candidate_metadata'] = frame['candidates']
    physical_seen = [any(context['frames'][i]['candidates'][k]['visible_gt'] for i in row['prefix_frames'])
                     for k in range(len(row['candidate_ids']))]
    physical_truth = row['target_index'] if physical_seen[row['target_index']] else len(physical_seen)
    row['physical_seen'] = physical_seen
    row['physical_truth_index'] = physical_truth
    row['truth_disagrees'] = physical_truth != (row['target_index'] if row['target_seen'] else len(physical_seen))
    active = [c for c in frame['candidates'] if c['legacy_visible']]
    row['quality_group'] = ('both_usable' if all(c['quality'] == 'usable' for c in active)
                            else 'both_with_weak') if row['both_visible'] else 'not_both'
    target = frame['candidates'][row['target_index']]['bbox_center_normalized']
    other = [c['bbox_center_normalized'] for c in active
             if c['candidate_index'] != row['target_index'] and c['bbox_center_normalized']]
    row['target_horizontal'] = 'unavailable'
    if target and other:
        dx = target[0] - sum(c[0] for c in other)/len(other)
        row['target_horizontal'] = 'left' if dx < -1e-6 else 'right' if dx > 1e-6 else 'aligned'
    return row
