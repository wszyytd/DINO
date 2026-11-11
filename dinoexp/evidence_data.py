"""Frozen full-view teacher features and GT-mask pooling for stage-one diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .core import checked_path
from .dense_cache import DenseFeatureCache, save_json
from .dense_model import DenseSearchHead
from .dense_views import prepare_reference, make_views
from .local_pairs import sha256


def mask_patch_weights(mask, image_size=518):
    """Exact letterbox geometry used by dense_views._prepare, area per DINO patch."""
    w, h = mask.size
    scale = min(image_size/w, image_size/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    canvas = Image.new('L', (image_size, image_size))
    canvas.paste(mask.resize((rw, rh), Image.Resampling.NEAREST),
                 ((image_size-rw)//2, (image_size-rh)//2))
    a = np.asarray(canvas, dtype=np.float32)/255
    side = image_size//14
    return torch.from_numpy(a.reshape(side, 14, side, 14).mean((1, 3)).copy())


def pool_masked_regions(features, mask, output_side=4):
    """Pool FEATURE regions, never crop/re-encode source RGB. Zero means absent."""
    side = mask.shape[0]
    if mask.shape != (side, side) or features.shape[0] != side*side:
        raise ValueError('mask and feature grids differ')
    d = features.shape[-1]
    keep = mask > 0
    if not keep.any():
        return features.new_zeros(output_side**2, d), torch.zeros(output_side**2, dtype=torch.bool)
    yy, xx = torch.where(keep)
    y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
    weights = mask[y0:y1, x0:x1].to(features)[None, None]
    content = features.reshape(side, side, d)[y0:y1, x0:x1].permute(2, 0, 1)[None]
    denom = F.adaptive_avg_pool2d(weights, output_side)
    tokens = F.adaptive_avg_pool2d(content*weights, output_side)/denom.clamp_min(1e-8)
    return tokens[0].flatten(1).T, denom.flatten() > 0


def build_evidence_cache(root, teacher_path, backbone_path, repo, output, device,
                         dense_cache=None, threads=4):
    torch.set_num_threads(threads)
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('mode') != 'discrete_view_graph':
        raise ValueError('need complete grid capture')
    teacher_cp = torch.load(teacher_path, map_location='cpu', weights_only=True)
    cfg = teacher_cp['feature_config']
    if teacher_cp.get('kind') != 'dense_reference_search_v1' or cfg['backbone_sha256'] != sha256(backbone_path):
        raise ValueError('teacher/backbone provenance mismatch')
    teacher = DenseSearchHead(**teacher_cp['model_config']).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(teacher_cp['model_state'], strict=True)
    previous = DenseFeatureCache(dense_cache) if dense_cache else None
    if previous and previous.config != cfg:
        raise ValueError('old dense cache does not match teacher')
    sources = {}
    for case in manifest['cases']:
        base = checked_path(root, case['path'])
        for name in ('frames.jsonl', 'scene.json', 'graph.json'):
            sources[str((base/name).relative_to(root))] = sha256(base/name)
    contract = dict(kind='oracle_vehicle_evidence_full_view_v1', teacher_sha256=sha256(teacher_path),
                    backbone_sha256=cfg['backbone_sha256'], manifest_sha256=sha256(root/'dataset_manifest.json'),
                    sources=sources, image_size=cfg['image_size'], roi_side=4,
                    camera_state_input=False, reference_conditioned=False, dtype='float32')
    marker = output/'cache.json'
    if output.exists():
        if not marker.is_file() or json.loads(marker.read_text())['contract'] != contract:
            raise ValueError('cache contract changed; use a new directory')
        meta = json.loads(marker.read_text())
    else:
        output.mkdir(parents=True)
        meta = dict(status='running', contract=contract, parts={}, scenes=[])
        save_json(marker, meta)
    backbone = None

    def raw_features(path, kind, expected_hash):
        nonlocal backbone
        if sha256(path) != expected_hash:
            raise ValueError('source image hash mismatch: '+str(path))
        key = kind+':'+expected_hash
        if previous and key in previous.images:
            row = previous.get(key)
            return (torch.from_numpy(row['tokens'][0] if kind == 'image' else row['tokens']).float().to(device),
                    torch.from_numpy(row['valid'][0] if kind == 'image' else row['valid']).bool().to(device))
        if backbone is None:
            backbone = torch.hub.load(str(repo), cfg['model'], source='local', pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location='cpu', weights_only=True), strict=True)
            backbone.to(device).eval().requires_grad_(False)
        with Image.open(path) as im:
            prepared, valid = prepare_reference(im.convert('RGB'), cfg['image_size'])
        a = np.asarray(prepared, np.float32)/255
        a = (a-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(a.transpose(2, 0, 1).copy())[None].to(device)
        with torch.inference_mode():
            layers = backbone.get_intermediate_layers(x, n=[i-1 for i in cfg['layers']], norm=True)
        return torch.stack(layers, 1)[0], torch.from_numpy(valid).bool().to(device)

    def part(key, maker):
        if key in meta['parts']:
            entry = meta['parts'][key]
            path = checked_path(output, entry['file'])
            if sha256(path) != entry['sha256']:
                raise ValueError('corrupt evidence cache part')
            return torch.load(path, map_location='cpu', weights_only=True)
        value = maker()
        name = hashlib.sha256(key.encode()).hexdigest()+'.pt'
        torch.save(value, output/name)
        meta['parts'][key] = dict(file=name, sha256=sha256(output/name))
        save_json(marker, meta)
        return value

    scene_records = []
    for case in manifest['cases']:
        base = checked_path(root, case['path'])
        scene = json.loads((base/'scene.json').read_text())
        frames = [json.loads(line) for line in (base/'frames.jsonl').read_text().splitlines()]
        audits = {row['state_id']: row for row in map(json.loads, (base/'audit/observations.jsonl').read_text().splitlines())}
        objects = sorted(scene['objects'])
        references, ref_valid, observations, masks = [], [], [], []
        hashes = []
        for uid in objects:
            ref = scene['references'][uid]
            path = checked_path(root, ref['object'])
            if sha256(path) != ref['hashes'][ref['object']]:
                raise ValueError('reference changed')
            def make_ref(path=path, ref=ref):
                tokens, valid = raw_features(path, 'reference', ref['hashes'][ref['object']])
                with torch.inference_mode():
                    r, rv, _ = teacher._reference(tokens[None], valid[None])
                return dict(tokens=r[0].cpu(), valid=rv[0].cpu())
            value = part(case['scene_id']+':ref:'+uid, make_ref)
            references.append(value['tokens']); ref_valid.append(value['valid'])
        for i, frame in enumerate(frames):
            path = checked_path(root, frame['image_path'])
            digest = sha256(path); hashes.append(digest)
            if digest != audits[frame['state_id']]['rgb']['file_sha256']:
                raise ValueError('RGB hash differs from capture audit')
            # Check every label even when resuming an existing cache.
            for uid in objects:
                label = frame['objects'][uid]['mask']
                if label and sha256(checked_path(root, label['path'])) != label['sha256']:
                    raise ValueError('mask hash mismatch')
            def make_frame(frame=frame, path=path, digest=digest, objects=objects):
                tokens, valid = raw_features(path, 'image', digest)
                with Image.open(path) as im:
                    geometry = make_views(im, cfg['image_size'], cfg['tile_size'],
                                          grid_size=cfg['grid_size'])['geometry'][:1]
                    size = im.size
                with torch.inference_mode():
                    q, _ = teacher.encode_views(tokens[None, None], valid[None, None],
                                                 torch.from_numpy(geometry)[None].to(device))
                q = q[0, 0].cpu()
                outputs, keep = [], []
                for uid in objects:
                    label = frame['objects'][uid]['mask']
                    if label:
                        with Image.open(checked_path(root, label['path'])) as im:
                            if im.size != size:
                                raise ValueError('mask/image size mismatch')
                            weight = mask_patch_weights(im.convert('L'), cfg['image_size'])
                    else:
                        weight = torch.zeros(cfg['image_size']//14, cfg['image_size']//14)
                    out, ok = pool_masked_regions(q, weight)
                    outputs.append(out); keep.append(ok)
                return dict(tokens=torch.stack(outputs), valid=torch.stack(keep), image_sha256=digest)
            value = part(case['scene_id']+':'+frame['state_id'], make_frame)
            if value['image_sha256'] != digest:
                raise ValueError('RGB changed since extraction')
            observations.append(value['tokens']); masks.append(value['valid'])
            if i % 10 == 0 or i+1 == len(frames):
                print(f"Evidence features {case['scene_id']}: {i+1}/{len(frames)}", flush=True)
        result = dict(observations=torch.stack(observations), valid=torch.stack(masks),
                      references=torch.stack(references), reference_valid=torch.stack(ref_valid),
                      state_ids=[f['state_id'] for f in frames], image_hashes=hashes, objects=objects,
                      graph=json.loads((base/'graph.json').read_text()), group=case['group'],
                      split=case['split'], scene_id=case['scene_id'])
        file = case['scene_id']+'.pt'; torch.save(result, output/file)
        scene_records.append(dict(file=file, sha256=sha256(output/file)))
    meta.update(status='complete', scenes=scene_records,
                teacher_training_images=teacher_cp['training_images'],
                teacher_training_groups=teacher_cp['training_groups'])
    save_json(marker, meta)
    return meta


def load_evidence_cache(path):
    path = Path(path)
    meta = json.loads((path/'cache.json').read_text())
    if meta.get('status') != 'complete' or meta['contract']['kind'] != 'oracle_vehicle_evidence_full_view_v1':
        raise ValueError('need complete stage-one evidence cache')
    scenes = []
    for row in meta['scenes']:
        file = checked_path(path, row['file'])
        if sha256(file) != row['sha256']:
            raise ValueError('scene evidence hash mismatch')
        scenes.append(torch.load(file, map_location='cpu', weights_only=True))
    return meta, scenes
