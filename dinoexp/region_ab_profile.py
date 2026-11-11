"""Actual uncached online timings; not estimates derived from cached matcher FPS."""
import copy
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .core import checked_path
from .evidence_memory import EvidenceMatcher
from .region_ab_data import read_json, write_json, sha, verify_sources, collect_frames
from .region_ab_prepare import FrozenRuntime, measured, pool
from .region_ab_training import forward_current
from .region_geometry import propose


@torch.inference_mode()
def profile(args):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    path = output/'online_profile.json'
    if path.exists():
        raise ValueError('online timing report exists; do not overwrite')
    meta, run = read_json(output/'prepared.json'), read_json(output/'training/run.json')
    if run.get('status') != 'complete' or run['prepared_sha256'] != sha(output/'prepared.json'):
        raise ValueError('profile requires matching complete prepared cache and training')
    verify_sources(root, meta['sources'])
    if sha(args.teacher) != meta['teacher_sha256']:
        raise ValueError('profile teacher checkpoint changed')
    actual_args = copy.copy(args); actual_args.dense_cache = None
    runtime = FrozenRuntime(actual_args); device = runtime.device
    if runtime.cfg != meta['feature_config']:
        raise ValueError('profile backbone/config differs from prepared A/B experiment')
    verify_sources(runtime.repo, meta['backbone_source'])
    frames, _, _ = collect_frames(root, runtime.cp)
    frame_map = {(f['scene'], f['state']): f for f in frames}
    records = read_json(output/'proposals/records.json')['records']
    held = [r for r in records if r['split'] != 'train']
    selected = [held[i] for i in np.linspace(0, len(held)-1, min(args.profile_pairs, len(held)), dtype=int)]
    models, refs = {}, {}
    seed = meta['config']['training']['seeds'][0]
    for v in ('A', 'B'):
        result = next(r for r in run['results'] if r['seed'] == seed and r['variant'] == v)
        ckpt_path = checked_path(output, result['checkpoint'])
        if sha(ckpt_path) != result['checkpoint_sha256']:
            raise ValueError('profile matcher checkpoint changed')
        cp = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = EvidenceMatcher(**cp['config']).to(device).eval()
        model.load_state_dict(cp['model_state'], strict=True); models[v] = model
    reference_stats = {}
    for r in selected:
        if r['reference_sha256'] not in refs:
            with Image.open(checked_path(root, r['reference'])) as im, measured(reference_stats, 'seconds', device):
                refs[r['reference_sha256']] = runtime.reference(im.convert('RGB'), r['reference_sha256'])

    def one(r, variant):
        frame = frame_map[r['scene'], r['frame']]
        with Image.open(checked_path(root, r['image'])) as im:
            image = im.convert('RGB')  # Camera RGB already available; exclude file decoding explicitly.
        ref = refs[r['reference_sha256']]
        timings = {}
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device); start = time.perf_counter()
        with measured(timings, 'teacher_seconds', device):
            views = runtime.views(image, r['image_sha256'])
            heat, _ = runtime.localization(views, ref, frame)
            candidates = propose(heat, image.size, meta['config']['proposal'])['candidates']
            del views
        if [c['box'] for c in candidates] != [c['box'] for c in r['candidates']]:
            raise ValueError('online proposal replay differs from cached experiment')
        with measured(timings, 'independent_feature_seconds', device):
            k = len(candidates)
            x = np.zeros((1, max(1, k), 16, 256), np.float32)
            valid = np.zeros((1, max(1, k), 16), bool)
            full = runtime.independent(image) if variant == 'A' else None
            for i, c in enumerate(candidates):
                if variant == 'A':
                    x[0, i], valid[0, i] = pool(full, image.size, c['box'])
                else:
                    crop = image.crop(c['box'])
                    q = runtime.independent(crop)
                    x[0, i], valid[0, i] = pool(q, crop.size, [0, 0, crop.width, crop.height])
        with measured(timings, 'matcher_with_transfers_seconds', device):
            inputs = dict(observation=runtime.tensor(x), valid=runtime.tensor(valid),
                          reference=runtime.tensor(ref[2])[None], reference_valid=runtime.tensor(ref[3])[None])
            score = forward_current(models[variant], inputs)[0].cpu().numpy()
            actual = np.concatenate([score[:k], score[-1:]])
            choice = int(np.argmax(actual))
        torch.cuda.synchronize(device)
        timings.update(total_online_seconds=time.perf_counter()-start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return dict(id=r['id'], split=r['split'], variant=variant, candidates=k,
                    choice=None if choice == k else choice, **timings)

    # Warm up every path once; never include warmup samples in timing averages.
    warmup = max(selected, key=lambda r: len(r['candidates']))
    for v in ('A', 'B'):
        one(warmup, v)
    results = []
    for i, r in enumerate(selected):
        for v in (('A', 'B') if i % 2 == 0 else ('B', 'A')):
            results.append(one(r, v))
        print(f'Online timing {i+1}/{len(selected)}', flush=True)
    keys = ['total_online_seconds', 'teacher_seconds', 'independent_feature_seconds', 'matcher_with_transfers_seconds']
    summary = {v: {key: dict(mean=float(np.mean([r[key] for r in results if r['variant'] == v])),
                             median=float(np.median([r[key] for r in results if r['variant'] == v]))) for key in keys} for v in ('A', 'B')}
    result = dict(status='complete', kind='region_ab_uncached_online_timing_v1', seed=seed, rows=results,
                  summary=summary, reference_once=reference_stats,
                  scope='RGB already in memory and reference encoded once. Each A/B decision recomputes the full 16-view teacher and independent feature branch without disk feature cache. Includes transfers/pooling/matcher, excludes image decode/model loading. Warmup excluded; variant execution order alternated. All models remain resident: peak VRAM is this profiling process, not a minimal deployment.',
                  selection='uniform deterministic positions over held-out frame-reference list; no accuracy-based selection',
                  prepared_sha256=sha(output/'prepared.json'))
    write_json(path, result)
    return result
