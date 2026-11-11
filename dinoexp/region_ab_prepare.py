"""CUDA-only frozen full teacher localization and reference-independent A/B encoders."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from .core import checked_path
from .dense_cache import DenseFeatureCache, feature_config
from .dense_model import DenseSearchHead
from .dense_views import make_views, prepare_reference, camera_embedding
from .local_paths import resolve_backbone_paths
from .region_geometry import propose, pool_weights, clip_box
from .region_ab_data import (sha, write_json, code_record, collect_frames, labels_for_frame,
                             save_features, digest_json, read_json, verify_sources)


def cuda_device(name):
    device = torch.device(name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires an authorized CUDA server; no CPU fallback')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


@contextmanager
def measured(stats, key, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        stats[key] = stats.get(key, 0.) + time.perf_counter()-start


class FrozenRuntime:
    def __init__(self, args):
        self.device = cuda_device(args.device)
        self.teacher_sha256 = sha(args.teacher)
        checkpoint = torch.load(args.teacher, map_location='cpu', weights_only=True)
        if sha(args.teacher) != self.teacher_sha256:
            raise ValueError('teacher checkpoint changed while reading')
        if checkpoint.get('kind') != 'dense_reference_search_v1':
            raise ValueError('need actual dense teacher checkpoint')
        self.cp, self.cfg = checkpoint, checkpoint['feature_config']
        cfg = self.cfg
        repo, backbone_path = resolve_backbone_paths(args.model_repo, args.backbone, cfg['model'], Path(__file__).resolve().parents[1])
        expected = feature_config(cfg['model'], cfg['image_size'], cfg['grid_size'],
                                  cfg['tile_size'], cfg['dtype'], sha(backbone_path))
        if cfg != expected or cfg['model'] != 'dinov2_vitb14' or cfg['image_size'] != 518:
            raise ValueError('teacher/backbone/preprocessing contract differs from specified ViT-B/14 518 experiment')
        self.backbone_path, self.repo = backbone_path, repo
        self.teacher = DenseSearchHead(**checkpoint['model_config']).to(self.device).eval().requires_grad_(False)
        self.teacher.load_state_dict(checkpoint['model_state'], strict=True)
        if self.teacher.config['dim'] != 256 or self.teacher.config['reference_grid'] != 8:
            raise ValueError('this region cache requires dim256 and 8x8 reference tokens')
        self.previous = DenseFeatureCache(args.dense_cache) if args.dense_cache else None
        if self.previous and self.previous.config != cfg:
            raise ValueError('dense cache must exactly match checkpoint; GT pooled caches are not supported')
        self.backbone = torch.hub.load(str(repo), cfg['model'], source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(backbone_path, map_location='cpu', weights_only=True), strict=True)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.view_batch = args.view_batch
        self.cache_hits = dict(image=0, reference=0)

    def tensor(self, value):
        return torch.from_numpy(np.asarray(value)).to(self.device)

    def raw(self, pictures):
        features = []
        for start in range(0, len(pictures), self.view_batch):
            array = np.stack([np.asarray(p, np.float32)/255 for p in pictures[start:start+self.view_batch]])
            array = (array-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
            x = self.tensor(array.transpose(0, 3, 1, 2).copy())
            layers = self.backbone.get_intermediate_layers(x, n=[n-1 for n in self.cfg['layers']], norm=True)
            features.append(torch.stack(layers, 1).cpu().numpy().astype(self.cfg['dtype']))
        return np.concatenate(features)

    def views(self, image, digest):
        v = make_views(image, self.cfg['image_size'], self.cfg['tile_size'],
                       self.cfg['rows'], self.cfg['cols'], self.cfg['grid_size'])
        pictures = v.pop('images'); v.pop('grid_size')
        key = 'image:'+digest
        if self.previous and key in self.previous.images:
            old = self.previous.get(key)
            for name in v:
                if not np.allclose(v[name], old[name], rtol=0, atol=1e-6):
                    raise ValueError('dense cache native geometry mismatch')
            self.cache_hits['image'] += 1
            v['tokens'] = old['tokens']
        else:
            v['tokens'] = self.raw(pictures)
        return v

    def reference(self, image, digest):
        p, valid = prepare_reference(image, self.cfg['image_size'])
        key = 'reference:'+digest
        if self.previous and key in self.previous.images:
            old = self.previous.get(key)
            if not np.array_equal(valid, old['valid']):
                raise ValueError('dense reference validity mismatch')
            tokens = old['tokens']; self.cache_hits['reference'] += 1
        else:
            tokens = self.raw([p])[0]
        raw, keep = self.tensor(tokens).float()[None], self.tensor(valid).bool()[None]
        ref, rv, _ = self.teacher._reference(raw, keep)
        return raw, keep, ref[0].cpu().numpy(), rv[0].cpu().numpy()

    def localization(self, views, reference, frame):
        state, valid = camera_embedding(frame['camera_pose'], frame['fov'])
        if not valid:
            raise ValueError('actual camera metadata required for this teacher')
        inputs = {key: self.tensor(views[key])[None] for key in ('geometry', 'indices', 'weights', 'priors', 'view_weights')}
        inputs.update(reference=reference[0], reference_valid=reference[1],
                      views=self.tensor(views['tokens']).float()[None],
                      view_valid=self.tensor(views['valid']).bool()[None],
                      camera_state=self.tensor(state)[None], camera_valid=self.tensor(np.array([valid])))
        out = self.teacher(**inputs)
        return out['spatial'][0].cpu().numpy(), float(out['null'][0])

    def independent(self, image, digest=None):
        prepared, valid = prepare_reference(image, self.cfg['image_size'])
        key = 'image:'+digest if digest else None
        if self.previous and key in self.previous.images:
            tokens = self.previous.get(key)['tokens'][0]
            self.cache_hits['image'] += 1
        else:
            tokens = self.raw([prepared])[0]
        w, h = image.size
        scale = min(518/w, 518/h)
        geometry = np.array([[[0, 0, 1, 1, max(1, round(w*scale))/518, max(1, round(h*scale))/518]]], np.float32)
        q, _ = self.teacher.encode_views(self.tensor(tokens).float()[None, None],
                                         self.tensor(valid).bool()[None, None], self.tensor(geometry))
        return q[0, 0].cpu().numpy()


def pool(q, size, box):
    weight = pool_weights(size, box)
    denom = weight.sum(1)
    tokens = weight @ q / np.maximum(denom[:, None], 1e-8)
    if not np.isfinite(tokens).all():
        raise ValueError('nonfinite pooled region')
    return tokens.astype(np.float32), denom > 0


def _fatal(error):
    # Resource errors must never silently become a different experiment.
    if isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower():
        raise error


@torch.inference_mode()
def prepare(args, config):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    if output.exists():
        raise ValueError('output exists; preserve it and choose a fresh --output')
    device = cuda_device(args.device)
    torch.set_num_threads(args.threads)
    free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(output.parent).free
    if free_disk < 2*1024**3:
        raise ValueError('less than 2 GiB free disk: free space or choose another new output')
    print(f'Preflight: GPU free {free_gpu/1024**3:.2f}/{total_gpu/1024**3:.2f} GiB, disk free {free_disk/1024**3:.2f} GiB', flush=True)
    runtime = FrozenRuntime(args)
    frames, sources, split = collect_frames(root, runtime.cp)
    output.mkdir()
    for name in ('proposals', 'A', 'B', 'references', 'preview'):
        (output/name).mkdir()
    provenance = dict(status='running', kind='region_encoding_ab_v1', config=config,
                      code=code_record(), sources=sources, split=split,
                      teacher_sha256=runtime.teacher_sha256, teacher_path=str(Path(args.teacher).resolve()),
                      teacher_epoch=runtime.cp.get('epoch'), teacher_model=runtime.cp['model_config'],
                      teacher_training_images=runtime.cp['training_images'],
                      teacher_training_groups=runtime.cp['training_groups'], feature_config=runtime.cfg,
                      backbone_sha256=sha(runtime.backbone_path), backbone_path=str(runtime.backbone_path),
                      backbone_repo=str(runtime.repo), references={}, artifacts={},
                      backbone_source={name: sha(runtime.repo/name) for name in
                                       ('hubconf.py', 'dinov2/models/vision_transformer.py', 'dinov2/layers/attention.py')
                                       if (runtime.repo/name).is_file()},
                      environment=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                                       numpy=str(np.__version__), gpu=torch.cuda.get_device_name(device),
                                       view_batch=args.view_batch, threads=args.threads,
                                       deterministic=True, tf32=False),
                      resources=dict(free_gpu_bytes=free_gpu, total_gpu_bytes=total_gpu, free_disk_bytes=free_disk))
    write_json(output/'prepared.json', provenance)
    stats, ref_cache, records, labels = {}, {}, [], {}
    indexes = {v: dict(kind='region_encoding_'+v+'_v1', entries={}) for v in ('A', 'B')}
    seen_pairs = {}
    torch.cuda.reset_peak_memory_stats(device)
    for frame_index, frame in enumerate(frames):
        with Image.open(checked_path(root, frame['image'])) as im:
            image = im.convert('RGB')
        frame_records = []
        with measured(stats, 'teacher_backbone_and_views_seconds', device):
            views = runtime.views(image, frame['image_sha256'])
        for ref in frame['references']:
            if ref['sha256'] not in ref_cache:
                with Image.open(checked_path(root, ref['path'])) as im, measured(stats, 'reference_once_seconds', device):
                    ref_cache[ref['sha256']] = runtime.reference(im.convert('RGB'), ref['sha256'])
                file = 'references/'+ref['sha256']+'.npz'
                np.savez_compressed(output/file, tokens=ref_cache[ref['sha256']][2], valid=ref_cache[ref['sha256']][3])
                provenance['references'][ref['sha256']] = dict(file=file, sha256=sha(output/file))
            pair = frame['image_sha256']+':'+ref['sha256']
            identity_signature = {uid: obj['mask']['sha256'] if obj['mask'] else None for uid, obj in frame['objects'].items()}
            signature = digest_json([identity_signature, ref['identity'], frame['split'], frame['camera_pose'], frame['fov'], frame['ignore_mask']])
            if pair in seen_pairs:
                if seen_pairs[pair]['signature'] != signature:
                    raise ValueError('duplicate RGB/reference has inconsistent labels/split/pose; resolve capture records')
                seen_pairs[pair]['record']['aliases'].append([frame['scene'], frame['state']])
                continue
            sample_id = frame['scene']+'/'+frame['state']+'/'+ref['identity']
            r = dict(id=sample_id, scene=frame['scene'], frame=frame['state'], image=frame['image'],
                     image_sha256=frame['image_sha256'], reference=ref['path'], reference_sha256=ref['sha256'],
                     target_identity=ref['identity'], size=frame['size'], split=frame['split'], aliases=[], teacher_error=None)
            with measured(stats, 'teacher_head_and_proposal_seconds', device):
                spatial, null = runtime.localization(views, ref_cache[ref['sha256']], frame)
                r.update(propose(spatial, image.size, config['proposal']))
                r.update(spatial=spatial.tolist(), teacher_null=null)
            # Structural preflight uses coordinates only; no GT changes any proposal.
            for candidate in r['candidates']:
                box = candidate['box']
                if clip_box(box, image.size) != box or not pool_weights(image.size, box).sum() > 0:
                    raise ValueError('proposal coordinate/letterbox preflight failed')
            records.append(r); frame_records.append(r)
            seen_pairs[pair] = dict(signature=signature, record=r)
        del views  # Never retain the 16 large backbone intermediates on disk or across frames.
        labels.update(labels_for_frame(root, frame, frame_records, config['label']))
        if frame_index < 3:
            from .region_ab_report import proposal_preview
            proposal_preview(output/'preview', image, frame_records)
        if (frame_index+1) % 5 == 0 or frame_index < 3:
            print(f'Teacher proposals {frame_index+1}/{len(frames)} RGB, {len(records)} pairs', flush=True)
    write_json(output/'proposals/records.json', dict(kind='teacher_regions_v1', config=config['proposal'],
               teacher_sha256=provenance['teacher_sha256'], feature_config=runtime.cfg, records=records))
    write_json(output/'labels.json', labels)
    for name in ('proposals/records.json', 'labels.json'):
        provenance['artifacts'][name] = sha(output/name)
    # Commit one fixed proposal artifact BEFORE either feature branch reads boxes.
    write_json(output/'prepared.json', provenance)
    records = read_json(output/'proposals/records.json')['records']
    by_frame = {}
    for r in records:
        by_frame.setdefault((r['scene'], r['frame']), []).append(r)
    for frame_index, frame in enumerate(frames):
        frame_records = by_frame.get((frame['scene'], frame['state']), [])
        if not frame_records:
            continue
        with Image.open(checked_path(root, frame['image'])) as im:
            image = im.convert('RGB')
        # Same immutable box list is passed to A and B; no annotations enter either encoder.
        error_a = None
        with measured(stats, 'A_feature_seconds', device):
            try:
                full = runtime.independent(image, frame['image_sha256'])
            except (ValueError, RuntimeError) as error:
                _fatal(error); full = None; error_a = str(error)
            for r in frame_records:
                k = len(r['candidates']); x = np.zeros((k, 16, 256), np.float32); valid = np.zeros((k, 16), bool)
                errors = []
                if error_a and k == 0:
                    errors.append(dict(candidate=None, error=error_a))
                for i, c in enumerate(r['candidates']):
                    if full is None:
                        errors.append(dict(candidate=i, error=error_a)); continue
                    try:
                        x[i], valid[i] = pool(full, image.size, c['box'])
                        if not valid[i].any():
                            errors.append(dict(candidate=i, error='empty pool'))
                    except ValueError as error:
                        errors.append(dict(candidate=i, error=str(error)))
                indexes['A']['entries'][r['id']] = save_features(output/'A', r['id'], x, valid, errors)
        with measured(stats, 'B_feature_seconds', device):
            for r in frame_records:
                k = len(r['candidates']); x = np.zeros((k, 16, 256), np.float32); valid = np.zeros((k, 16), bool)
                errors = []
                for i, c in enumerate(r['candidates']):
                    box = clip_box(c['box'], image.size)
                    if box is None:
                        errors.append(dict(candidate=i, error='invalid box')); continue
                    try:
                        crop = image.crop(box)
                        q = runtime.independent(crop)
                        x[i], valid[i] = pool(q, crop.size, [0, 0, crop.width, crop.height])
                        if not valid[i].any():
                            errors.append(dict(candidate=i, error='empty pool'))
                    except (ValueError, RuntimeError) as error:
                        _fatal(error); errors.append(dict(candidate=i, error=str(error)))
                indexes['B']['entries'][r['id']] = save_features(output/'B', r['id'], x, valid, errors)
        if (frame_index+1) % 5 == 0 or frame_index < 3:
            print(f'Prepared {frame_index+1}/{len(frames)} RGB, {len(records)} unique pairs; candidates={sum(len(r["candidates"]) for r in records)}', flush=True)
            write_json(output/'progress.json', dict(frames=frame_index+1, pairs=len(records), seconds=stats,
                                                   errors={v: sum(bool(e['errors']) for e in indexes[v]['entries'].values()) for v in indexes}))
    for name in ('proposals/records.json', 'labels.json'):
        if sha(output/name) != provenance['artifacts'][name]:
            raise ValueError('shared candidate/label file changed during feature extraction')
    for v in indexes:
        indexes[v].update(proposal_sha256=provenance['artifacts']['proposals/records.json'],
                          labels_sha256=provenance['artifacts']['labels.json'])
        write_json(output/v/'cache.json', indexes[v])
        provenance['artifacts'][v+'/cache.json'] = sha(output/v/'cache.json')
    stats.update(unique_rgb=len({r['image_sha256'] for r in records}), unique_pairs=len(records),
                 candidates=sum(len(r['candidates']) for r in records),
                 gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                 gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device), dense_cache_hits=runtime.cache_hits,
                 cache_bytes={v: sum(p.stat().st_size for p in (output/v).rglob('*') if p.is_file()) for v in ('A', 'B', 'proposals', 'references')},
                 timing_scope='Synchronized preparation wall time includes disk writes. A full encoding shared across references of one RGB; teacher backbone shared too. B per candidate per reference. No online FPS claim; matcher timing reported separately.',
                 online_measurement_available=False)
    verify_sources(root, sources)
    verify_sources(runtime.repo, provenance['backbone_source'])
    if sha(args.teacher) != runtime.teacher_sha256 or sha(runtime.backbone_path) != runtime.cfg['backbone_sha256']:
        raise ValueError('weights changed during feature preparation')
    provenance.update(status='complete', preparation=stats)
    write_json(output/'prepared.json', provenance)
    print('Prepared cache complete:', output, flush=True)
    return provenance
