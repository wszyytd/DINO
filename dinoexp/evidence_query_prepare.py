"""CUDA frozen feature preparation. No network invocation from pure-data CLI paths."""
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .core import checked_path
from .region_ab_data import read_json, write_json, sha, digest_json
from .region_ab_prepare import FrozenRuntime, pool, cuda_device
from .evidence_query_data import (validate_manifest,render_sample,pixel_hash,code_hashes,
                                 load_old_observation,verify_code_sources)


def encode_crop(runtime, crop, role):
    if role=='queries':
        _,_,tokens,valid=runtime.reference(crop,pixel_hash(crop))
    elif role=='observations':
        tokens,valid=pool(runtime.independent(crop),crop.size,[0,0,*crop.size])
    else: raise ValueError('unknown feature role')
    return tokens.astype(np.float32),valid.astype(bool)


@torch.inference_mode()
def prepare(args):
    output=Path(args.output); manifest_path=output/'manifest.json'; m=read_json(manifest_path)
    validate_manifest(m)
    if (output/'prepared.json').exists() or (output/'features').exists():
        raise ValueError('cache already started; preserve it and use a new output')
    device=cuda_device(args.device); torch.set_num_threads(args.threads)
    root=Path(args.data or m['data_root']); source=Path(args.source_run or m['source_run'])
    for name,digest in m['source_hashes'].items():
        if sha(source/name)!=digest: raise ValueError('source run hash mismatch: '+name)
    codes=code_hashes(); mh=sha(manifest_path)
    verify_code_sources(Path(__file__).resolve().parents[1],m['encoder_source'].get('code') or {})
    runtime=FrozenRuntime(args)
    enc=m['encoder_source']
    if runtime.teacher_sha256!=enc['teacher_sha256'] or sha(runtime.backbone_path)!=enc['backbone_sha256'] or runtime.cfg!=enc['teacher_feature_config']:
        raise ValueError('source teacher/backbone/preprocessing mismatch')
    verify_code_sources(runtime.repo,enc.get('backbone_source') or {})
    if any(p.requires_grad for p in list(runtime.teacher.parameters())+list(runtime.backbone.parameters())):
        raise ValueError('encoder is not frozen')
    encoder=dict(teacher=runtime.teacher_sha256,backbone=sha(runtime.backbone_path),preprocessing=runtime.cfg,
        backbone_source=enc.get('backbone_source'),
        branches=dict(observations='independent+pool 16x256',queries='reference 64x256'),code_hashes=codes)
    p=dict(status='running',manifest_sha256=mh,manifest_digest=digest_json(m),encoder_hashes=encoder,
        encoder_signature=digest_json(encoder),code_hashes=codes,observations={},queries={},
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device)))
    feature_dir=output/'features'/p['encoder_signature']
    feature_dir.mkdir(parents=True); write_json(output/'prepared.json',p); start=time.perf_counter()
    # Old clean caches are accepted only through the verified old plan and branch contract.
    old_path=source/'prepared.json'; old=read_json(old_path) if old_path.exists() else None
    if old and (old.get('status')!='complete' or old.get('kind')!='region_sequence_features_v1'
                or old.get('plan_sha256')!=m['source_hashes']['plan.json']):
        raise ValueError('old cache provenance mismatch')
    reused=0
    for ep in m['episodes']:
        for st in ep['steps']:
            for role,items in (('observations',st['observations']),('queries',st['queries'])):
                for item in items:
                    key=item['feature_key']
                    if key in p[role]: continue
                    sample=m['samples'][item['sample_id']]; perturb=item.get('perturbation',{'kind':'clean'})
                    crop=render_sample(root,sample,perturb) # Validate RGB even when reusing a feature.
                    reuse=old and role=='observations' and perturb['kind']=='clean' and sample['source_observation_key'] in old['observations']
                    if reuse:
                        entry=old['observations'][sample['source_observation_key']]
                        tokens,valid=load_old_observation(entry,source)
                        reused+=1
                    else: tokens,valid=encode_crop(runtime,crop,role)
                    n=16 if role=='observations' else 64
                    if tokens.shape!=(n,256) or valid.shape!=(n,) or not valid.any() or not np.isfinite(tokens).all():
                        raise ValueError('invalid encoded feature')
                    path=feature_dir/f'{key}.npz'; np.savez_compressed(path,tokens=tokens.astype(np.float32),valid=valid.astype(bool))
                    p[role][key]=dict(file=path.relative_to(output).as_posix(),sha256=sha(path),role=role,
                        sample_id=item['sample_id'],sample_digest=digest_json(sample),parent_source_sha256=sample['parent_source_sha256'],
                        perturbation=perturb,encoder_signature=p['encoder_signature'],
                        cache_key=digest_json([p['encoder_signature'],key]),reused=bool(reuse))
    if mh!=sha(manifest_path) or codes!=code_hashes() or sha(args.teacher)!=encoder['teacher'] or sha(runtime.backbone_path)!=encoder['backbone']:
        raise ValueError('inputs changed during encoding')
    for sample in m['samples'].values():
        if sha(checked_path(root,sample['source_rgb']))!=sample['source_rgb_sha256']: raise ValueError('RGB changed during encoding')
    verify_code_sources(runtime.repo,enc.get('backbone_source') or {})
    for name,digest in m['source_hashes'].items():
        if sha(source/name)!=digest: raise ValueError('source run changed during encoding')
    p.update(status='complete',seconds=time.perf_counter()-start,reused_clean_features=reused,
        frozen=True,encoder_parameters_optimized=False)
    write_json(output/'prepared.json',p)
    return dict(status=p['status'],observations=len(p['observations']),queries=len(p['queries']),reused=reused)
