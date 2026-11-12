"""CUDA-only extraction of mask-assisted RGB crops, retaining frozen DINO features."""
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .core import checked_path
from .region_ab_data import sha, write_json, verify_sources
from .region_ab_prepare import FrozenRuntime, cuda_device, pool
from .region_sequence_data import load_plan


@torch.inference_mode()
def prepare(args):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    meta, records, _ = load_plan(output, root)
    if any((output/name).exists() for name in ('features', 'references', 'prepared.json')):
        raise ValueError('sequence features already started; preserve them and choose a new output')
    device = cuda_device(args.device)
    torch.set_num_threads(args.threads)
    runtime = FrozenRuntime(args)
    if (runtime.teacher_sha256 != meta['teacher_sha256']
            or sha(runtime.backbone_path) != meta['backbone_sha256']
            or runtime.cfg != meta['teacher_feature_config']):
        raise ValueError('sequence encoder weights differ from source teacher-region experiment')
    for name, expected in meta['backbone_source'].items():
        if sha(runtime.repo/name) != expected:
            raise ValueError('external DINO source differs from teacher-region experiment: '+name)
    (output/'features').mkdir()
    (output/'references').mkdir()
    provenance = dict(kind='region_sequence_features_v1', status='running',
                      plan_sha256=sha(output/'plan.json'), observations={}, references={},
                      feature_contract='Original RGB crop -> frozen ViT-B/14 + independent adapter '
                                       '-> 4x4/16x256 tokens. No teacher score or ID as features.',
                      assistance=meta['assistance'], device=str(device),
                      environment=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                                       gpu=torch.cuda.get_device_name(device), numpy=str(np.__version__)))
    write_json(output/'prepared.json', provenance)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    values = list(records.values())
    for i, record in enumerate(values):
        digest = record['reference_sha256']
        if digest not in provenance['references']:
            with Image.open(checked_path(root, record['reference'])) as im:
                _, _, tokens, valid = runtime.reference(im.convert('RGB'), digest)
            file = 'references/'+digest+'.npz'
            np.savez_compressed(output/file, tokens=tokens, valid=valid)
            provenance['references'][digest] = dict(file=file, sha256=sha(output/file))
        wanted = [o for o in record['observations'] if o['write_allowed']
                  and o['key'] not in provenance['observations']]
        if wanted:
            with Image.open(checked_path(root, record['image'])) as im:
                image = im.convert('RGB')
            for observation in wanted:
                crop = image.crop(observation['box'])
                q = runtime.independent(crop)
                tokens, valid = pool(q, crop.size, [0, 0, crop.width, crop.height])
                if not valid.any() or not np.isfinite(tokens).all():
                    raise ValueError('failed sequence crop; never silently relabel: '+record['id'])
                key = observation['key']
                file = 'features/'+key+'.npz'
                np.savez_compressed(output/file, observation=tokens, valid=valid)
                provenance['observations'][key] = dict(file=file, sha256=sha(output/file))
        if i == 0 or (i+1) % 20 == 0:
            write_json(output/'prepared.json', provenance)
            print(f'Sequence features {i+1}/{len(values)} pairs; unique crops='
                  f'{len(provenance["observations"])}', flush=True)
    torch.cuda.synchronize(device)
    verify_sources(root, meta['sources'])
    load_plan(output, root)
    if sha(output/'plan.json') != provenance['plan_sha256']:
        raise ValueError('sequence plan changed during extraction')
    for name, expected in meta['backbone_source'].items():
        if sha(runtime.repo/name) != expected:
            raise ValueError('external DINO source changed during extraction: '+name)
    if (sha(args.teacher) != runtime.teacher_sha256
            or sha(runtime.backbone_path) != meta['backbone_sha256']):
        raise ValueError('weights changed during sequence extraction')
    provenance.update(status='complete', seconds=time.perf_counter()-start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    write_json(output/'prepared.json', provenance)
    print('Sequence features complete:', output, flush=True)
    return provenance
