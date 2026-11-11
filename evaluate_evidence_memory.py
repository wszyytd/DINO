"""Recheck saved stage-one checkpoints, including fixed fitting-side diagnostics."""
import argparse
import json
from pathlib import Path
import torch
from dinoexp.dense_cache import save_json
from dinoexp.evidence_data import load_evidence_cache
from dinoexp.evidence_memory import EvidenceMatcher
from dinoexp.evidence_training import partition_scenes, make_routes, evaluate
from dinoexp.local_features import resolve_device
from dinoexp.local_pairs import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    run=json.loads((args.run/'run.json').read_text())
    if run['status']!='complete':
        raise ValueError('training run incomplete')
    if sha256(args.cache/'cache.json')!=run['cache_manifest_sha256']:
        raise ValueError('cache differs from training run')
    meta, scenes=load_evidence_cache(args.cache)
    partitions, scope=partition_scenes(scenes,meta)
    fit=make_routes(scenes,partitions['train'],run['args']['routes'],run['args']['length'],902)
    probe=json.loads((args.run/'probe_routes.json').read_text())
    device=resolve_device(args.device,torch)
    results=[]
    for row in run['results']:
        path=args.run/f"{row['mode']}_seed{row['seed']}"/'last.pt'
        cp=torch.load(path,map_location='cpu',weights_only=True)
        if cp['feature_contract']!=meta['contract']:
            raise ValueError('checkpoint/feature mismatch')
        model=EvidenceMatcher(**cp['config']).to(device);model.load_state_dict(cp['model_state'])
        fitting=evaluate(model,fit,scenes,device,run['args']['batch_size'])
        replay=evaluate(model,probe,scenes,device,run['args']['batch_size'])
        if replay['metrics']!=row['metrics']:
            raise ValueError('saved checkpoint did not reproduce reported metrics')
        results.append(dict(mode=row['mode'],seed=row['seed'],checkpoint_sha256=sha256(path),
                            fit=fitting['metrics'],probe=replay['metrics']))
        print(row['mode'],row['seed'],'fit both-visible',fitting['metrics']['both_visible'],flush=True)
    save_json(args.run/'checkpoint_recheck.json',dict(scope=scope,results=results))


if __name__=='__main__':
    main()
