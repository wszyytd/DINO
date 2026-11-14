"""CUDA evaluation of independent queries and isolated single-car interventions."""
import os
from pathlib import Path
import time

import torch

from .region_ab_data import read_json,write_json,sha,digest_json
from .region_ab_prepare import cuda_device
from .region_sequence_training import make_model,advance
from .evidence_query_data import load_bundle,pack_query_batch
from .evidence_query_training import score_queries,clone_state,tensor_batch,VARIANTS
from .evidence_query_metrics import describe_scores,summarize_updates


def single_vehicle_control(model,method,before,observation,valid,queries,candidate_index):
    if before is None:
        before=model.initial_state(observation.shape[0],observation.shape[1],observation.device,observation.dtype)
    full=advance(model,method,observation,valid,clone_state(before))
    skip_valid=valid.clone(); skip_valid[:,candidate_index]=False
    skip=advance(model,method,observation,skip_valid,clone_state(before))
    eligible=before['seen'][:,candidate_index] & valid[:,candidate_index].any(-1)
    if eligible.any() and not torch.equal(full['seen'][eligible],skip['seen'][eligible]):
        raise ValueError('paired candidate availability mismatch')
    return dict(full_state=full,skip_state=skip,full_logits=score_queries(model,full,**queries),
        skip_logits=score_queries(model,skip,**queries),eligible=eligible,
        first_write=~before['seen'][:,candidate_index] & valid[:,candidate_index].any(-1))


def repeat_control(model,method,after,observation,valid,queries,candidate_index):
    chosen=torch.zeros_like(valid); chosen[:,candidate_index]=valid[:,candidate_index]
    state=clone_state(after); results=[]
    for i in range(1,9):
        state=advance(model,method,observation,chosen,state)
        if i in (1,8): results.append(dict(repetitions=i,state=clone_state(state),logits=score_queries(model,state,**queries)))
    return results


@torch.inference_mode()
def evaluate(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    device=cuda_device(args.device); torch.set_num_threads(args.threads)
    output=Path(args.output); m,p=load_bundle(output)
    run=read_json(output/'runs'/args.budget/'run.json')
    if run['status']!='complete' or run['manifest_digest']!=digest_json(m) or run['prepared_sha256']!=sha(output/'prepared.json'):
        raise ValueError('training provenance mismatch / incomplete')
    episodes=[ep for ep in m['episodes'] if ep['split']==args.split]
    if not episodes: raise ValueError('requested split has no episodes')
    target=output/'evaluation'/args.budget/args.split
    if target.exists(): raise ValueError('evaluation already exists; preserve saved results')
    target.mkdir(parents=True)
    payload=dict(status='running',synthetic=False,budget=args.budget,split=args.split,manifest_digest=digest_json(m),
        run_sha256=sha(output/'runs'/args.budget/'run.json'),rows=[],assistance='simulation identity + mask region routing')
    write_json(target/'records.json',payload)
    rows=[]; start=time.perf_counter()
    for experiment in run['experiments']:
        checkpoint=output/experiment['checkpoint']
        if sha(checkpoint)!=experiment['checkpoint_sha256']: raise ValueError('checkpoint hash mismatch')
        cp=torch.load(checkpoint,map_location=device,weights_only=True)
        variant=experiment['variant']; method=VARIANTS[variant][0]
        if cp['manifest_digest']!=digest_json(m) or cp['variant']!=variant: raise ValueError('checkpoint provenance mismatch')
        model=make_model(m['config'],experiment['seed'],device,method)
        model.load_state_dict(cp['model_state']); model.eval()
        method_rows=[]
        for ep in episodes:
            b=tensor_batch(pack_query_batch([ep],p,output),device); ids=ep['identities']
            state=model.initial_state(1,len(ids),device,b['observation'].dtype)
            past={uid:set() for uid in ids}
            for t,st in enumerate(ep['steps']):
                before=state; obs=b['observation'][:,t]; valid=b['observation_valid'][:,t]
                queries=dict(queries=b['query'][:,t],valid=b['query_valid'][:,t],active=b['query_active'][:,t])
                state=advance(model,method,obs,valid,clone_state(before))
                normal=score_queries(model,state,**queries)
                seen=[uid for i,uid in enumerate(ids) if bool(state['seen'][0,i])]
                def base(q):
                    sample=m['samples'][q['sample_id']]
                    return dict(episode_id=ep['id'],scene_id=ep['scene_id'],step=t,query_id=q['sample_id'],
                        query_identity=sample['vehicle_uid'],query_group=q['group'],query_view_group=sample['view_group'],
                        variant=variant,seed=experiment['seed'],identities=ids,seen=seen,
                        source_reference_id=ep['source_reference_id'],written_ids=[o['sample_id'] for o in st['observations'] if o['write']])
                descriptions=[]
                for j,q in enumerate(st['queries']):
                    d=describe_scores(normal[0,j].cpu().tolist(),ids,seen,q['label']); descriptions.append(d)
                    method_rows.append(dict(base(q),kind='query',vehicle_uid=m['samples'][q['sample_id']]['vehicle_uid'],
                        view_group=m['samples'][q['sample_id']]['view_group'],**d))
                for item in st['observations']:
                    if not item['write']: continue
                    uid=item['vehicle_uid']; i=ids.index(uid); sample=m['samples'][item['sample_id']]
                    control=single_vehicle_control(model,method,before,obs,valid,queries,i)
                    skip_seen=[u for ki,u in enumerate(ids) if bool(control['skip_state']['seen'][0,ki])]
                    diagnostic=dict(vehicle_uid=uid,view_group=sample['view_group'],quality=item['perturbation']['kind'],
                        first_write=bool(control['first_write'][0]),candidate_complementary=(sample['view_group'] not in past[uid]) if sample['view_group'] is not None else None,
                        observation_id=item['sample_id'],observation_feature_key=item['feature_key'])
                    for j,q in enumerate(st['queries']):
                        full=descriptions[j]; skip=describe_scores(control['skip_logits'][0,j].cpu().tolist(),ids,skip_seen,q['label'])
                        method_rows.append(dict(base(q),**diagnostic,kind='update',eligible=bool(control['eligible'][0]),
                            full_correct=full['correct'],skip_correct=skip['correct'],full_choice=full['choice'],skip_choice=skip['choice'],
                            full_scores=full['scores'],skip_scores=skip['scores'],skip_seen=skip_seen,label=q['label'],
                            margin_delta=full['margin']-skip['margin'] if full['margin'] is not None and skip['margin'] is not None else None,
                            null_probability_delta=full['null_probability']-skip['null_probability']))
                    for control_repeat in repeat_control(model,method,state,obs,valid,queries,i):
                        for j,q in enumerate(st['queries']):
                            repeated=describe_scores(control_repeat['logits'][0,j].cpu().tolist(),ids,seen,q['label'])
                            method_rows.append(dict(base(q),**diagnostic,kind='repeat',repetitions=control_repeat['repetitions'],
                                full_correct=descriptions[j]['correct'],full_choice=descriptions[j]['choice'],
                                repeat_choice=repeated['choice'],full_scores=descriptions[j]['scores'],repeat_scores=repeated['scores'],label=q['label']))
                for item in st['observations']:
                    if item['write']: past[item['vehicle_uid']].add(m['samples'][item['sample_id']]['view_group'])
        write_json(target/f'{variant}-s{experiment["seed"]}.json',dict(rows=method_rows,status='complete'))
        rows.extend(method_rows); del model,cp
    torch.cuda.synchronize(device)
    payload.update(status='complete',rows=rows,seconds=time.perf_counter()-start,
        timing_scope='cached update/read/control evaluation + IO; excludes RGB encoding')
    write_json(target/'records.json',payload); write_json(target/'summary.json',summarize_updates(rows))
    return dict(status='complete',rows=len(rows),split=args.split,budget=args.budget)
