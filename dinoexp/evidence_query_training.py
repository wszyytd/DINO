"""Short-episode BPTT over frozen feature caches; no encoder in optimizer."""
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .region_ab_data import read_json, write_json, sha, digest_json
from .region_ab_prepare import cuda_device
from .region_sequence_training import make_model, advance, parameter_hash
from .evidence_query_data import load_bundle, pack_query_batch, rng_for, code_hashes, training_weights

VARIANTS=dict(A_last=('last',True,False),B_mean=('mean',True,False),
    C_memory_current=('memory',False,False),D_memory_history=('memory',True,False),
    E_memory_repeat=('memory',True,True))


def clone_state(state):
    return {k:v.clone() for k,v in state.items()}


def score_queries(model,state,queries,valid,active):
    b,j=queries.shape[:2]; k=state['seen'].shape[1]
    if valid.shape!=queries.shape[:-1] or active.shape!=(b,j): raise ValueError('invalid query shapes')
    if (active & ~valid.any(-1)).any(): raise ValueError('active query has no valid tokens')
    # Keep a differentiable zero for an empty query set; never pass it to old score.
    result=queries.new_zeros((b,j,k+1))+state['slots'].sum()*0
    bi,ji=active.nonzero(as_tuple=True)
    if len(bi):
        selected={key:value[bi] for key,value in state.items()}
        logits=model.score(queries[bi,ji],valid[bi,ji],selected)
        available=torch.cat([selected['seen'],torch.ones(len(bi),1,dtype=torch.bool,device=queries.device)],-1)
        logits=logits.masked_fill(~available,-1e4)
        result[bi,ji]=logits
    return result


def _reduce(values,mask):
    """Mean queries within step, valid steps within episode, eligible episodes."""
    per_step_count=mask.sum(-1); step_active=per_step_count>0
    per_step=(values.masked_fill(~mask,0).sum(-1)/per_step_count.clamp_min(1))
    steps=step_active.sum(-1); ep_active=steps>0
    per_ep=per_step.sum(-1)/steps.clamp_min(1)
    loss=per_ep.sum()/ep_active.sum().clamp_min(1)
    return loss,dict(queries=int(mask.sum().item()),steps=int(step_active.sum().item()),
        episodes=int(ep_active.sum().item()),queries_per_step=per_step_count.detach().cpu().tolist(),
        steps_per_episode=steps.detach().cpu().tolist())


def query_losses(normal_logits,repeat_logits,batch,config,class_weights):
    if class_weights.shape!=(2,): raise ValueError('weights are [target, Null]')
    active=batch['query_active'] & batch['step_valid'][...,None]
    labels=batch['query_label']; k=normal_logits.shape[-1]-1
    if ((labels[active]<0)|(labels[active]>k)).any(): raise ValueError('query label outside candidates/Null')
    # Inactive labels never index CE, including padding supplied by external callers.
    safe=labels.masked_fill(~active,k)
    ce=F.cross_entropy(normal_logits.reshape(-1,k+1),safe.reshape(-1),reduction='none').reshape(labels.shape)
    weights=class_weights[(safe==k).long()]
    counts={}; losses={}; raw={}
    for name,group in (('current',0),('history',1)):
        mask=active & (batch['query_group']==group)
        losses[name],c=_reduce(ce*weights,mask); raw[name],_=_reduce(ce,mask)
        counts.update({name+'_'+key:value for key,value in c.items()})
    repeat_mask=torch.zeros_like(active)
    kl=normal_logits.sum(-1)*0
    if repeat_logits is not None:
        repeat_mask=active & batch['repeat_active'][...,None] & (labels!=k) & (normal_logits.detach().argmax(-1)==labels)
        kl=F.kl_div(F.log_softmax(repeat_logits,-1),F.softmax(normal_logits.detach(),-1),reduction='none').sum(-1)
    losses['repeat'],c=_reduce(kl,repeat_mask)
    counts.update({'repeat_'+key:value for key,value in c.items()})
    losses['total']=losses['current']+config['history_weight']*losses['history']+config['repeat_weight']*losses['repeat']
    losses['unweighted_ce']=raw; losses['counts']=counts
    return losses


def rollout_queries(model,method,batch,config):
    b,t,k=batch['observation'].shape[:3]
    state=model.initial_state(b,k,batch['observation'].device,batch['observation'].dtype)
    normal,repeat,repeat_active=[],[],[]; advances=0; query_reads=0
    for ti in range(t):
        obs=batch['observation'][:,ti]
        valid=batch['observation_valid'][:,ti] & batch['step_valid'][:,ti,None,None]
        previous_seen=state['seen']
        state=advance(model,method,obs,valid,state); advances+=1
        active=batch['query_active'][:,ti] & batch['step_valid'][:,ti,None]
        queries=dict(queries=batch['query'][:,ti],valid=batch['query_valid'][:,ti],active=active)
        logits=score_queries(model,state,**queries); normal.append(logits); query_reads+=int(active.sum().item())
        rc=batch['repeat_candidate'][:,ti]; chosen=torch.zeros_like(valid)
        enabled=torch.zeros(b,dtype=torch.bool,device=obs.device)
        if config['repeat_weight']>0:
            for bi in range(b):
                ci=int(rc[bi])
                if ci>=0 and bool(previous_seen[bi,ci]) and bool(valid[bi,ci].any()):
                    chosen[bi,ci]=valid[bi,ci]; enabled[bi]=True
        if enabled.any():
            extra=advance(model,method,obs,chosen,clone_state(state)); advances+=1
            repeat.append(score_queries(model,extra,**queries)); query_reads+=int(active.sum().item())
        else: repeat.append(logits)
        repeat_active.append(enabled)
    return dict(normal_logits=torch.stack(normal,1),repeat_logits=torch.stack(repeat,1),
        batch={**batch,'repeat_active':torch.stack(repeat_active,1)},final_state=state,
        forward_counts=dict(advance_calls=advances,query_reads=query_reads))


def tensor_batch(arrays,device):
    return {key:torch.from_numpy(value).to(device) for key,value in arrays.items()}


def train(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    device=cuda_device(args.device); torch.set_num_threads(args.threads)
    output=Path(args.output); m,p=load_bundle(output); cfg=m['config']; codes=code_hashes()
    episodes=[e for e in m['episodes'] if e['split']=='train']
    if not episodes: raise ValueError('no train episodes; probe is not a fallback')
    train_history=sum(q['group']==1 for ep in episodes for st in ep['steps'] for q in st['queries'])
    if cfg['group_mode']=='grouped' and not train_history: raise ValueError('grouped train has no history supervision')
    updates=cfg['smoke_updates'] if args.budget=='smoke' else cfg['full_updates']
    if args.budget=='smoke' and not 1<=updates<=200: raise ValueError('smoke budget must be 1..200')
    if args.budget=='full' and updates!=2300: raise ValueError('full protocol requires fixed 2300 updates')
    variants=list(VARIANTS) if args.variant=='all' else [args.variant]
    seeds=[7] if args.budget=='smoke' else cfg['seeds']
    if args.budget=='full':
        smoke=read_json(output/'runs'/'smoke'/'run.json')
        if smoke['status']!='complete' or smoke['manifest_digest']!=digest_json(m):
            raise ValueError('complete a matching smoke run before full training')
        for v in variants:
            if not any(x['variant']==v and x['reload_max_error']<=2e-5 for x in smoke['experiments']):
                raise ValueError('matching smoke checkpoint reload check required: '+v)
        evaluated=list((output/'evaluation'/'smoke').glob('*/records.json'))
        if not any(read_json(f).get('status')=='complete' and read_json(f).get('manifest_digest')==digest_json(m) for f in evaluated):
            raise ValueError('complete smoke evaluation before full matrix; no accuracy threshold is applied')
    run_root=output/'runs'/args.budget
    if run_root.exists(): raise ValueError('run already exists; never overwrite training results')
    run_root.mkdir(parents=True)
    run=dict(status='running',budget=args.budget,config=cfg,manifest_digest=digest_json(m),
        prepared_sha256=sha(output/'prepared.json'),code_hashes=codes,experiments=[],
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device)),
        history_effective=train_history>0,
        scope='joint Writer/Updater/Gate/Reader/Matcher training; frozen encoder caches; identity-assisted')
    write_json(run_root/'run.json',run)
    weights,frequency=training_weights(episodes)
    run.update(class_weights=weights.tolist(),class_counts=frequency)
    write_json(run_root/'run.json',run)
    for seed in seeds:
        gen=rng_for(seed,'schedule')
        schedule=gen.integers(len(episodes),size=(updates,cfg['batch_size'])).tolist()
        schedule_ids=[[episodes[i]['id'] for i in row] for row in schedule]
        write_json(run_root/f'schedule-{seed}.json',schedule_ids)
        shared_hash=None
        for variant in variants:
            method,history,repeat=VARIANTS[variant]
            loss_cfg=dict(cfg,history_weight=cfg['history_weight'] if history else 0.,repeat_weight=cfg['repeat_weight'] if repeat else 0.)
            model=make_model(cfg,seed,device,method); model.train()
            initial_hash=parameter_hash(model,common=True)
            if shared_hash is not None and shared_hash!=initial_hash: raise ValueError('unfair common initialization')
            shared_hash=initial_hash
            directory=run_root/f'{variant}-s{seed}'; directory.mkdir()
            torch.save(model.state_dict(),directory/'initial.pt')
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            cw=torch.from_numpy(weights).to(device); rows=[]; start=time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device); counts=dict(advance_calls=0,query_reads=0)
            for update,indices in enumerate(schedule):
                batch=tensor_batch(pack_query_batch([episodes[i] for i in indices],p,output),device)
                optimizer.zero_grad(set_to_none=True)
                out=rollout_queries(model,method,batch,loss_cfg)
                loss=query_losses(out['normal_logits'],out['repeat_logits'],out['batch'],loss_cfg,cw)
                if not torch.isfinite(loss['total']): raise ValueError('nonfinite training loss')
                loss['total'].backward() # exactly once, after the complete short sequence
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True)
                optimizer.step()
                for key in counts: counts[key]+=out['forward_counts'][key]
                rows.append(dict(update=update+1,loss={key:float(loss[key].detach()) for key in ('total','current','history','repeat')},
                    applied_loss=dict(current=float(loss['current'].detach()),history=float(loss['history'].detach())*loss_cfg['history_weight'],
                        repeat=float(loss['repeat'].detach())*loss_cfg['repeat_weight']),
                    unweighted_ce={key:float(value.detach()) for key,value in loss['unweighted_ce'].items()},
                    counts=loss['counts'],gradient_norm=float(norm)))
                if (update+1)%10==0 or update+1==updates:
                    write_json(directory/'log.json',rows)
            torch.cuda.synchronize(device)
            checkpoint=dict(kind='evidence_query_v1',variant=variant,seed=seed,model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),config=cfg,loss_config=loss_cfg,updates=updates,
                manifest_digest=digest_json(m),prepared_sha256=run['prepared_sha256'],code_hashes=codes,
                common_initial_hash=initial_hash,schedule_digest=digest_json(schedule_ids),class_counts=frequency,
                class_weights=weights.tolist(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(device))
            torch.save(checkpoint,directory/'final.pt')
            # Reuse fixed TRAIN inputs; this is reload validation, never probe tuning.
            model.eval()
            with torch.inference_mode(): expected=rollout_queries(model,method,batch,loss_cfg)['normal_logits']
            del optimizer,model,out,loss
            loaded=make_model(cfg,seed,device,method)
            loaded.load_state_dict(torch.load(directory/'final.pt',map_location=device,weights_only=True)['model_state']); loaded.eval()
            with torch.inference_mode(): actual=rollout_queries(loaded,method,batch,loss_cfg)['normal_logits']
            error=float((expected-actual).abs().max())
            if error>2e-5: raise ValueError('checkpoint reload mismatch')
            np.savez_compressed(directory/'reload_logits.npz',expected=expected.cpu().numpy(),actual=actual.cpu().numpy())
            record=dict(variant=variant,seed=seed,common_initial_hash=initial_hash,schedule_digest=digest_json(schedule_ids),
                checkpoint=(directory/'final.pt').relative_to(output).as_posix(),checkpoint_sha256=sha(directory/'final.pt'),
                reload_max_error=error,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(device),
                forward_counts=counts,updates=updates,loss_config=loss_cfg)
            run['experiments'].append(record); write_json(run_root/'run.json',run)
            del loaded,batch,expected,actual
    if codes!=code_hashes(): raise ValueError('code changed during training')
    run['status']='complete'; write_json(run_root/'run.json',run)
    return dict(status='complete',experiments=len(run['experiments']),budget=args.budget)
