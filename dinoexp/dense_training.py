"""Dense supervision, stage-specific freezing and original-image evaluation."""
import csv
import html
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from .core import checked_path
from .dense_cache import DenseFeatureCache,save_json
from .dense_data import read_dense_manifest,make_targets
from .dense_loss import dense_loss
from .dense_model import DenseSearchHead
from .dense_views import camera_embedding
from .local_features import resolve_device
from .local_pairs import sha256


def sample_tensors(samples,cache,device):
    items=[]
    for row in samples:
        r=cache.get('reference:'+row['reference_sha256'])
        v=cache.get('image:'+row['image_sha256'])
        state,has_state=camera_embedding(row.get('camera_pose'),row.get('fov'))
        items.append(dict(reference=r['tokens'],reference_valid=r['valid'],views=v['tokens'],
            view_valid=v['valid'],**{k:v[k] for k in ('geometry','indices','weights','priors','view_weights')},
            camera_state=state,camera_valid=np.array(has_state)))
    result={}
    for key in items[0]:
        tensor=torch.from_numpy(np.stack([i[key] for i in items])).to(device)
        result[key]=tensor.long() if key=='indices' else (tensor.bool() if key in (
            'reference_valid','view_valid','camera_valid') else tensor.float())
    return result


def _load(args):
    data=read_dense_manifest(args.manifest,args.data_root)
    cache=DenseFeatureCache(args.cache)
    if cache.manifest_sha256!=data['_manifest_sha256']:
        raise ValueError('dense cache manifest hash differs from labels')
    return data,cache


def load_head(path,cache,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('kind')!='dense_reference_search_v1' or checkpoint['feature_config']!=cache.config:
        raise ValueError('need compatible dense checkpoint; crop matcher weights cannot be reused')
    model=DenseSearchHead(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    return model,checkpoint


def _predict(model,rows,cache,root,device,gamma=2.,null_weight=1.):
    model.eval();outputs=[];losses=[]
    with torch.no_grad():
        for row in rows:
            out=model(**sample_tensors([row],cache,device))
            label=make_targets(row,root,cache.config['grid_size'])
            target=torch.as_tensor(label['target'][:-1],device=device)
            if ((out['coverage'][0]<=0)&(target>0)).any():
                raise ValueError('positive target lies outside native-grid coverage; reduce grid size')
            losses.append(float(dense_loss(out['probabilities'],[label],gamma,null_weight)))
            outputs.append(out['probabilities'][0].cpu().numpy())
    if not rows:
        raise ValueError('no usable dense evaluation samples')
    return outputs,float(np.mean(losses))


def _metrics(rows,outputs,root,grid):
    present=top1=top5=exists=known=0
    for row,p in zip(rows,outputs,strict=True):
        label=make_targets(row,root,grid)
        if row['presence']!='absent':
            target=label['target'][:-1]>0
            order=np.argsort(-p[:-1])
            present+=1;top1+=bool(target[order[0]]);top5+=bool(target[order[:5]].any())
        if row['presence']!='unknown':
            known+=1;exists+=bool((p[-1]>=.5)==(row['presence']=='absent'))
    return dict(samples=len(rows),present_samples=present,spatial_top1_hits=top1,
        spatial_top1=(top1/present if present else None),spatial_top5=(top5/present if present else None),
        existence_at_0_5=(exists/known if known else None),
        warning='Spatial grid hits are not vehicle identity accuracy; Null threshold is uncalibrated.')


def train_dense(args):
    data,cache=_load(args)
    train=[r for r in data['samples'] if r['split']=='train']
    validation=[r for r in data['samples'] if r['split']=='val']
    mode='simulation' if args.stage=='simulation' else 'confirmed_regions'
    if (not train or not validation or any(r['supervision']!=mode for r in train+validation)
            or args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.threads<1):
        raise ValueError('need usable train/val samples for selected stage and positive training settings')
    if Path(args.output).exists():
        raise ValueError('training output exists; choose a new directory')
    labels={r['id']:make_targets(r,data['_root'],cache.config['grid_size']) for r in train+validation}
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device=resolve_device(args.device,torch)
    if args.stage=='real':
        if not args.checkpoint:
            raise ValueError('real adaptation requires a simulation dense checkpoint')
        model,source=load_head(args.checkpoint,cache,device)
        if source['stage'] not in ('simulation','real'):
            raise ValueError('real adaptation needs simulation-trained dense weights')
        val_hashes={r[k] for r in validation for k in ('reference_sha256','image_sha256')}
        if (set(source['training_images'])&val_hashes or
                set(source['training_groups'])&{r['group'] for r in validation}):
            raise ValueError('validation images/groups overlap checkpoint pretraining')
        model.freeze_real_branches()
    else:
        if args.checkpoint:
            raise ValueError('simulation starts task modules from scratch; checkpoint is for real stage')
        model=DenseSearchHead(input_dim=cache.config['input_dim'],dim=args.dim,
            layers=len(cache.config['layers']),heads=args.heads,grid_size=cache.config['grid_size']).to(device)
        source=None
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    output=Path(args.output);output.mkdir(parents=True)
    run=dict(status='running',stage=args.stage,train_samples=len(train),validation_samples=len(validation),
        feature_config=cache.config,model_config=model.config,manifest_sha256=data['_manifest_sha256'],
        seed=args.seed,device=str(device),epochs=args.epochs,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        warning='Teacher-inspired self-implementation; no teacher task weights. Performance unverified.')
    save_json(output/'training_run.json',run)
    history=[];best=float('inf')
    provenance=dict(kind='dense_reference_search_v1',stage=args.stage,feature_config=cache.config,
        model_config=model.config,manifest_sha256=data['_manifest_sha256'],
        training_images=sorted({r[k] for r in train for k in ('reference_sha256','image_sha256')}
                               |(set(source['training_images']) if source else set())),
        training_groups=sorted({r['group'] for r in train}|(set(source['training_groups']) if source else set())))
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(train)).tolist();total=0.
        for start in range(0,len(order),args.batch_size):
            rows=[train[i] for i in order[start:start+args.batch_size]]
            target=[labels[r['id']] for r in rows]
            optimizer.zero_grad(set_to_none=True)
            out=model(**sample_tensors(rows,cache,device))
            for i,item in enumerate(target):
                t=torch.as_tensor(item['target'][:-1],device=device)
                if ((out['coverage'][i]<=0)&(t>0)).any():
                    raise ValueError('positive mask is outside grid coverage; reduce grid size')
            loss=dense_loss(out['probabilities'],target,args.gamma,args.null_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite dense loss; incomplete output preserved')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(rows)
        predictions,val_loss=_predict(model,validation,cache,data['_root'],device,args.gamma,args.null_weight)
        record=dict(epoch=epoch,train_loss=total/len(train),val_loss=val_loss,
                    validation=_metrics(validation,predictions,data['_root'],cache.config['grid_size']))
        history.append(record);save_json(output/'history.json',history)
        snapshot=dict(provenance,epoch=epoch,model_state=model.state_dict())
        if val_loss<best:
            best=val_loss;torch.save(snapshot,output/'best.pt')
        if args.stage=='simulation' and epoch==3:
            torch.save(snapshot,output/'epoch_003.pt')
        torch.save(snapshot,output/'last.pt')
        print(f'Dense epoch {epoch}/{args.epochs}: train={record["train_loss"]:.6f}, val={val_loss:.6f}',flush=True)
    run.update(status='complete',best_sha256=sha256(output/'best.pt'),last_sha256=sha256(output/'last.pt'))
    save_json(output/'training_run.json',run)
    return run


def evaluate_dense(args):
    data,cache=_load(args)
    rows=[r for r in data['samples'] if r['split']==args.split]
    torch.set_num_threads(args.threads);device=resolve_device(args.device,torch)
    model,checkpoint=load_head(args.checkpoint,cache,device)
    hashes={r[k] for r in rows for k in ('reference_sha256','image_sha256')}
    if (hashes&set(checkpoint['training_images']) or
            {r['group'] for r in rows}&set(checkpoint['training_groups'])):
        raise ValueError('dense evaluation overlaps checkpoint training')
    output=Path(args.output)
    if output.exists():
        raise ValueError('evaluation output exists; preserve it and choose a new directory')
    probabilities,loss=_predict(model,rows,cache,data['_root'],device)
    result=dict(status='running',checkpoint_sha256=sha256(args.checkpoint),
        manifest_sha256=data['_manifest_sha256'],split=args.split,loss=loss,
        **_metrics(rows,probabilities,data['_root'],cache.config['grid_size']))
    output.mkdir(parents=True);(output/'assets').mkdir()
    cards=[];grid=cache.config['grid_size']
    with (output/'scores.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['id','presence','null_uncalibrated','top1_x','top1_y'])
        for i,(row,p) in enumerate(zip(rows,probabilities,strict=True)):
            with Image.open(checked_path(data['_root'],row['image'])) as im:
                source=im.convert('RGB');source.thumbnail((1400,1000))
            with Image.open(checked_path(data['_root'],row['reference'])) as im:
                ref=im.convert('RGB');ref.thumbnail((300,300));ref.save(output/'assets'/f'{i}_reference.png')
            heat=p[:-1].reshape(grid,grid)
            alpha=Image.fromarray((heat/max(float(heat.max()),1e-12)*160).astype(np.uint8)).resize(
                source.size,Image.Resampling.BILINEAR)
            overlay=Image.composite(Image.new('RGB',source.size,(255,0,0)),source,alpha)
            overlay.save(output/'assets'/f'{i}_heatmap.png')
            top=int(np.argmax(p[:-1]));x,y=(top%grid+.5)/grid,(top//grid+.5)/grid
            writer.writerow([row['id'],row['presence'],float(p[-1]),x,y])
            cards.append(f'<article><h2>{html.escape(row["id"])}</h2><p>Null（未校准） {p[-1]:.4f}; '
                f'真值 {html.escape(row["presence"])}</p><img width="240" src="assets/{i}_reference.png">'
                f'<img width="900" src="assets/{i}_heatmap.png"></article>')
    np.savez_compressed(output/'probabilities.npz',probabilities=np.stack(probabilities),
                        ids=np.array([r['id'] for r in rows]))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>参考图引导空间搜索</title>'
        '<style>body{font:17px sans-serif;max-width:1200px;margin:30px auto}img{max-width:100%}article{border-top:1px solid #ccc}</style>'
        '<h1>参考图引导航拍搜索</h1><p>热图来自受空间监督的输出，按图归一化显示；不等于车辆部件或身份概率。</p>'
        +''.join(cards),encoding='utf-8')
    result['status']='complete';save_json(output/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result
