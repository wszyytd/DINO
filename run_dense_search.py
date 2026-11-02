"""Verified masks -> frozen multiview features -> spatial training -> held-out report."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from dinoexp.dense_data import read_dense_manifest
from dinoexp.dense_cache import DenseFeatureCache
from dinoexp.local_paths import resolve_backbone_paths


def make_plan(args):
    data=read_dense_manifest(args.manifest,args.data_root)
    if {r['split'] for r in data['samples']}!={'train','val','test'}:
        raise ValueError('one-shot dense run needs separate train/val/test; never split correlated views randomly')
    mode='simulation' if args.stage=='simulation' else 'confirmed_regions'
    if any(r['supervision']!=mode for r in data['samples'] if r['split'] in ('train','val')):
        raise ValueError('manifest supervision differs from selected training stage')
    if args.stage=='real':
        if not args.checkpoint or not args.checkpoint.is_file():
            raise ValueError('real adaptation requires an existing simulation dense checkpoint')
        import torch
        source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if source.get('kind')!='dense_reference_search_v1' or source.get('stage') not in ('simulation','real'):
            raise ValueError('real adaptation needs a simulation-trained dense checkpoint, not crop weights')
    elif args.checkpoint:
        raise ValueError('simulation starts from random task modules; checkpoint is for real adaptation')
    if args.output.exists():
        raise ValueError('output exists; preserve it and choose a new run directory')
    model,size,dim,heads=(('dinov2_vitb14',518,256,8) if args.preset=='teacher'
                          else ('dinov2_vits14',224,64,4))
    scripts=Path(__file__).resolve().parent
    def command(script,*values):
        return [sys.executable,'-u',str(scripts/script),*map(str,values)]
    base=['--manifest',args.manifest,'--data-root',data['_root'],'--device',args.device]
    cache=args.cache or args.output/'features'
    commands=[]
    if args.cache:
        bank=DenseFeatureCache(cache)
        if bank.manifest_sha256!=data['_manifest_sha256'] or bank.config['model']!=model or bank.config['image_size']!=size:
            raise ValueError('reused cache differs from dense manifest or selected preset')
    else:
        repo,weight=resolve_backbone_paths(args.model_repo,args.backbone,model,scripts)
        commands.append(command('cache_dense_features.py',*base,'--output',cache,'--model',model,
            '--image-size',size,'--model-repo',repo,'--checkpoint',weight,'--dtype',args.dtype))
    epochs=args.epochs if args.epochs is not None else (5 if args.stage=='simulation' else 60)
    train=command('train_dense_search.py',*base,'--cache',cache,'--output',args.output/'train',
        '--stage',args.stage,'--epochs',epochs,'--dim',dim,'--heads',heads,'--batch-size',args.batch_size)
    if args.stage=='real':
        if not args.checkpoint or not args.checkpoint.is_file():
            raise ValueError('real adaptation requires an existing simulation dense checkpoint')
        train+=['--checkpoint',str(args.checkpoint)]
    commands.append(train)
    commands.append(command('evaluate_dense_search.py',*base,'--cache',cache,'--checkpoint',
        args.output/'train/best.pt','--output',args.output/'test','--split','test'))
    return commands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path)
    p.add_argument('--output',type=Path,default=Path('out')/('dense_search_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--cache',type=Path);p.add_argument('--model-repo',type=Path);p.add_argument('--backbone',type=Path)
    p.add_argument('--preset',choices=['teacher','small'],default='teacher')
    p.add_argument('--stage',choices=['simulation','real'],default='simulation')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--device',default='auto')
    p.add_argument('--dtype',choices=['float32','float16'],default='float32');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();commands=make_plan(args)
    print('Preset:',args.preset, '(small is an engineering smoke alternative)' if args.preset=='small' else '')
    for cmd in commands:
        print(subprocess.list2cmdline(list(map(str,cmd))),flush=True)
        if not args.dry_run:
            subprocess.run(list(map(str,cmd)),check=True)
    if not args.dry_run:
        print('Dense test report:',args.output/'test/index.html')


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,RuntimeError,subprocess.CalledProcessError) as error:
        print('ERROR:',error,file=sys.stderr);sys.exit(2)
