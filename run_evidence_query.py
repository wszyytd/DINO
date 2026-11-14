"""Independent query memory experiment. plan/report do not import torch."""
import argparse
import json
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    for command in ('plan','prepare','train','evaluate','report'):
        s=commands.add_parser(command)
        s.add_argument('--output',required=True,help='new experiment root, containing manifest.json')
        if command=='plan':
            s.add_argument('--source-run',required=True); s.add_argument('--data',required=True)
            s.add_argument('--config',default='configs/evidence_query_v1.json')
        if command in ('prepare','report'):
            s.add_argument('--data',help='relocated RGB root; default recorded manifest root')
        if command=='prepare':
            s.add_argument('--source-run',help='relocated original region-sequence output')
            s.add_argument('--teacher',required=True); s.add_argument('--model-repo'); s.add_argument('--backbone')
            s.add_argument('--dense-cache'); s.add_argument('--view-batch',type=int,default=4)
        if command in ('prepare','train','evaluate'):
            s.add_argument('--device',default='cuda:0'); s.add_argument('--threads',type=int,default=4)
        if command in ('train','evaluate'):
            s.add_argument('--budget',choices=('smoke','full'),default='smoke')
        if command=='train':
            s.add_argument('--variant',choices=('A_last','B_mean','C_memory_current','D_memory_history','E_memory_repeat','all'),default='all')
        if command=='evaluate':
            s.add_argument('--split',choices=('train','validation','val','test','probe'),required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='plan':
        from dinoexp.evidence_query_data import build_manifest,code_hashes
        from dinoexp.region_ab_data import read_json,write_json
        out=Path(args.output)
        if out.exists(): raise ValueError('output exists; choose a new experiment root')
        manifest=build_manifest(Path(args.source_run),Path(args.data),read_json(Path(args.config)))
        out.mkdir(parents=True)
        write_json(out/'manifest.json',manifest)
        write_json(out/'checks.json',dict(local_data='passed',coverage=manifest['coverage'],split_audit=manifest['split_audit'],
            cuda_tests='NOT RUN',training='NOT RUN',model_effectiveness='NOT ESTABLISHED',code_hashes=code_hashes()))
        result=dict(status='complete',coverage=manifest['coverage'],split_audit=manifest['split_audit'])
    elif args.command=='prepare':
        from dinoexp.evidence_query_prepare import prepare
        result=prepare(args)
    elif args.command=='train':
        from dinoexp.evidence_query_training import train
        result=train(args)
    elif args.command=='evaluate':
        from dinoexp.evidence_query_eval import evaluate
        result=evaluate(args)
    else:
        from dinoexp.evidence_query_report import report
        result=report(args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
