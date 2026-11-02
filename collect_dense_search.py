"""Resume a small visible-mask search dataset from completed expanded-scene configs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from dinoexp.dense_collection import collect_dense


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source',type=Path);group.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path,default=Path('data')/('dense_search_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--views',nargs='+',default=['overview','descend'])
    p.add_argument('--max-train-groups',type=int)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--take-tick-ownership',action='store_true')
    args=p.parse_args()
    result=collect_dense(args.source,args.resume or args.output,args.views,args.max_train_groups,
        resume=bool(args.resume),dry_run=args.dry_run,take_tick=args.take_tick_ownership)
    print(json.dumps(result,indent=2),flush=True)
    if not args.dry_run:print('Dense manifest:',(args.resume or args.output)/'dense_manifest.json')


if __name__=='__main__':
    try:main()
    except (ImportError,OSError,ValueError,RuntimeError) as error:
        print('ERROR:',error,file=sys.stderr);sys.exit(2)
