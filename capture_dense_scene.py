"""Capture full aerial RGB and verified visible instance masks."""
import argparse
import json
from pathlib import Path
import sys
from dinoexp.dense_capture import capture_dense_scene,validate_dense_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--views',nargs='+',help='select query names, accepting _raised variants')
    p.add_argument('--take-tick-ownership',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.views:
        if 'grid_capture' in cfg:
            p.error('--views cannot restrict a complete grid capture')
        cfg['views']=[next((v for v in cfg['views'] if v['name'].removesuffix('_raised')==name),None) for name in args.views]
        if any(v is None for v in cfg['views']):raise ValueError('requested dense query view missing')
    cfg = validate_dense_config(cfg)
    if cfg.get('hide_environment_objects'):raise ValueError('environment hiding unsupported for dense capture')
    if args.output.exists():raise ValueError('dense capture output already exists')
    if args.dry_run:
        print(json.dumps(dict(query_frames=len(cfg['views']),null_frames=int('grid_capture' not in cfg),references=1,
                              carla_connected=False,note='Offline config validation only; no files created.')));return
    import carla
    from capture import connect
    _,world,versions=connect(carla,cfg)
    capture_dense_scene(carla,world,versions,cfg,args.output,args.take_tick_ownership)


if __name__=='__main__':
    try:main()
    except (ImportError,OSError,ValueError,RuntimeError) as error:
        print('ERROR:',error,file=sys.stderr);sys.exit(2)
