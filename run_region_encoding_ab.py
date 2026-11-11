"""New independent A/B single-frame experiment; never runs on CPU implicitly."""
import argparse
import copy
import os
from pathlib import Path

from dinoexp.region_geometry import DEFAULTS, validate_config
from dinoexp.region_ab_data import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'profile', 'report', 'run'])
    parser.add_argument('--data', default='data/search_grid_smoke_v1')
    parser.add_argument('--teacher', default='out/teacher_grid_fit_e20_retry/last.pt')
    parser.add_argument('--model-repo')
    parser.add_argument('--backbone')
    parser.add_argument('--dense-cache', help='Optional exact dense_reference_features cache, NEVER the GT memory cache')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='Full fixed JSON config, written into prepared provenance before inference')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--view-batch', type=int, default=1)
    parser.add_argument('--profile-pairs', type=int, default=10)
    args = parser.parse_args()
    if min(args.threads, args.view_batch, args.profile_pairs) < 1:
        parser.error('threads/view-batch/profile-pairs must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = read_json(args.config) if args.config else copy.deepcopy(DEFAULTS)
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    if args.stage in ('prepare', 'run'):
        from dinoexp.region_ab_prepare import prepare
        prepare(args, config)
    elif args.config and config != read_json(Path(args.output)/'prepared.json')['config']:
        parser.error('later stages must use the frozen prepared config')
    if args.stage in ('train', 'run'):
        from dinoexp.region_ab_training import train
        train(args)
    if args.stage in ('profile', 'run'):
        from dinoexp.region_ab_profile import profile
        profile(args)
    if args.stage in ('report', 'run'):
        from dinoexp.region_ab_report import report
        report(args)


if __name__ == '__main__':
    main()
