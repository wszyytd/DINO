"""Mask-assisted causal region sequences; plan/preview/report never import torch."""
import argparse
import copy
import os
from pathlib import Path

from dinoexp.region_ab_data import read_json
from dinoexp.region_sequence_data import DEFAULT_CONFIG, plan, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'preview', 'prepare', 'train', 'report', 'run'])
    parser.add_argument('--data', default='data/search_grid_smoke_v1')
    parser.add_argument('--regions', help='Existing complete region_encoding_ab_v1 output; saved raw teacher heatmaps')
    parser.add_argument('--output', required=True, help='New sequence output, never an existing A/B directory')
    parser.add_argument('--config', help='Complete sequence JSON config, frozen in plan.json')
    parser.add_argument('--teacher', default='out/teacher_grid_fit_e20_retry/last.pt')
    parser.add_argument('--model-repo')
    parser.add_argument('--backbone')
    parser.add_argument('--dense-cache')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--view-batch', type=int, default=1)
    args = parser.parse_args()
    if min(args.threads, args.view_batch) < 1:
        parser.error('positive threads and view-batch required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.stage in ('plan', 'run'):
        if not args.regions:
            parser.error('--regions is required to reuse verified teacher heatmaps')
        config = read_json(args.config) if args.config else copy.deepcopy(DEFAULT_CONFIG)
        validate_config(config)
        plan(args, config)
    elif args.config and read_json(args.config) != read_json(Path(args.output)/'plan.json')['config']:
        parser.error('later stages must use the frozen plan config')
    if args.stage in ('prepare', 'run'):
        from dinoexp.region_sequence_prepare import prepare
        prepare(args)
    if args.stage in ('train', 'run'):
        from dinoexp.region_sequence_training import train
        train(args)
    if args.stage in ('preview', 'report', 'run'):
        from dinoexp.region_sequence_report import report
        report(args, preview=args.stage == 'preview')


if __name__ == '__main__':
    main()
