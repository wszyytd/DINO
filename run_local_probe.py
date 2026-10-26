"""Run the existing-image local-attention pilot; each failed stage stops the pipeline."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from dinoexp.local_paths import resolve_backbone_paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='existing scene or suite directory')
    p.add_argument('--model-repo', type=Path, help='existing source; otherwise discover local cache')
    p.add_argument('--backbone', type=Path, help='official checkpoint; otherwise discover local cache')
    p.add_argument('--output', type=Path, default=Path('out') / (
        'local_match_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--group', default='charger-location0-pilot',
                   help='related repeats/location swaps must keep the same group')
    p.add_argument('--model', choices=['dinov2_vits14', 'dinov2_vitb14'], default='dinov2_vits14')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def make_plan(args):
    if args.output.exists():
        raise ValueError('output directory already exists; choose a new run name')
    scripts = Path(__file__).resolve().parent
    root = args.data.resolve()
    repo, backbone = resolve_backbone_paths(args.model_repo, args.backbone, args.model, scripts)
    if not root.is_dir():
        raise ValueError(f'capture data directory does not exist: {root}')
    if args.size <= 0 or args.size % 14 or min(args.epochs, args.batch_size, args.threads) < 1:
        raise ValueError('size must be divisible by 14; epochs, batch size and threads must be positive')
    pairs = args.output / 'pairs.json'
    cache, train, report = (args.output / p for p in ('features', 'train', 'evaluation'))
    def command(script, *arguments):
        return [sys.executable, '-u', str(scripts / script), *map(str, arguments)]
    return [
        command('prepare_pairs.py', '--data-root', root.parent, '--dataset', root.name,
                '--group', args.group, '--output', pairs),
        command('cache_local_features.py', '--manifest', pairs, '--data-root', root.parent,
                '--output', cache, '--model-repo', repo, '--checkpoint', backbone,
                '--model', args.model, '--size', args.size, '--device', args.device,
                '--batch-size', args.batch_size),
        command('train_local_matcher.py', '--train-manifest', pairs, '--cache', cache,
                '--output', train, '--allow-train-only', '--device', args.device,
                '--epochs', args.epochs, '--batch-size', args.batch_size, '--threads', args.threads),
        command('evaluate_local_matcher.py', '--manifest', pairs, '--cache', cache,
                '--checkpoint', train / 'last.pt', '--data-root', root.parent,
                '--output', report, '--allow-training-data', '--device', args.device,
                '--batch-size', args.batch_size, '--threads', args.threads),
    ]


def execute_plan(plan, runner=subprocess.run):
    for command in plan:
        print(subprocess.list2cmdline(command), flush=True)
        runner(command, check=True)


def main():
    args = parse_args()
    plan = make_plan(args)
    print('TRAIN-ONLY PILOT: same data used to inspect fitting; no held-out performance claim.')
    if args.dry_run:
        for command in plan:
            print(subprocess.list2cmdline(command))
    else:
        execute_plan(plan)
        print(f'Report: {args.output / "evaluation" / "index.html"}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
