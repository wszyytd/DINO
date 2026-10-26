"""Train/validate on a completed expanded capture, then evaluate the selected head on test."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from dinoexp.local_pairs import check_split, read_manifest
from run_local_probe import execute_plan


def find_collection(data_dir):
    for path in sorted(Path(data_dir).glob('more_photos_*'), reverse=True):
        marker = path / 'collection.json'
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding='utf-8'))['status'] == 'complete':
                    return path
            except (OSError, ValueError, KeyError):
                continue
    raise ValueError('no complete data/more_photos_* collection found; finish capture or use --data')


def make_plan(root, args):
    root = Path(root).resolve()
    if json.loads((root / 'collection.json').read_text(encoding='utf-8'))['status'] != 'complete':
        raise ValueError('collection is incomplete; inspect collection.json')
    if args.output.exists():
        raise ValueError('training output already exists; choose a new run directory')
    manifests = {s: root / f'pairs_{s}.json' for s in ('train', 'val', 'test')}
    rows = {s: read_manifest(p)['pairs'] for s, p in manifests.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        check_split(rows[left], rows[right])
    scripts = Path(__file__).resolve().parent
    def cmd(name, *values):
        return [sys.executable, '-u', str(scripts / name), *map(str, values)]
    cache, train = args.output / 'features', args.output / 'train'
    cache_cmd = cmd('cache_local_features.py', '--data-root', root, '--output', cache,
                    '--device', args.device, '--batch-size', args.batch_size)
    for path in manifests.values():
        cache_cmd += ['--manifest', str(path)]
    if args.model_repo:
        cache_cmd += ['--model-repo', str(args.model_repo)]
    if args.backbone:
        cache_cmd += ['--checkpoint', str(args.backbone)]
    train_cmd = cmd('train_local_matcher.py', '--train-manifest', manifests['train'],
        '--val-manifest', manifests['val'], '--cache', cache, '--output', train,
        '--device', args.device, '--epochs', args.epochs,
        '--batch-size', args.batch_size, '--threads', args.threads)
    evaluations = [cmd('evaluate_local_matcher.py', '--manifest', manifests[split],
        '--cache', cache, '--checkpoint', train / 'best.pt', '--data-root', root,
        '--output', args.output / split, '--device', args.device,
        '--batch-size', args.batch_size, '--threads', args.threads) for split in ('val', 'test')]
    return [cache_cmd, train_cmd, *evaluations]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, help='completed collection; default latest data/more_photos_*')
    p.add_argument('--output', type=Path, default=Path('out') / (
        'more_photos_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--model-repo', type=Path)
    p.add_argument('--backbone', type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.data if args.data else find_collection(Path(__file__).resolve().parent / 'data')
    print(f'Collection: {root.resolve()}', flush=True)
    plan = make_plan(root, args)
    if args.dry_run:
        for command in plan:
            print(subprocess.list2cmdline(command))
    else:
        execute_plan(plan)
        print(f'Held-out test report: {args.output / "test" / "index.html"}')


if __name__ == '__main__':
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
