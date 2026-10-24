"""Build confirmed positive/negative crop pairs from an existing CARLA capture."""

import argparse
import json
from pathlib import Path

from dinoexp.local_pairs import build_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--dataset', required=True, help='capture name relative to --data-root')
    p.add_argument('--group', required=True, help='same location/vehicles/related repeats share ID')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = build_manifest(args.data_root, args.dataset, args.group)
    if args.output.exists():
        raise ValueError('output exists; choose a new manifest path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    positives = sum(r['label'] for r in manifest['pairs'])
    print(f'Saved {len(manifest["pairs"])} pairs: {positives} positive, '
          f'{len(manifest["pairs"]) - positives} negative. Group: {args.group}')
    print('No random train/validation split. This pilot is for training-flow inspection.')


if __name__ == '__main__':
    main()
