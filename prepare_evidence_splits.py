"""Create nested, paired crop-training manifests from a completed collection."""
import argparse
from pathlib import Path

from dinoexp.evidence_roles import load_corpus, make_manifests
from dinoexp.evidence_training import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--value-positions', nargs='+', required=True)
    p.add_argument('--fractions', nargs='+', type=float, default=[.25, .5, 1.])
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    episodes, audit = load_corpus(args.data, args.cache)
    manifests = make_manifests(episodes, args.value_positions, args.fractions, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'audit.json', audit)
    for index, m in enumerate(manifests, 1):
        destination = args.output/f'size_{index}.json'
        save_json(destination, m)
        print(f'{destination}: fraction={m["fraction"]}, '
              f'roles={ {r:len(ids) for r,ids in m["roles"].items()} }, '
              f'reserved_test={len(m["reserved_test"])}')


if __name__ == '__main__':
    main()
