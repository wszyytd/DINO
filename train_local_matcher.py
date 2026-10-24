"""Train a reference-conditioned local matching head on cached frozen DINOv2 features."""

import argparse
from pathlib import Path

from dinoexp.local_training import train_head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-train-only', action='store_true', help='explicit pilot, no validation claim')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--probes', type=int, default=4)
    p.add_argument('--reference-grid', type=int, default=8)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--threads', type=int, default=4)
    train_head(p.parse_args())


if __name__ == '__main__':
    main()
