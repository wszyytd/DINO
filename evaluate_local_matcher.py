"""Compare learned local matching with frozen CLS cosine on exactly the same crop pairs."""

import argparse
from pathlib import Path

from dinoexp.local_training import evaluate_head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='trained matching head, not backbone')
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-training-data', action='store_true', help='explicit fitting diagnostics')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    evaluate_head(p.parse_args())


if __name__ == '__main__':
    main()
