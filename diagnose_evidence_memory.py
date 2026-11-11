"""Inspect frozen evidence checkpoints without training; execute on the server."""
import argparse
from pathlib import Path

from dinoexp.evidence_diagnostics import run_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=['current', 'mean', 'memory'],
                        default=['current', 'mean', 'memory'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--partitions', nargs='+', choices=['train', 'probe'], default=['train', 'probe'])
    parser.add_argument('--max-routes', type=int, default=0,
                        help='limit reference-conditioned tasks per split; partial smoke, no full replay claim')
    parser.add_argument('--repeat-samples', type=int, default=2,
                        help='anchors per history type / reference / scene; zero disables repeats')
    parser.add_argument('--repeat-counts', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='extra reads after the first observation')
    parser.add_argument('--examples', type=int, default=24)
    run_diagnostics(parser.parse_args())


if __name__ == '__main__':
    main()
