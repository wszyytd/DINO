"""Train current/mean/learned-memory oracle baselines on identical grid routes."""
import argparse
from pathlib import Path
from dinoexp.evidence_data import load_evidence_cache
from dinoexp.evidence_training import train_comparison


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--routes', type=int, default=48)
    p.add_argument('--length', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seeds', nargs='+', type=int, default=[7,17,27])
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    args=p.parse_args()
    metadata, scenes=load_evidence_cache(args.cache)
    train_comparison(scenes, metadata, args)
