"""Stage two: pretrained LiVOS on existing vehicle captures, no training."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'preview', 'evaluate', 'report', 'run'],
                        default='run')
    parser.add_argument('--data', default='data/search_grid_smoke_v1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--livos-repo', default=str(Path(__file__).resolve().parent.parent/'LiVOS'))
    parser.add_argument('--weights', help='explicit official flat LiVOS .pth; no automatic download')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--size', type=int, default=480, help='short image side, or -1 for original')
    parser.add_argument('--length', type=int, default=48)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--max-cases', type=int, default=12)
    parser.add_argument('--min-init-pixels', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if (args.size != -1 and args.size < 32) or args.threads < 1:
        parser.error('--size must be -1 or at least 32; --threads must be positive')
    if args.stage in ('evaluate', 'run') and not args.weights:
        parser.error('model inference requires explicit --weights')
    if args.stage in ('audit', 'run'):
        from dinoexp.livos_vehicle_data import plan
        plan(args)
    if args.stage == 'preview':
        from dinoexp.livos_vehicle_report import report
        report(args, preview=True)
    if args.stage in ('evaluate', 'run'):
        from dinoexp.livos_vehicle_eval import evaluate
        evaluate(args)
    if args.stage in ('report', 'run'):
        from dinoexp.livos_vehicle_report import report
        report(args)


if __name__ == '__main__':
    main()
