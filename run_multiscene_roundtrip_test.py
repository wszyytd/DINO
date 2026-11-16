"""Use accepted multi-scene capture data with the unchanged LiVOS inference core."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('audit', 'preview', 'evaluate', 'report', 'run'), default='audit')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--livos-repo', default='../LiVOS')
    parser.add_argument('--weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.size <= 0 or args.threads <= 0:
        parser.error('size and threads must be positive')
    if args.stage in ('evaluate', 'run') and not args.weights:
        parser.error('explicit --weights is required')
    from dinoexp.multiscene_routes import load_plan, prepare, render_report
    if args.stage in ('audit', 'run'):
        prepare(args)
    if args.stage in ('preview', 'run'):
        render_report(args, preview=True)
    if args.stage in ('evaluate', 'run'):
        load_plan(args)
        from dinoexp.livos_vehicle_eval import evaluate
        evaluate(args)
    if args.stage in ('report', 'run'):
        load_plan(args)
        render_report(args)


if __name__ == '__main__':
    main()
