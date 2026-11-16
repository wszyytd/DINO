"""LiVOS on a longer two-vehicle outward walk and one observed return."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'preview', 'evaluate', 'report', 'run'], default='run')
    parser.add_argument('--data', default='data/search_grid_smoke_v1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--livos-repo', default=str(Path(__file__).resolve().parent.parent/'LiVOS'))
    parser.add_argument('--weights', help='explicit LiVOS checkpoint; no automatic download')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--min-view-pixels', type=int, default=8192)
    parser.add_argument('--max-routes', type=int, default=1, help='maximum primary routes per scene')
    parser.add_argument('--max-outward', type=int, default=17, help='search limit, not padded length')
    parser.add_argument('--beam-width', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if (args.size != -1 and args.size < 32) or args.threads < 1:
        parser.error('--size must be -1 or >= 32; --threads must be positive')
    if min(args.min_view_pixels, args.max_routes, args.beam_width) < 1 or not 3 <= args.max_outward <= 33:
        parser.error('positive route limits required; --max-outward must be between 3 and 33')
    if args.stage in ('evaluate', 'run') and not args.weights:
        parser.error('model inference requires explicit --weights')
    from dinoexp.livos_roundtrip_routes import load_roundtrip_plan, prepare, render_report
    if args.stage in ('audit', 'run'):
        prepare(args)
    if args.stage in ('preview', 'run'):
        render_report(args, preview=True)
    if args.stage in ('evaluate', 'run'):
        load_roundtrip_plan(args)
        from dinoexp.livos_vehicle_eval import evaluate
        evaluate(args)
        load_roundtrip_plan(args)
    if args.stage in ('report', 'run'):
        render_report(args)


if __name__ == '__main__':
    main()
