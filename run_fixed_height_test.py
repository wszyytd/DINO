"""Test fixed existing region-sequence checkpoints on a supplied-box orbit dataset."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('audit', 'preview', 'prepare', 'evaluate', 'report', 'run'), default='run')
    parser.add_argument('--data', default='data/fixed_height_similar_20261104')
    parser.add_argument('--checkpoint-run', default='out/region_sequence_mask_20261112_152736')
    parser.add_argument('--output', required=True)
    parser.add_argument('--teacher', default='out/teacher_grid_fit_e20_retry/last.pt')
    parser.add_argument('--model-repo')
    parser.add_argument('--backbone')
    parser.add_argument('--dense-cache')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--view-batch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.view_batch, args.batch_size) < 1:
        parser.error('threads, view-batch and batch-size must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.stage in ('audit', 'run'):
        from dinoexp.fixed_height_data import audit
        audit(args)
    if args.stage in ('preview', 'report'):
        from dinoexp.fixed_height_report import report
        report(args, preview=args.stage == 'preview')
    if args.stage in ('prepare', 'run'):
        from dinoexp.fixed_height_eval import prepare
        prepare(args)
    if args.stage in ('evaluate', 'run'):
        from dinoexp.fixed_height_eval import evaluate
        evaluate(args)
    if args.stage == 'run':
        from dinoexp.fixed_height_report import report
        report(args)


if __name__ == '__main__':
    main()
