"""Cache frozen spatial DINOv2 features once, then train the matching head cheaply."""

import argparse
from pathlib import Path

from dinoexp.local_features import extract_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, action='append', required=True)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-repo', type=Path, required=True, help='local official DINOv2 source')
    p.add_argument('--checkpoint', type=Path, required=True, help='official backbone checkpoint')
    p.add_argument('--model', choices=['dinov2_vits14', 'dinov2_vitb14'], default='dinov2_vits14')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9, 12])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    extract_cache(args.manifest, args.data_root, args.output, args.model_repo,
                  args.checkpoint, args.model, args.size, args.layers,
                  args.batch_size, args.device)


if __name__ == '__main__':
    main()
