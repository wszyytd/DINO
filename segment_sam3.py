"""Text-prompted SAM 3 vehicle-part trial on existing CARLA crops, without recapture."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from dinoexp.parts import discover_scenes
from dinoexp.sam3_parts import Sam3PartAdapter, load_prompts
from segment_parts import run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('out/sam3_parts01'))
    parser.add_argument('--weights', type=Path, help='authorized local official sam3.pt')
    parser.add_argument('--prompts', type=Path,
                        default=Path(__file__).resolve().parent / 'configs/sam3_parts.json')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--conf', type=float, default=.5)
    parser.add_argument('--cache-dir', type=Path, default=Path('models/sam3'))
    parser.add_argument('--revision', default='main')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        raise ValueError('confidence threshold must be in (0,1]')
    scenes = discover_scenes(args.data)
    prompts = load_prompts(args.prompts)
    if args.dry_run:
        print(json.dumps(dict(scenes=[s['name'] for s in scenes],
                              crops=sum(len(s['rows']) for s in scenes), prompts=prompts,
                              device=args.device, conf=args.conf,
                              warning='proposals only; no validated correspondence or exclusion'),
                         ensure_ascii=True, indent=2))
        return
    if args.output.exists():
        raise ValueError('output already exists; choose a fresh directory')
    if args.output.resolve().is_relative_to(args.data.resolve()):
        raise ValueError('output must be outside capture input')
    if args.weights and not args.weights.is_file():
        raise ValueError(f'local weights not found: {args.weights}')
    # Resolve gated weights before model construction or output creation.
    if args.weights:
        weights = args.weights.resolve()
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        try:
            weights = Path(hf_hub_download('facebook/sam3', 'sam3.pt',
                           revision=args.revision, cache_dir=str(args.cache_dir)))
        except GatedRepoError as error:
            raise RuntimeError(
                'SAM 3 weights require access approval at https://huggingface.co/facebook/sam3. '
                'After approval run hf auth login and retry, or pass authorized local --weights. '
                'No output or inference result was created.'
            ) from error
    import torch
    import sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install a driver-compatible PyTorch or use --device cpu')
    if device == 'cpu':
        print('SAM 3 CPU trial can be slow; using up to 8 threads.', flush=True)
        torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f'Loading SAM 3 on {device}: {weights}', flush=True)
    model = build_sam3_image_model(checkpoint_path=str(weights), load_from_HF=False,
                                   device=device, enable_inst_interactivity=False)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.conf)
    adapter = Sam3PartAdapter(processor, prompts)
    args.parts = ['all']
    args.imgsz = 1008
    args.iou = None
    args.device = device
    args.model_repo = 'facebook/sam3'
    use_bfloat16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    context = torch.autocast('cuda', dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
    with context:
        run_inference(args, scenes, adapter, weights,
                      dict(torch=torch.__version__, sam3=getattr(sam3, '__version__', 'unknown'),
                           sam3_source=str(Path(sam3.__file__).resolve()),
                           precision='bfloat16_autocast' if use_bfloat16 else 'float32',
                           requested_revision=args.revision if not args.weights else None))


if __name__ == '__main__':
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
