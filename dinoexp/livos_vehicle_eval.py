"""Official LiVOS inference on planned vehicle routes. CUDA only; no training."""
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

from .core import checked_path
from .livos_vehicle_data import binary_mask, load_plan, plan_digest
from .livos_vehicle_metrics import build_summary, score_frame
from .region_ab_data import sha, write_json


def repo_sources(repo):
    files = sorted((repo/'livos').rglob('*.py'))
    if not files or not (repo/'livos/model/livos_wrapper.py').is_file():
        raise ValueError('--livos-repo must point to the official LiVOS checkout')
    return {p.relative_to(repo).as_posix(): sha(p) for p in files}


def load_runtime(repo, weights, device):
    """Load every checkpoint tensor strictly; suppress redundant ImageNet downloads only."""
    import torch
    repo = Path(repo).resolve()
    weights = Path(weights).resolve()
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('model inference is server CUDA only; audit/preview need no torch')
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError('supply a nonempty official LiVOS .pth via --weights')
    if 'livos' in sys.modules:
        raise ValueError('fresh process required to avoid loading a different livos package')
    sys.path.insert(0, str(repo))
    wrapper = importlib.import_module('livos.model.livos_wrapper')
    evaluation = importlib.import_module('livos.eval')
    if Path(wrapper.__file__).resolve() != repo/'livos/model/livos_wrapper.py':
        raise ValueError('imported LiVOS from the wrong checkout')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or
            not torch.is_tensor(v) for k, v in state.items()):
        raise ValueError('expected official flat tensor state_dict, no partial/nested checkpoint')
    # Official constructors fetch ImageNet weights before loading a complete LiVOS model.
    # Construct the same architecture without that redundant fetch. The strict load below
    # must cover every tensor, so no randomly initialized parameter can survive.
    resnet = wrapper.resnet
    original18, original50 = resnet.resnet18, resnet.resnet50
    with ExitStack() as stack:
        stack.enter_context(patch.object(resnet, 'resnet18',
                            lambda pretrained=True, extra_dim=0:
                            original18(pretrained=False, extra_dim=extra_dim)))
        stack.enter_context(patch.object(resnet, 'resnet50',
                            lambda pretrained=True, extra_dim=0:
                            original50(pretrained=False, extra_dim=extra_dim)))
        network = wrapper.LIVOS(model_type='base')
    network.load_state_dict(state, strict=True)
    network.requires_grad_(False).to(device).eval()
    del state
    return torch, network, evaluation.InferenceCore


def inference_inputs(root, frame, target, first, size, *, resize_rgb=True):
    """Later frames read RGB only; target masks are read here only on initialization."""
    with Image.open(checked_path(root, frame['image'])) as image:
        image = image.convert('RGB')
        width, height = image.size
        if size > 0:
            ratio = size/min(width, height)
            resized = (size, int(height*ratio)) if width <= height else (int(width*ratio), size)
            if resize_rgb:
                image = image.resize(resized, Image.Resampling.BILINEAR)
        else:
            resized = image.size
        rgb = np.asarray(image, dtype=np.float32).copy()/255.
    mask = None
    if first:
        gt = binary_mask(root, frame['objects'][target]['mask'], frame['size'])
        mask_image = Image.fromarray(gt.astype(np.uint8))
        mask = np.asarray(mask_image.resize(resized, Image.Resampling.NEAREST)).copy()
        if not mask.any():
            raise ValueError('initial target disappears after resizing')
    return rgb.transpose(2, 0, 1), mask


def state_metadata(core):
    fields = ['state_BNCC', 'key_sum_BCHW', 'sensory_BNCHW',
              'obj_mem_sum_BNQC', 'last_masks_BNHW']
    tensors = {name: getattr(core, name) for name in fields}
    return dict(shapes={name: list(t.shape) for name, t in tensors.items()},
                tensor_bytes={name: t.numel()*t.element_size() for name, t in tensors.items()},
                persistent_tensor_bytes=sum(t.numel()*t.element_size() for t in tensors.values()))


def evaluate(args):
    root, output, repo = (Path(p).resolve() for p in (args.data, args.output, args.livos_repo))
    plan = load_plan(output, root)
    if (output/'inference.json').exists() or (output/'predictions').exists():
        raise ValueError('inference output exists; choose a fresh output/run, no overwrite')
    if not args.weights:
        raise ValueError('--weights must explicitly name an official LiVOS checkpoint')
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise ValueError('checkpoint not found: '+str(weights))
    sources, weight_hash = repo_sources(repo), sha(weights)
    torch, network, core_class = load_runtime(repo, weights, args.device)
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    records = []
    prediction_root = output/'predictions'
    prediction_root.mkdir()
    write_json(output/'inference.started.json', dict(status='running',
               plan_sha256=plan_digest(plan), weights_sha256=weight_hash,
               livos_sources=sources, protocol=plan['protocol']))
    with torch.inference_mode():
        for case in plan['cases']:
            core = core_class(network)
            directory = prediction_root/case['id']
            directory.mkdir()
            for step, key in enumerate(case['frames']):
                frame = plan['frames'][key]
                rgb, initial = inference_inputs(root, frame, case['target'], step == 0,
                                                plan['config']['size'], resize_rgb=False)
                image_tensor = torch.from_numpy(rgb)
                if plan['config']['size'] > 0:
                    image_tensor = resize(image_tensor, plan['config']['size'],
                                          InterpolationMode.BILINEAR, antialias=True)
                image_tensor = image_tensor.to(args.device)
                mask_tensor = None if initial is None else torch.from_numpy(initial).to(args.device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                prob = core.step(image_tensor, mask_HW=mask_tensor,
                                 objects=[1] if step == 0 else None,
                                 is_last_frame=step == len(case['frames'])-1)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                if prob.ndim != 3 or prob.shape[0] != 2 or not torch.isfinite(prob).all():
                    raise ValueError('invalid probability map from official single-target inference')
                # Resize class probabilities before argmax, like the official ResultSaver.
                prob = torch.nn.functional.interpolate(prob.unsqueeze(0),
                        size=(frame['size'][1], frame['size'][0]), mode='bilinear',
                        align_corners=False)[0]
                prediction = (prob.argmax(0) == 1).cpu().numpy()
                pred_path = directory/f'{step:03d}.png'
                Image.fromarray(prediction.astype(np.uint8)*255).save(pred_path)
                # Evaluation truth is accessed only after the model produced its prediction.
                masks = {uid: binary_mask(root, obj['mask'], frame['size'])
                         for uid, obj in frame['objects'].items()}
                ignored = binary_mask(root, frame['ignore'], frame['size'])
                cfg = plan['config']
                metric = score_frame(prediction, masks, case['target'], ignored,
                                     **{k: cfg[k] for k in ('success_iou', 'wrong_iou',
                                                          'wrong_margin', 'absent_pixels')})
                records.append(dict(case=case['id'], step=step, frame=key, target=case['target'],
                                    initial=step == 0, prediction=pred_path.relative_to(output).as_posix(),
                                    prediction_sha256=sha(pred_path), inference_seconds=seconds,
                                    **metric, state=state_metadata(core)))
            print('LiVOS case complete:', case['id'], case['target'], flush=True)
            del core
    if sources != repo_sources(repo) or weight_hash != sha(weights):
        raise ValueError('LiVOS source or checkpoint changed during inference')
    load_plan(output, root)
    write_json(output/'records.json', records)
    write_json(output/'summary.json', build_summary(records))
    write_json(output/'inference.json', dict(status='complete', kind='livos_vehicle_inference_v1',
               plan_sha256=plan_digest(plan), weights_sha256=weight_hash,
               weights_name=weights.name, livos_sources=sources,
               torch_version=torch.__version__, cuda_version=torch.version.cuda,
               gpu=torch.cuda.get_device_name(), mixed_precision=False,
               weights_load='strict=True; every tensor required; no download during construction',
               model='official LIVOS(base) and official InferenceCore; single target, ID=1',
               torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               inference_seconds=sum(r['inference_seconds'] for r in records),
               timing='synchronized InferenceCore.step only; excludes load/resize/IO/scoring',
               artifacts={name: sha(output/name) for name in ('records.json', 'summary.json')}))
    print('Saved vehicle inference:', output, flush=True)
