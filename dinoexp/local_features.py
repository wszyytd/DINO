"""Frozen DINOv2 spatial feature cache with image/checkpoint provenance."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .core import checked_path, prepare_image
from .local_pairs import read_manifest, sha256, verify_images
from .local_paths import resolve_backbone_paths


def preprocess(image, size):
    rgb = np.asarray(prepare_image(image, size), dtype=np.float32) / 255
    rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array(
        [.229, .224, .225], np.float32)
    mask = ImageOps.pad(Image.new('L', image.size, 255), (size, size),
                        method=Image.Resampling.NEAREST, color=0)
    # A patch containing any native content remains available; full padding is excluded.
    side = size // 14
    valid = np.asarray(mask).reshape(side, 14, side, 14).max(axis=(1, 3)).reshape(-1) > 0
    return rgb.transpose(2, 0, 1).copy(), valid


def resolve_device(name, torch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if name == 'auto' else name
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu or fix the Torch installation')
    return torch.device(device)


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)
        meta = json.loads((self.root / 'cache.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'complete' or meta.get('format_version') != 1:
            raise ValueError('feature cache incomplete or unsupported')
        self.config, self.images = meta['config'], meta['images']

    def get(self, digest):
        if digest not in self.images or len(digest) != 64 or any(
                c not in '0123456789abcdef' for c in digest):
            raise ValueError('image not found in feature cache')
        target = checked_path(self.root, digest + '.npz')
        if sha256(target) != self.images[digest]['sha256']:
            raise ValueError('cached feature hash mismatch')
        with np.load(target, allow_pickle=False) as data:
            item = {k: data[k].copy() for k in ('tokens', 'valid', 'cls')}
        side = self.config['image_size'] // 14
        expected = (len(self.config['layers']), side * side, self.config['input_dim'])
        if item['tokens'].shape != expected or item['valid'].shape != (side * side,):
            raise ValueError('cached feature shape mismatch')
        if item['cls'].shape != (expected[-1],) or not item['valid'].any():
            raise ValueError('invalid CLS or padding mask')
        if not np.isfinite(item['tokens']).all() or not np.isfinite(item['cls']).all():
            raise ValueError('nonfinite cached features')
        return item


def extract_cache(manifests, data_root, output, model_repo, checkpoint,
                  model_name='dinov2_vits14', size=224, layers=(3, 6, 9, 12),
                  batch_size=4, device='auto'):
    import torch
    if size <= 0 or size % 14 or batch_size < 1:
        raise ValueError('image size must be divisible by 14; positive batch size required')
    if not layers or list(layers) != sorted(set(layers)) or min(layers) < 1 or max(layers) > 12:
        raise ValueError('layers must be unique increasing one-based indices in [1,12]')
    output = Path(output)
    if output.exists():
        raise ValueError('cache output already exists; choose a new directory')
    model_repo, checkpoint = resolve_backbone_paths(
        model_repo, checkpoint, model_name, Path(__file__).resolve().parent.parent)
    images = {}
    for manifest_path in manifests:
        manifest = read_manifest(manifest_path)
        verify_images(manifest, data_root)
        for row in manifest['pairs']:
            for role in ('reference', 'query'):
                images[row[role + '_sha256']] = row[role]
    device = resolve_device(str(device), torch)
    backbone = torch.hub.load(str(model_repo), model_name, source='local', pretrained=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(False).to(device).eval()
    config = {'model': model_name, 'image_size': size, 'layers': list(layers),
              'input_dim': int(backbone.embed_dim), 'backbone_sha256': sha256(checkpoint),
              'preprocess': 'letterbox-imagenet-v1', 'cls_layer': 12}
    metadata = {'status': 'running', 'format_version': 1, 'config': config,
                'torch_version': str(torch.__version__), 'device': str(device),
                'backbone_source': str(Path(model_repo).resolve()),
                'manifests_sha256': [sha256(p) for p in manifests], 'images': {}}
    output.mkdir(parents=True, exist_ok=False)
    marker = output / 'cache.json'
    marker.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    items = list(images.items())
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            inputs, masks = [], []
            for _, rel in chunk:
                with Image.open(checked_path(data_root, rel)) as image:
                    array, valid = preprocess(image.convert('RGB'), size)
                inputs.append(array)
                masks.append(valid)
            x = torch.from_numpy(np.stack(inputs)).to(device)
            requested = sorted(set(layers) | {12})
            result = backbone.get_intermediate_layers(
                x, n=[i - 1 for i in requested], return_class_token=True, norm=True)
            by_layer = dict(zip(requested, result, strict=True))
            tokens = torch.stack([by_layer[i][0] for i in layers], dim=1).float().cpu().numpy()
            cls = by_layer[12][1].float().cpu().numpy()
            for i, (digest, rel) in enumerate(chunk):
                # Store float32: changing dtype is a numerical experiment, not a cache shortcut.
                target = output / (digest + '.npz')
                np.savez_compressed(target, tokens=tokens[i], valid=masks[i], cls=cls[i])
                metadata['images'][digest] = {'source': rel, 'sha256': sha256(target)}
            print(f'Frozen backbone: {min(start + batch_size, len(items))}/{len(items)}', flush=True)
    metadata['status'] = 'complete'
    marker.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def cls_cosine(reference, query):
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(query))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError('invalid CLS norm')
    return float(np.clip(np.dot(reference, query) / denominator, -1, 1))
