"""Mask-weighted descriptors in patch and foreground-normalized coordinates."""

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import normalize

from .core import checked_path


def foreground_patch_weights(mask, size):
    """Measure how much foreground each DINO patch contributes to each 4×4 cell."""
    patch = 14
    if size < patch or size % patch:
        raise ValueError("image size must be divisible by the DINO patch width")
    width, height = mask.size
    # Match dense_views._prepare's resize and letterbox. BOX resampling counts mask area.
    scale = min(size / width, size / height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    mask = np.asarray(mask.resize((rw, rh), Image.Resampling.BOX))
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("foreground mask is empty after resizing")
    # Map the tight foreground rectangle, in this resized crop, to 16 cells.
    col = 4 * (x - x.min()) // (x.max() - x.min() + 1)
    row = 4 * (y - y.min()) // (y.max() - y.min() + 1)
    cell = row * 4 + col
    left, top = (size - rw) // 2, (size - rh) // 2
    side = size // patch
    patch_id = (y + top) // patch * side + (x + left) // patch
    count = side * side
    area = np.bincount(cell * count + patch_id, weights=mask[y, x].astype(float) / 255,
                       minlength=16 * count)
    return (area.reshape(16, count) / (patch * patch)).astype(np.float32)


def foreground_keys(tokens, weights):
    # Normalized weighted sums equal normalized means. Unoccupied cells stay zero.
    cells = weights @ tokens.float()
    key = cells.sum(0)
    if not torch.isfinite(key).all() or key.norm() == 0:
        raise ValueError("empty or non-finite foreground descriptor")
    return dict(foreground_mean=normalize(key, dim=0), foreground_grid=normalize(cells, dim=-1))


def keys_from_sample(runtime, root, sample, raw):
    # The paired open_crop call has validated this mask against the experiment plan.
    with Image.open(checked_path(root, sample["mask"]["path"])) as image:
        mask = image.crop(sample["bbox"]).convert("L").point(lambda p: 255 if p else 0)
    weights = runtime.tensor(foreground_patch_weights(mask, runtime.cfg["image_size"]))
    return foreground_keys(raw[-1], weights)


def spatial_cosines(probes, bank, fallback):
    """Average corresponding cell cosines on the pairwise intersection of occupied cells."""
    common = probes.any(dim=-1)[:, None] & bank.any(dim=-1)[None, :]
    count = common.sum(dim=-1)
    cos = torch.einsum("qgd,kgd->qkg", probes, bank)
    return torch.where(count > 0, (cos * common).sum(dim=-1) / count.clamp_min(1), fallback)
