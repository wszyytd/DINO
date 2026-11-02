"""Strict visible-mask contracts for dense search; bounding boxes are not masks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import checked_path
from .local_pairs import sha256
from .dense_views import camera_embedding


def _mask(root, relative, size):
    with Image.open(checked_path(root, relative)) as im:
        if im.size != size or im.mode not in ('L', '1'):
            raise ValueError('mask must be native-sized single-channel binary image')
        a = np.asarray(im)
    if not set(np.unique(a)).issubset({0, 1, 255, False, True}):
        raise ValueError('mask must be binary')
    return a > 0


def mask_arrays(sample, root):
    with Image.open(checked_path(root, sample['image'])) as im:
        size = im.size
    positives = [_mask(root, p, size) for p in sample['positive_masks']]
    negative = np.zeros(size[::-1], bool)
    for path in sample.get('negative_masks', []):
        negative |= _mask(root, path, size)
    unknown = (_mask(root, sample['unknown_mask'], size) if sample.get('unknown_mask')
               else np.zeros_like(negative))
    union = np.zeros_like(negative)
    for mask in positives:
        if not mask.any() or (union & mask).any():
            raise ValueError('positive instance masks must be nonempty and disjoint')
        union |= mask
    if (union & negative).any() or (unknown & (union | negative)).any():
        raise ValueError('positive, negative and unknown masks overlap')
    return positives, negative, unknown


def read_dense_manifest(path, data_root=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('format_version') != 1 or data.get('task') != 'reference_guided_dense_search':
        raise ValueError('need a dense visible-mask manifest, not a crop-pair manifest')
    samples = data.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('dense manifest has no samples')
    root = Path(data_root or path.parent).resolve()
    groups, images, identifiers = {}, {}, set()
    for sample in samples:
        for key in ('id', 'split', 'group', 'reference', 'image', 'reference_sha256',
                    'image_sha256', 'supervision', 'presence', 'mask_source'):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError('missing dense sample field: ' + key)
        if sample['id'] in identifiers or sample['split'] not in ('train', 'val', 'test'):
            raise ValueError('duplicate sample id or invalid split')
        identifiers.add(sample['id'])
        if sample['supervision'] not in ('simulation', 'confirmed_regions'):
            raise ValueError('unknown supervision mode')
        if sample['mask_source'] not in ('carla_instance_verified','reviewed_manual','reviewed_sam2'):
            raise ValueError('unsupported visible mask source; boxes are not segmentation labels')
        if sample['supervision'] == 'confirmed_regions' and sample.get('reviewed') is not True:
            raise ValueError('real positive/negative masks need explicit review')
        if not isinstance(sample.get('positive_masks'), list):
            raise ValueError('positive_masks list required; boxes cannot substitute for masks')
        if sample['presence'] not in ('present', 'absent', 'unknown'):
            raise ValueError('invalid presence')
        if sample['presence'] == 'absent' and sample['positive_masks']:
            raise ValueError('absent sample cannot have positive masks')
        if sample['presence'] != 'absent' and not sample['positive_masks']:
            raise ValueError('present/unknown sample needs confirmed positive masks')
        if sample['supervision'] == 'simulation' and sample['presence'] == 'unknown':
            raise ValueError('simulation requires known presence')
        if sample['supervision'] == 'confirmed_regions' and not sample['positive_masks']:
            raise ValueError('negative-only/all-unknown real samples cannot define localization CE')
        for role in ('reference', 'image'):
            expected = sample[role+'_sha256']
            if sha256(checked_path(root, sample[role])) != expected:
                raise ValueError('source image hash mismatch')
            if expected in images and images[expected] != sample['split']:
                raise ValueError('image overlap across splits, including references')
            images[expected] = sample['split']
        if sample['group'] in groups and groups[sample['group']] != sample['split']:
            raise ValueError('physical group overlap across splits')
        groups[sample['group']] = sample['split']
        paths = sample['positive_masks'] + sample.get('negative_masks', [])
        if sample.get('unknown_mask'):
            paths = paths + [sample['unknown_mask']]
        if len(paths) != len(set(paths)):
            raise ValueError('duplicate mask path')
        for relative in paths:
            if sha256(checked_path(root, relative)) != sample.get('mask_sha256', {}).get(relative):
                raise ValueError('mask hash mismatch')
        camera_embedding(sample.get('camera_pose'), sample.get('fov'))
        positives, negative, unknown = mask_arrays(sample, root)
        if sample['supervision']=='confirmed_regions' and not negative.any():
            raise ValueError('real localization needs confirmed negative regions; unknown is not negative')
        if sample['supervision']=='simulation' and unknown.any():
            raise ValueError('simulation full-grid supervision cannot include unknown pixels')
        if sample['supervision'] == 'simulation' and sample['presence'] == 'present' and not positives:
            raise ValueError('present simulation needs visible masks')
    return dict(data, _root=root, _manifest_sha256=sha256(path))


def make_targets(sample, root, grid_size):
    positives, negative, _ = mask_arrays(sample, root)
    def area(mask):
        return np.asarray(Image.fromarray(mask.astype(np.float32)).resize(
            (grid_size, grid_size), Image.Resampling.BOX)).reshape(-1).copy()
    target = np.zeros(grid_size**2+1, np.float32)
    if sample['presence'] == 'absent':
        target[-1] = 1
    elif sample['supervision'] == 'confirmed_regions':
        for mask in positives:
            weight = area(mask)
            target[:-1] += weight / max(float(weight.sum()), 1e-12) / len(positives)
    else:
        union = np.logical_or.reduce(positives)
        weight = area(union)
        target[:-1] = weight / max(float(weight.sum()), 1e-12)
    if sample['supervision'] == 'simulation':
        known = np.ones(grid_size**2, bool)
    else:
        union = np.logical_or.reduce(positives) | negative
        known = area(union) > 0
        if not (known & (target[:-1]==0)).any():
            raise ValueError('no separate confirmed negative grid cells; increase grid resolution or review labels')
    if not np.isclose(target.sum(), 1):
        raise ValueError('no usable localization target after grid conversion')
    return dict(mode=sample['supervision'], target=target, known=known)
