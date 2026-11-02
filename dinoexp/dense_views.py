"""Letterboxed multiview geometry in native-image coordinates (no torch)."""
from collections import Counter
import math
import numpy as np
from PIL import Image


def bilinear_map(xy, grid_size):
    if grid_size < 1 or not np.isfinite(xy).all():
        raise ValueError('finite positions and positive grid_size required')
    p = np.clip(np.asarray(xy, float) * grid_size - .5, 0, grid_size - 1)
    lo = np.floor(p).astype(np.int64); hi = np.minimum(lo + 1, grid_size - 1)
    dx, dy = (p - lo).T
    indices = np.stack((lo[:, 1]*grid_size+lo[:, 0], lo[:, 1]*grid_size+hi[:, 0],
                        hi[:, 1]*grid_size+lo[:, 0], hi[:, 1]*grid_size+hi[:, 0]), -1)
    weights = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
    return indices, weights.astype(np.float32)


def _prepare(image, rect, size):
    x, y, w, h = rect
    native = image.crop((x, y, x+w, y+h)).convert('RGB')
    scale = min(size/w, size/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    left, top = (size-rw)//2, (size-rh)//2
    prepared = Image.new('RGB', (size, size), (124, 116, 104))
    prepared.paste(native.resize((rw, rh), Image.Resampling.BICUBIC), (left, top))
    mask = np.zeros((size, size), np.uint8); mask[top:top+rh, left:left+rw] = 1
    side = size//14
    fraction = mask.reshape(side, 14, side, 14).mean((1, 3)).reshape(-1)
    yy, xx = np.meshgrid(np.arange(side)*14+7, np.arange(side)*14+7, indexing='ij')
    # Partial edge patches map to their actual content, never outside the source.
    sx = x + np.clip((xx.reshape(-1)-left)/rw, 0, 1)*w
    sy = y + np.clip((yy.reshape(-1)-top)/rh, 0, 1)*h
    return prepared, fraction, np.stack((sx/image.width, sy/image.height), -1), (rw/size, rh/size)


def make_views(image, image_size=518, tile_size=512, rows=3, cols=5, grid_size=32):
    if (image_size < 14 or image_size % 14 or min(tile_size, rows, cols, grid_size) < 1
            or image.width < 1 or image.height < 1):
        raise ValueError('positive geometry and input size divisible by14 required')
    width, height = image.size
    tw, th = min(tile_size, width), min(tile_size, height)
    rectangles = [(0, 0, width, height)] + [
        (round(x), round(y), tw, th) for y in np.linspace(0, height-th, rows)
        for x in np.linspace(0, width-tw, cols)]
    multiplicity = Counter(rectangles)
    images, valid, geometry, indices, weights, priors, view_weights = [], [], [], [], [], [], []
    for rect in rectangles:
        prepared, fraction, xy, resize = _prepare(image, rect, image_size)
        idx, weight = bilinear_map(xy, grid_size)
        keep = fraction > 0
        weight *= fraction[:, None]
        x, y, w, h = rect
        images.append(prepared); valid.append(keep)
        geometry.append([x/width, y/height, w/width, h/height, *resize])
        indices.append(idx); weights.append(weight)
        priors.append(min(4., math.sqrt(width*height/(w*h))))
        view_weights.append(1/multiplicity[rect])
    return dict(images=images, valid=np.array(valid), geometry=np.array(geometry, np.float32),
                indices=np.array(indices), weights=np.array(weights),
                priors=np.array(priors, np.float32), view_weights=np.array(view_weights, np.float32),
                grid_size=grid_size)


def prepare_reference(image, image_size):
    prepared, fraction, _, _ = _prepare(image, (0, 0, image.width, image.height), image_size)
    return prepared, fraction > 0


def camera_embedding(pose=None, fov=None):
    """Measured world Z and orientation/FOV; no fabricated height above ground."""
    if pose is None or fov is None:
        return np.zeros(8, np.float32), False
    if len(pose) != 6 or not np.isfinite([*pose, fov]).all() or not 0 < fov < 180:
        raise ValueError('camera_pose needs six finite values and valid fov')
    angles = [math.radians(a) for a in pose[3:]]
    return np.array([pose[2]/100, *[v for a in angles for v in (math.sin(a), math.cos(a))],
                     fov/180], np.float32), True
