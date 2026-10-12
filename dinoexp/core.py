"""Geometry and data checks independent of CARLA and PyTorch."""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def camera_pose(center, vehicle_yaw, azimuth, elevation, distance):
    if distance <= 0 or not 0 <= elevation <= 90:
        raise ValueError("distance must be positive; elevation must be in [0, 90]")
    az, el = math.radians(vehicle_yaw + azimuth), math.radians(elevation)
    horizontal = distance * math.cos(el)
    x = center[0] + horizontal * math.cos(az)
    y = center[1] + horizontal * math.sin(az)
    z = center[2] + distance * math.sin(el)
    yaw = math.degrees(math.atan2(center[1] - y, center[0] - x))
    return (x, y, z, -elevation, yaw, 0.0)


def pose_values(transform):
    p, r = transform.location, transform.rotation
    return (p.x, p.y, p.z, r.pitch, r.yaw, r.roll)


def pose_matches(actual, desired, position_tolerance=0.03, angle_tolerance=0.5):
    return math.dist(actual[:3], desired[:3]) <= position_tolerance and all(
        abs((a - b + 180) % 360 - 180) <= angle_tolerance
        for a, b in zip(actual[3:], desired[3:], strict=True)
    )


def select_frame(frames, minimum_frame, desired):
    for frame, pose, payload in frames:
        if frame >= minimum_frame and pose_matches(pose, desired):
            return payload
    return None


def project_box(vertices, inverse, width, height, fov):
    vertices = np.asarray(vertices, dtype=float)
    inverse = np.asarray(inverse, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must be a nonempty Nx3 array")
    if inverse.shape != (4, 4) or not 0 < fov < 180:
        raise ValueError("invalid camera matrix or FOV")
    ue = (inverse @ np.column_stack([vertices, np.ones(len(vertices))]).T).T
    if not np.isfinite(ue).all() or np.any(ue[:, 0] <= 0):
        raise ValueError("vehicle bounding box crosses or is behind the camera")
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    u = focal * ue[:, 1] / ue[:, 0] + width / 2
    v = -focal * ue[:, 2] / ue[:, 0] + height / 2
    return (float(u.min()), float(v.min()), float(u.max()), float(v.max()))


def crop_bounds(box, width, height, padding=0.04):
    x0, y0, x1, y1 = box
    if not all(math.isfinite(x) for x in box) or padding < 0:
        raise ValueError("invalid crop")
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("vehicle clipped or too small; change view/distance")
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    return (
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(width, math.ceil(x1 + dx)),
        min(height, math.ceil(y1 + dy)),
    )


def cosine_scores(reference, queries):
    ref = np.asarray(reference, dtype=float)
    q = np.asarray(queries, dtype=float)
    if ref.ndim != 1 or q.ndim != 2 or q.shape[1] != len(ref):
        raise ValueError("feature shape mismatch")
    norms = np.linalg.norm(q, axis=1)
    nr = np.linalg.norm(ref)
    if (
        not np.isfinite(ref).all()
        or not np.isfinite(q).all()
        or nr < 1e-12
        or np.any(norms < 1e-12)
    ):
        raise ValueError("nonfinite or zero feature vector")
    return np.clip(q @ ref / (norms * nr), -1, 1)


def prepare_image(image, size=224):
    if size <= 0 or size % 14:
        raise ValueError("image size must be positive and divisible by 14")
    return ImageOps.pad(
        image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC, color=(124, 116, 104)
    )


def local_box(annotation, reference, dimensions):
    if (
        annotation.get("visible") is not True
        or reference.get("visible") is not True
        or not reference.get("part")
        or annotation.get("part") != reference.get("part")
    ):
        return None
    values = annotation.get("box")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("ROI box must be normalized [left, top, right, bottom]")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI coordinates must be ordered and within [0, 1]")
    w, h = dimensions
    box = (math.floor(x0 * w), math.floor(y0 * h), math.ceil(x1 * w), math.ceil(y1 * h))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("ROI is too small")
    return box


def checked_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("dataset paths must be relative")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("path escapes dataset directory")
    return result
