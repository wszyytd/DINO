import math
import re
import copy

from .core import camera_pose


def capture_view_pose(center, anchor_yaw, target_yaw, view, *, is_reference,
                      query_yaw_mode='target'):
    if query_yaw_mode not in ('anchor', 'target'):
        raise ValueError('query_yaw_mode must be anchor or target')
    yaw = target_yaw if is_reference or query_yaw_mode == 'target' else anchor_yaw
    return camera_pose(center, yaw, view['azimuth'], view['elevation'], view['distance'])


def with_resolution(config, width=None, height=None):
    """Change native sensor pixels, preserving aspect ratio and observation geometry."""
    result = copy.deepcopy(config)
    if width is None and height is None:
        return result
    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0):
        raise ValueError("provide both --width and --height as positive integers")
    if width * config['height'] != height * config['width']:
        raise ValueError("resolution control must preserve original aspect ratio")
    result.update(width=width, height=height)
    return result


def candidate_poses(anchor, separation, yaw_offsets=None):
    if separation < 7:
        raise ValueError("candidate separation must be at least 7 m")
    yaw = math.radians(anchor[4])
    yaw_offsets = [0, 0] if yaw_offsets is None else yaw_offsets
    if (len(yaw_offsets) != 2 or any(type(y) not in (int, float) or not math.isfinite(y)
                                   for y in yaw_offsets)):
        raise ValueError('need two finite vehicle yaw offsets')
    poses = []
    for sign, offset in zip((-1, 1), yaw_offsets, strict=True):
        pose = list(anchor)
        pose[0] += sign * separation / 2 * math.cos(yaw)
        pose[1] += sign * separation / 2 * math.sin(yaw)
        pose[2] += 0.3
        pose[4] += offset
        poses.append(pose)
    return poses


def validate_scene(config):
    if config.get('query_yaw_mode', 'target') not in ('anchor', 'target'):
        raise ValueError('query_yaw_mode must be anchor or target')
    candidates = config["candidates"]
    ids = [c["identity"] for c in candidates]
    if (
        len(ids) != 2
        or len(set(ids)) != 2
        or any(not re.fullmatch("[A-Za-z0-9_-]+", i) for i in ids)
    ):
        raise ValueError("preview needs exactly two distinct safe candidate IDs")
    if config["target_identity"] not in ids:
        raise ValueError("target_identity must name one of the candidates")
    if len({(c["blueprint"], c["color"]) for c in candidates}) == 1:
        raise ValueError(
            "identical configured appearance: this probe needs an observable difference"
        )
    return config
