import math
import re


def candidate_poses(anchor, separation):
    if separation < 7:
        raise ValueError("candidate separation must be at least 7 m")
    yaw = math.radians(anchor[4])
    poses = []
    for sign in (-1, 1):
        pose = list(anchor)
        pose[0] += sign * separation / 2 * math.cos(yaw)
        pose[1] += sign * separation / 2 * math.sin(yaw)
        pose[2] += 0.3
        poses.append(pose)
    return poses


def validate_scene(config):
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
