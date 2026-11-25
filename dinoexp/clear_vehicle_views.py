"""Capture protocol and offline checks for the unobstructed two-vehicle Writer comparison."""

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from .core import camera_pose, camera_pose_matches, checked_path, crop_bounds
from .memory_replay import audit_plan
from .memory_retention import angular_gap, open_crop
from .region_ab_data import read_json, sha


SCHEMA = "clear_vehicle_orbits_v1"
TRAIN_CASES, VAL_CASES = 8, 3
MIN_PIXELS = 8192
NEAR, FAR = 25, 45


def orbit_views(case_index):
    """Both cars share every relative camera setting. Paired queries use a second pose."""
    rng = random.Random(20261124 + case_index)
    origin = rng.uniform(-180, 180)
    directions = [origin + offset for offset in (0, 90, 180, 270)]
    rng.shuffle(directions)
    views = []
    for direction in directions:
        for offset in (-7, -2, 2, 7):
            angle = direction + offset + rng.uniform(-1, 1)
            elevation, distance = rng.uniform(18, 32), rng.uniform(8, 10)
            query_offset = rng.choice((-1, 1)) * rng.uniform(4, 8)
            for role, az, el, radius in (
                ("write", angle, elevation, distance),
                ("query", angle + query_offset, elevation + 3, distance + .6),
            ):
                views.append(dict(step=len(views) // 2, role=role, azimuth=(az + 180) % 360 - 180,
                                  elevation=el, distance=radius))
    return views


def depth_mm(raw_rgb):
    # CARLA encodes metric depth with R as the least significant of three bytes.
    rgb = raw_rgb.astype(np.uint32)
    packed = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
    return np.minimum(packed * (1_000_000. / 0xffffff), 65535).astype(np.uint16)


def crop_and_roi(mask, projection):
    """Include the entire projected 3D box, even where a branch could hide the car."""
    height, width = mask.shape
    y, x = np.nonzero(mask)
    if len(x) < MIN_PIXELS or not np.isfinite(projection).all():
        raise ValueError("vehicle mask too small or invalid projected box")
    visible = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    crop = list(crop_bounds(visible, width, height))
    bounds = [math.floor(min(visible[0], projection[0])) - 2,
              math.floor(min(visible[1], projection[1])) - 2,
              math.ceil(max(visible[2], projection[2])) + 2,
              math.ceil(max(visible[3], projection[3])) + 2]
    if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > width or bounds[3] > height:
        raise ValueError("vehicle or projected box clipped by the frame")
    return crop, bounds


def require_clear(depth, roi, far_corner_m):
    x0, y0, x1, y1 = roi
    if depth.shape != (y1 - y0, x1 - x0) or not 0 < far_corner_m < 64:
        raise ValueError("invalid empty-scene depth or vehicle extent")
    if int(depth.min()) <= math.ceil((far_corner_m + 1.) * 1000):
        raise ValueError("obstructed view: background inside the vehicle depth clearance")


def verify_occupied_depth(target_mm, corner_distances_m):
    # Allow transparent windows and the difference between corner and surface range.
    near = (min(corner_distances_m) - 1.5) * 1000
    far = (max(corner_distances_m) + 1.) * 1000
    if not target_mm.size or np.mean((target_mm >= near) & (target_mm <= far)) < .8:
        raise ValueError("depth sensor failed the occupied-vehicle check")


def sample_key(identity, step, role):
    return f"{identity}/{step:02d}_{role}"


def make_route(case, samples):
    identities = [case["scene_id"] + ":" + car for car in ("A", "B")]
    angles = {uid: [] for uid in identities}
    steps = []
    for view in orbit_views(case["index"])[::2]:
        t = view["step"]
        writes = {}
        for uid in identities:
            k, q = (sample_key(uid, t, role) for role in ("write", "query"))
            old = angles[uid]
            w_angle, q_angle = (samples[key]["azimuth"] for key in (k, q))
            writes[uid] = dict(sample=k, independent=q,
                               previous_view_gap=min((angular_gap(w_angle, a) for a in old), default=None),
                               probe_previous_view_gap=min((angular_gap(q_angle, a) for a in old), default=None))
            old.append(w_angle)
        steps.append(dict(step=t, frame=f"{case['scene_id']}/{t:02d}", writes=writes))
    return dict(id=case["scene_id"], scene=case["scene_id"], split=case["split"], group=case["group"],
                identities=identities, samples=samples, steps=steps)


def verify_sample(root, case, uid, view, sample):
    stem = f"{case['path']}/{uid.split(':')[-1]}/{view['step']:02d}_{view['role']}"
    depth = sample["empty_depth"]
    expected_pose = camera_pose(sample["vehicle_center"], sample["vehicle_yaw"], view["azimuth"],
                                view["elevation"], view["distance"])
    if (sample["identity"] != uid or sample["image"] != stem + ".png"
            or sample["mask"]["path"] != stem + "_mask.png" or depth["path"] != stem + "_depth.png"
            or any(sample[k] != view[k] for k in view)
            or not camera_pose_matches(sample["pose"], expected_pose)
            or not camera_pose_matches(sample["pose"], depth["pose"], .01, .05)
            or depth["frame"] <= sample["sensor_frame"]):
        raise ValueError(f"inconsistent clean capture records for {stem}")
    foreground = open_crop(root, sample, foreground=True)
    if not foreground.getbbox():
        raise ValueError(f"empty foreground RGB: {stem}")
    with Image.open(checked_path(root, sample["mask"]["path"])) as im:
        mask = np.asarray(im) != 0
    crop, roi = crop_and_roi(mask, sample["projection"])
    path = checked_path(root, depth["path"])
    if crop != sample["bbox"] or roi != depth["roi"] or sha(path) != depth["sha256"]:
        raise ValueError(f"crop, clearance area or depth file changed: {stem}")
    with Image.open(path) as im:
        if im.mode not in ("I", "I;16"):
            raise ValueError(f"expected a lossless 16-bit depth image: {stem}")
        require_clear(np.asarray(im), roi, sample["far_corner_m"])
    return foreground


def audit_dataset(root):
    manifest_path = root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("experiment 8 requires a completed clear_vehicle_orbits_v1 capture")
    plan = read_json(root / "plan.json")
    if (plan["max_query_angle"], plan["novelty_angle"], plan["min_visible_pixels"], plan["excluded"]) != (
            NEAR, FAR, MIN_PIXELS, []):
        raise ValueError("clean replay plan changed its view thresholds or excluded routes")
    train, val, _, coverage = audit_plan(plan, sha(manifest_path))
    cases = manifest["cases"]
    if (len(train), len(val), len(cases)) != (TRAIN_CASES, VAL_CASES, TRAIN_CASES + VAL_CASES):
        raise ValueError("incomplete clean training or validation split")
    if len({c["anchor_spawn_index"] for c in cases}) != len(cases):
        raise ValueError("clean routes share a capture location")
    thumbnails, photos = [], set()
    for i, (case, route) in enumerate(zip(cases, plan["routes"], strict=True)):
        scene = f"clear_{i:02d}"
        if (case["index"] != i or case["scene_id"] != scene or case["path"] != f"cases/{scene}"
                or case["split"] != ("train" if i < TRAIN_CASES else "val")
                or route != make_route(case, route["samples"])
                or sha(root / case["path"] / "route.json") != case["route_sha256"]
                or read_json(root / case["path"] / "route.json") != route):
            raise ValueError(f"clean route or manifest mismatch: {scene}")
        expected = set()
        for uid in route["identities"]:
            for view in orbit_views(i):
                key = sample_key(uid, view["step"], view["role"])
                expected.add(key)
                sample = route["samples"][key]
                if sample["rgb_sha256"] in photos:
                    raise ValueError(f"RGB reused in the clean capture: {key}")
                photos.add(sample["rgb_sha256"])
                fg = verify_sample(root, case, uid, view, sample)
                if view["step"] % 4 == 0 and case["split"] == "val":
                    fg.thumbnail((150, 100))
                    cx, cy = sample["vehicle_center"][:2]
                    x, y = sample["pose"][:2]
                    azimuth = (math.degrees(math.atan2(y - cy, x - cx)) - sample["vehicle_yaw"] + 180) % 360 - 180
                    thumbnails.append((uid, view["role"], azimuth, fg))
        if expected != set(route["samples"]):
            raise ValueError(f"unexpected samples in {scene}")
    sheet = Image.new("RGB", (95 + 8 * 150, len(thumbnails) // 8 * 125))
    draw = ImageDraw.Draw(sheet)
    for i, (uid, role, azimuth, fg) in enumerate(thumbnails):
        left, top = 95 + i % 8 * 150, i // 8 * 125
        if i % 8 == 0:
            draw.text((4, top + 45), uid, fill="white")
        sheet.paste(fg, (left + (150 - fg.width) // 2, top + (100 - fg.height) // 2))
        draw.text((left + 12, top + 100), f"{role}  {azimuth:.1f} deg", fill="white")
    return dict(images=len(photos), coverage=coverage), sheet
