"""Pure-data planning and paired metrics for vehicle memory retention."""

import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .core import checked_path
from .region_ab_data import read_json, sha
from .region_geometry import clip_box


def angular_gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def choose_view_probe(anchor, pool, route_hashes, max_gap):
    candidates = []
    for key, sample in pool.items():
        if sample["point"] == anchor["point"] or sample["rgb_sha256"] in route_hashes:
            continue
        gap = angular_gap(sample["azimuth"], anchor["azimuth"])
        if gap <= max_gap:
            candidates.append((gap, key))
    return min(candidates)[1] if candidates else None


def _json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _missing_files(root, frames):
    missing = []
    for row in frames:
        paths = [row["image_path"]]
        masks = [obj["mask"] for obj in row["objects"].values()] + [row["ignore_mask"]]
        paths.extend(mask["path"] for mask in masks if mask)
        for relative in paths:
            path = checked_path(root, relative)
            if not path.is_file() or not path.stat().st_size:
                missing.append(relative)
    return missing


def _route_plan(root, directory, case, frames, min_pixels, max_gap):
    scene = read_json(directory / "scene.json")
    graph = read_json(directory / "graph.json")["states"]
    route, = read_json(directory / "roundtrip_routes.json")["routes"]
    if route["scene"] != case["scene_id"]:
        raise ValueError(f"route/scene mismatch: {case['scene_id']}")
    ids = sorted(scene["objects"])
    if len(ids) != 2:
        raise ValueError("diagnostic requires two vehicle identities")
    audit = {row["state_id"]: row["rgb"] for row in _json_lines(directory / "audit/observations.jsonl")}
    states = {row["state_id"]: row for row in frames}
    if len(states) != len(frames) or states.keys() != audit.keys() or states.keys() != graph.keys():
        raise ValueError(f"frame, audit and graph coverage differ: {case['scene_id']}")
    route_hashes = {audit[key]["file_sha256"] for key in route["states"]}
    samples = {}
    for key, row in states.items():
        if row["annotation_validity"] != "valid":
            continue
        rgb = audit[key]
        width, height = rgb["width"], rgb["height"]
        cx, cy = rgb["pose"][:2]
        for uid in ids:
            obj = row["objects"][uid]
            quality = obj["quality"]
            if not obj["mask"] or quality["visible_pixels"] < min_pixels:
                continue
            x0, y0, x1, y1 = quality["bbox_xyxy"]
            dx, dy = .04 * (x1 - x0), .04 * (y1 - y0)
            bbox = clip_box((x0 - dx, y0 - dy, x1 + dx, y1 + dy), (width, height))
            if bbox is None:
                raise ValueError(f"empty vehicle crop: {key}/{uid}")
            car = scene["objects"][uid]
            x, y = car["center"][:2]
            azimuth = (math.degrees(math.atan2(cy - y, cx - x)) - car["pose"][4] + 180) % 360 - 180
            samples[f"{key}/{uid}"] = dict(frame=key, identity=uid, point=graph[key]["point_id"],
                image=row["image_path"], rgb_sha256=rgb["file_sha256"], size=[width, height],
                bbox=bbox, azimuth=azimuth, boundary=quality["touches_boundary"],
                visible_pixels=quality["visible_pixels"], mask=obj["mask"])

    pools = {uid: {key: s for key, s in samples.items() if s["identity"] == uid and not s["boundary"]}
             for uid in ids}
    used = set()
    written = set()
    angles = {uid: [] for uid in ids}
    steps = []
    for step, frame in enumerate(route["states"]):
        observations = {}
        for uid in ids:
            key = f"{frame}/{uid}"
            if key not in samples:
                continue
            sample = samples[key]
            dedup_key = (uid, sample["rgb_sha256"], tuple(sample["bbox"]))
            if dedup_key in written:
                continue
            written.add(dedup_key)
            used.add(key)
            independent = choose_view_probe(sample, pools[uid], route_hashes, max_gap)
            if independent is not None:
                used.add(independent)
            gap = min((angular_gap(sample["azimuth"], a) for a in angles[uid]), default=None)
            query_gap = (min((angular_gap(samples[independent]["azimuth"], a)
                         for a in angles[uid]), default=None) if independent else None)
            angles[uid].append(sample["azimuth"])
            observations[uid] = dict(sample=key, independent=independent, previous_view_gap=gap,
                                     probe_previous_view_gap=query_gap)
        steps.append(dict(step=step, frame=frame, writes=observations))
    if any(not angles[uid] for uid in ids):
        raise ValueError(f"candidate has no valid writes: {route['id']}")
    selected = {key: samples[key] for key in sorted(used)}
    verified = set()
    for s in selected.values():
        if s["image"] not in verified:
            path = checked_path(root, s["image"])
            if sha(path) != s["rgb_sha256"]:
                raise ValueError(f"RGB hash mismatch: {s['image']}")
            verified.add(s["image"])
        mask = s["mask"]
        if sha(checked_path(root, mask["path"])) != mask["sha256"]:
            raise ValueError(f"mask hash mismatch: {mask['path']}")
    return dict(id=route["id"], scene=case["scene_id"], split=case["split"], group=case["group"],
                identities=ids, samples=selected, steps=steps)


def build_plan(root, split, allow_incomplete, min_pixels, max_gap, novelty_gap):
    root = Path(root).resolve()
    if min_pixels < 1 or not 0 < max_gap < novelty_gap <= 180:
        raise ValueError("require positive pixels and 0 < query angle <= 180 below novelty angle")
    manifest_path = root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["schema"] != "multiscene_roundtrip_collection_v1" or manifest["status"] != "complete":
        raise ValueError("expected an accepted multi-scene capture manifest")
    routes, excluded = [], []
    for case in manifest["cases"]:
        if split != "all" and case["split"] != split:
            continue
        directory = checked_path(root, case["path"])
        frames = _json_lines(directory / "frames.jsonl")
        if len(frames) != manifest["config"]["grid"]["size"] ** 2 * 8:
            raise ValueError(f"unexpected grid frame count: {case['scene_id']}")
        missing = _missing_files(root, frames)
        audit = directory / "audit/observations.jsonl"
        if not audit.is_file() or not audit.stat().st_size:
            missing.append(audit.relative_to(root).as_posix())
        if missing:
            if not allow_incomplete:
                raise ValueError(f"{case['scene_id']}: {len(missing)} missing/empty files. "
                                 "Restore them or explicitly use --allow-incomplete.")
            excluded.append(dict(scene=case["scene_id"], split=case["split"],
                                 bad_file_count=len(missing), examples=missing[:5]))
            continue
        routes.append(_route_plan(root, directory, case, frames, min_pixels, max_gap))
    if not routes:
        raise ValueError("no eligible routes remain")
    coverage = defaultdict(lambda: defaultdict(int))
    for route in routes:
        row = coverage[route["split"]]
        row["routes"] += 1
        row["steps"] += len(route["steps"])
        for step in route["steps"]:
            for w in step["writes"].values():
                row["writes"] += 1
                row["independent_anchors"] += w["independent"] is not None
                row["independent_after_first"] += w["independent"] is not None and w["previous_view_gap"] is not None
                row["novel_anchors"] += (w["independent"] is not None
                    and (w["previous_view_gap"] or 0) >= novelty_gap
                    and (w["probe_previous_view_gap"] or 0) >= novelty_gap)
    if not any(row["independent_after_first"] for row in coverage.values()):
        raise ValueError("no independent view probes after the first write; check data coverage")
    return dict(kind="memory_retention_plan_v1", dataset_sha256=sha(manifest_path), split=split,
                min_visible_pixels=min_pixels, max_query_angle=max_gap, novelty_angle=novelty_gap,
                routes=routes, excluded=excluded, coverage=coverage)


def open_crop(root, sample, *, foreground=False):
    path = checked_path(root, sample["image"])
    if sha(path) != sample["rgb_sha256"]:
        raise ValueError(f"RGB changed after plan: {sample['image']}")
    with Image.open(path) as image:
        if list(image.size) != sample["size"]:
            raise ValueError(f"RGB size mismatch: {sample['image']}")
        crop = image.convert("RGB").crop(sample["bbox"])
    if not foreground:
        return crop
    record = sample["mask"]
    mask_path = checked_path(root, record["path"])
    if sha(mask_path) != record["sha256"]:
        raise ValueError(f"instance mask changed after plan: {record['path']}")
    with Image.open(mask_path) as mask:
        if list(mask.size) != sample["size"] or mask.mode not in ("1", "L"):
            raise ValueError(f"invalid instance mask dimensions or mode: {record['path']}")
        keep = mask.crop(sample["bbox"]).convert("L").point(lambda p: 255 if p else 0)
    if not keep.getbbox():
        raise ValueError(f"empty instance mask inside crop: {record['path']}")
    return Image.composite(crop, Image.new("RGB", crop.size), keep)


def score_pair(logits, seen, target):
    other = 1 - target
    own = logits[target] if seen[target] else None
    rival = logits[other] if seen[other] else None
    null = logits[2]
    indices = [i for i in range(2) if seen[i]] + [2]
    best = max(indices, key=logits.__getitem__)
    decision = "target" if best == target else "other" if best == other else "null"
    return dict(target=own, other=rival, null=null, decision=decision,
                margin=own - rival if own is not None and rival is not None else None)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["kind"]].append(row)
        if row.get("novel"):
            groups[row["split"], "novel_view"].append(row)
        if row["kind"] == "new_view" and "prior_writes" in row:
            count = row["prior_writes"] + 1
            bucket = "1" if count == 1 else "2-3" if count < 4 else "4+"
            groups[row["split"], f"new_view_after_{bucket}_writes"].append(row)
    result = defaultdict(dict)
    for (split, kind), group in groups.items():
        paired = [r for r in group if r.get("intervened", True)
                  and r["before"]["margin"] is not None and r["after"]["margin"] is not None]
        delta = [r["after"]["margin"] - r["before"]["margin"] for r in paired]
        origins = [r for r in group if r.get("origin", {}).get("decision") == "target"]
        result[split][kind] = dict(
            total=len(group), paired=len(paired),
            unique_probes=len({(r["route"], r["identity"], r["probe"]) for r in group}),
            correct_after=sum(r["after"]["decision"] == "target" for r in group),
            help=sum(r["before"]["decision"] != "target" and r["after"]["decision"] == "target" for r in paired),
            harm=sum(r["before"]["decision"] == "target" and r["after"]["decision"] != "target" for r in paired),
            changed=sum(r["before"]["decision"] != r["after"]["decision"] for r in paired),
            mean_margin_delta=sum(delta) / len(delta) if delta else None,
            mean_abs_margin_delta=sum(map(abs, delta)) / len(delta) if delta else None,
            initially_correct=len(origins),
            later_wrong=sum(r["after"]["decision"] != "target" for r in origins))
    return result
