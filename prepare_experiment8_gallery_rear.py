"""Generate review sheets and masks for the training-only rear query candidates."""

import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from dinoexp.core import checked_path
from dinoexp.memory_replay import audit_plan
from dinoexp.region_ab_data import read_json, sha, write_json
from prepare_experiment8_cutouts import attach_masks, preview
from prepare_experiment8_offline import SOURCE
from run_experiment8_long import read_reference


BASELINE = Path("out/experiment8_cutouts_20261127_084108_Cmx6Mi")
KIND = "experiment8_gallery_rear_review_v1"


def stage_candidates(source, stage, route, cameras):
    scene = route["scene"]
    folder = checked_path(source, scene)
    target = stage / scene
    target.mkdir()
    if route["split"] != "train" or route["group"] != "spawn-42":
        raise ValueError(f"rear candidates must come from training location 42: {scene}")
    with (folder / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = route["identities"]
    relevant = [r for r in rows if r["role"] == "query" and int(r["sequence"]) in (*cameras, 11, 16)]
    by_view = {(f"{scene}:{r['identity']}", int(r["sequence"])): r for r in relevant}
    keys = {(uid, camera) for uid in ids for camera in (*cameras, 11, 16)}
    if by_view.keys() != keys or len(by_view) != len(relevant):
        raise ValueError(f"incomplete or duplicate rear camera rows: {scene}")
    for key, sample in route["samples"].items():
        if sample["camera_index"] in (11, 16):
            row = by_view[sample["identity"], sample["camera_index"]]
            if (row["crop"] != sample["source_crop"] or row["frame"] != sample["frame"]
                    or sha(checked_path(folder, row["crop"])) != sample["rgb_sha256"]):
                raise ValueError(f"rear source take differs from the original training plan: {key}")

    samples = {}
    for uid in ids:
        for camera in cameras:
            row = by_view[uid, camera]
            if row["status"] != "captured" or int(row["position"]) != 42:
                raise ValueError(f"missing capture in training location 42: {uid}, camera {camera}")
            relative = f"{scene}/{uid[-1]}_{camera:02}.png"
            path = stage / relative
            path.write_bytes(checked_path(folder, row["crop"]).read_bytes())
            with Image.open(path) as image:
                size = list(image.size)
            x, y = json.loads(row["camera_pose"])[:2]
            tx, ty, _, _, yaw, _ = json.loads(row["target_pose"])
            azimuth = (math.degrees(math.atan2(y - ty, x - tx)) - yaw + 180) % 360 - 180
            samples[f"{uid}/{camera:02}"] = dict(identity=uid, azimuth=azimuth, image=relative, rgb_sha256=sha(path),
                bbox=[0, 0, *size], size=size, source_crop=row["crop"], frame=row["frame"], camera_index=camera, role="query")
    return {**route, "samples": samples}


def prepare_review(source, baseline, output, weights, cameras):
    from generate_experiment8_masks import generate

    config, plan = read_reference(baseline)
    train, _, _, _ = audit_plan(plan, config["dataset_sha256"])
    if output.exists():
        raise ValueError(f"choose a new review directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "review"
        stage.mkdir()
        routes = [stage_candidates(source, stage, r, cameras) for r in train]
        pack = Path(tmp) / "masks.zip"
        plan_path = stage / "plan.json"
        write_json(plan_path, {"routes": routes})
        generate(source, stage, pack, weights)
        segmentation = attach_masks(stage, {"routes": routes}, pack)
        manifest = read_json(baseline / "data/dataset_manifest.json")
        if segmentation["weights_sha256"] != manifest["segmentation"]["weights_sha256"]:
            raise ValueError("supplemental query masks use different weights")
        for route in routes:
            preview(stage, route)
        write_json(stage / "review.json", dict(kind=KIND, base_plan_sha256=config["plan_sha256"],
                    cameras=cameras, segmentation=segmentation,
                    samples={r["id"]: r["samples"] for r in routes}))
        plan_path.unlink()
        stage.rename(output)
    return output


def load_review(root, plan_sha256, train, val, *, include_tail10=False):
    review = read_json(root / "review.json")
    if (review["kind"] != KIND or review["base_plan_sha256"] != plan_sha256
            or set(review["samples"]) != {r["id"] for r in train}):
        raise ValueError("rear review does not match the training routes and plan")
    held_out = {sample["rgb_sha256"] for r in val for sample in r["samples"].values()}
    cameras = (17, 10) if include_tail10 else (17,)
    if not set(cameras).issubset(review["cameras"]):
        raise ValueError(f"rear review must include cameras {cameras}")
    routes, supplemental = [], {}
    for route in train:
        candidates = review["samples"][route["id"]]
        new = {}
        for uid in route["identities"]:
            for camera in cameras:
                key = f"{uid}/{camera:02}"
                sample = candidates.get(key)
                if (not sample or key in route["samples"] or sample["identity"] != uid
                        or sample["camera_index"] != camera or sample["role"] != "query"
                        or sample["rgb_sha256"] in held_out):
                    raise ValueError(f"invalid or missing reviewed training query: {key}")
                new[key] = sample
                supplemental[key] = root
        routes.append({**route, "samples": {**route["samples"], **new}})
    return routes, supplemental


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--skip-camera10", action="store_true", help="generate only camera 17")
    args = parser.parse_args()
    output = prepare_review(args.source, args.baseline, args.output, args.weights, (17,) if args.skip_camera10 else (17, 10))
    print(f"Review {output}/*/cutouts.jpg for occlusion and mask quality before training.", flush=True)
