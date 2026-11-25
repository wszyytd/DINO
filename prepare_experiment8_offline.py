"""Prepare the visually reviewed vehicle-box pilot from the existing fixed-height photos."""

import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from dinoexp.core import checked_path
from dinoexp.memory_replay import audit_plan
from dinoexp.memory_retention import angular_gap
from dinoexp.region_ab_data import digest_json, read_json, sha, write_json


SOURCE = Path("data/fixed_height_similar_20261104")
# (write camera index, held-out query camera index). Each index is a 15-degree orbit step.
# Reserve refB entirely because that take repeats refA's camera locations.
VIEWS = (
    ((0, 1), (3, 4), (5, 6), (7, 8), (11, None), (16, None), (21, 22), (23, None)),
    ((0, 1), (2, None), (11, 12), (13, 14), (15, 16), (17, None), (20, 21), (22, 23)),
)
# Pin the exact reviewed crops, identities, frames and poses across the server copy.
REVIEWED = (
    ("6ccd64851d1f16884452259680009b5a825fdc5016e653475249504a601dcfc3",
     "5b77518c8e80f1780957eec956633df293141022adb59de38c9fdcc7e7939b4f"),
    ("89712ebdfcd42ac1fd6c38ac3ab42e94fe3183a55d25d7179d9de7f70a1569a8",
     "b689b49d753fdb53335e771194c8ec6e8cb2dea114cda32445f3f6cf666d92a1"),
    ("63eaf856ee58270fff94a850bf411697a884a6f52ac3649b344dc5b8b3245192",
     "c2c79cf5f56dabd33a1325c9b33ac10812a0530581c3f5abff90f69ff84b07a1"),
)


def read_case(source, stage, case, location, fingerprint):
    name = case["name"]
    indices = [i for pair in VIEWS[location] for i in pair if i is not None]
    if len(set(indices)) != len(indices):
        raise ValueError("offline pilot reused a camera position")
    group = case["config"]["anchor_spawn_index"]
    folder = checked_path(source, name)
    run = read_json(folder / "run.json")
    if (run["status"] != "complete" or run["config"] != case["config"]
            or group != (42, 64)[location]):
        raise ValueError(f"capture layout differs from the reviewed source: {name}")
    with (folder / "metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "query" and int(r["sequence"]) in indices]
    (stage / name).mkdir()
    samples, keys, proof = {}, {}, []
    roles = {w: "write" for w, _ in VIEWS[location]}
    for r in rows:
        i, car = int(r["sequence"]), r["identity"]
        uid = f"{name}:{car}"
        key = f"{uid}/{i:02d}"
        if car not in ("A", "B") or key in samples or r["status"] != "captured" or int(r["position"]) != group:
            raise ValueError(f"unexpected reviewed observation: {key}")
        original = checked_path(folder, r["crop"])
        relative = f"{name}/{car}_{i:02d}.png"
        target = stage / relative
        target.write_bytes(original.read_bytes())
        digest = sha(target)
        with Image.open(target) as im:
            size = list(im.size)
        x, y = json.loads(r["camera_pose"])[:2]
        tx, ty, _, _, yaw, _ = json.loads(r["target_pose"])
        azimuth = (math.degrees(math.atan2(y - ty, x - tx)) - yaw + 180) % 360 - 180
        samples[key] = dict(identity=uid, azimuth=azimuth, image=relative, rgb_sha256=digest,
                            bbox=[0, 0, *size], size=size, source_crop=r["crop"], frame=r["frame"],
                            camera_index=i, role=roles.get(i, "query"))
        keys[i, uid] = key
        proof.append((r["sample_id"], car, r["sequence"], r["position"], r["camera_pose"],
                      r["target_pose"], r["frame"], r["blueprint"], digest))
    if len(samples) != 2 * len(indices) or digest_json(sorted(proof)) != fingerprint:
        raise ValueError(f"reviewed images or camera metadata differ: {name}")

    ids = [f"{name}:{car}" for car in "AB"]
    seen = {uid: [] for uid in ids}
    steps = []
    for t, (w, q) in enumerate(VIEWS[location]):
        writes = {}
        for uid in ids:
            key = keys[w, uid]
            probe = keys[q, uid] if q is not None else None
            angle = samples[key]["azimuth"]
            previous = seen[uid]
            probe_gap = None
            if probe:
                qa = samples[probe]["azimuth"]
                if not 0 < angular_gap(qa, angle) <= 25:
                    raise ValueError(f"query outside the independent view window: {probe}")
                probe_gap = min((angular_gap(qa, a) for a in previous), default=None)
            writes[uid] = dict(sample=key, independent=probe, probe_previous_view_gap=probe_gap,
                               previous_view_gap=min((angular_gap(angle, a) for a in previous), default=None))
            previous.append(angle)
        steps.append(dict(step=t, frame=samples[keys[w, ids[0]]]["frame"], writes=writes))
    return dict(id=name, scene=name, group=f"spawn-{group}", split=("train", "val")[location],
                identities=ids, samples=samples, steps=steps)


def preview(stage, route):
    # Each row shows both vehicles and their held-out views for one write step.
    sheet = Image.new("RGB", (4 * 165, len(route["steps"]) * 115))
    draw = ImageDraw.Draw(sheet)
    for step in route["steps"]:
        for col, uid in enumerate(route["identities"]):
            record = step["writes"][uid]
            for offset, key in enumerate((record["sample"], record["independent"])):
                if key is None:
                    continue
                sample = route["samples"][key]
                x, y = 165 * (col * 2 + offset), 115 * step["step"]
                with Image.open(stage / sample["image"]) as im:
                    im.thumbnail((162, 92))
                    sheet.paste(im, (x, y))
                draw.text((x, y + 93), f"{uid[-1]} {sample['role']} {sample['azimuth']:.1f} deg", fill="white")
    sheet.save(stage / route["scene"] / "review.jpg")


def prepare(source, output):
    if output.exists():
        raise ValueError(f"choose a new output directory: {output}")
    collection = read_json(source / "collection.json")
    if collection["status"] != "complete":
        raise ValueError("incomplete fixed-height collection")
    cases = {c["name"]: c for c in collection["plan"]["cases"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "prepared"
        stage.mkdir()
        routes, used = [], []
        for pair, fingerprints in enumerate(REVIEWED):
            for loc, fingerprint in enumerate(fingerprints):
                name = f"pair{pair}_location{loc}_refA"
                route = read_case(source, stage, cases[name], loc, fingerprint)
                preview(stage, route)
                routes.append(route)
                used.append(dict(scene_id=name, split=route["split"], group=route["group"],
                                 pair_kind=cases[name]["pair_kind"], reviewed_sha256=fingerprint))
        manifest = stage / "dataset_manifest.json"
        write_json(manifest, dict(schema="fixed_height_reviewed_boxes_v1", status="complete",
                                  source=SOURCE.name, crop_mode="bbox", cases=used,
                                  selection="visually screened for external occlusion; no depth or instance masks"))
        digest = sha(manifest)
        plan = dict(kind="memory_retention_plan_v1", dataset_sha256=digest, routes=routes,
                    max_query_angle=25, novelty_angle=45, excluded=[])
        _, _, _, coverage = audit_plan(plan, digest)
        plan["coverage"] = coverage
        write_json(stage / "plan.json", plan)
        stage.rename(output)
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
