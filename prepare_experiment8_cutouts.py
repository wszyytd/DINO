"""Attach pixel masks to the exact 162 crops in the offline experiment 8 pilot."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image, ImageDraw

from dinoexp.memory_replay import audit_plan
from dinoexp.memory_retention import open_crop
from dinoexp.region_ab_data import read_json, sha, write_json
from prepare_experiment8_offline import SOURCE, prepare


MASK_PACK = Path(__file__).resolve().parent / "configs/experiment8_vehicle_masks.zip"
SCHEMA = "fixed_height_reviewed_cutouts_v1"


def attach_masks(root, plan, archive):
    samples = {key: sample for r in plan["routes"] for key, sample in r["samples"].items()}
    with ZipFile(archive) as pack:
        index = json.loads(pack.read("index.json"))
        if index["records"].keys() != samples.keys():
            raise ValueError("mask pack and sample IDs differ")
        for key, sample in samples.items():
            record = index["records"][key]
            if record["rgb_sha256"] != sample["rgb_sha256"] or sha(root / sample["image"]) != sample["rgb_sha256"]:
                raise ValueError(f"RGB differs from reviewed input: {key}")
            member = Path(sample["image"]).with_suffix(".mask.png").as_posix()
            if record["mask"] != member:
                raise ValueError(f"mask not paired to expected crop: {key}")
            blob = pack.read(member)
            if hashlib.sha256(blob).hexdigest() != record["mask_sha256"]:
                raise ValueError(f"mask checksum mismatch: {key}")
            with Image.open(BytesIO(blob)) as im:
                if im.mode != "L" or im.size != tuple(sample["size"]) or {p for _, p in im.getcolors()} != {0, 255}:
                    raise ValueError(f"expected nonempty native binary mask: {key}")
            (root / member).write_bytes(blob)
            sample["mask"] = dict(path=member, sha256=record["mask_sha256"])
    return {k: v for k, v in index.items() if k != "records"}


def preview(root, route):
    ordered = sorted(route["samples"].values(), key=lambda s: (s["camera_index"], s["identity"]))
    width, height = 252, 124
    sheet = Image.new("RGB", (4 * width, ((len(ordered)+3)//4)*height), "#dadde1")
    draw = ImageDraw.Draw(sheet)
    for i, s in enumerate(ordered):
        x, y = (i % 4)*width, (i//4)*height
        for j, foreground in enumerate((False, True)):
            crop = open_crop(root, s, foreground=foreground)
            crop.thumbnail((122, 100))
            sheet.paste(crop, (x + j*126 + (126-crop.width)//2, y+(102-crop.height)//2))
        draw.text((x+4, y+103), f"{s['camera_index']:02} {s['identity'][-1]} {s['role']}  RGB / foreground", fill="black")
    sheet.save(root / route["scene"] / "cutouts.jpg", quality=92)


def prepare_cutouts(source, output, archive=MASK_PACK):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"choose a fresh output directory: {output}")
    with TemporaryDirectory(dir=output.parent) as work:
        stage = Path(work) / "data"
        prepare(source, stage)
        plan = read_json(stage / "plan.json")
        manifest = read_json(stage / "dataset_manifest.json")
        base_plan_hash = sha(stage / "plan.json")
        meta = attach_masks(stage, plan, archive)
        for route in plan["routes"]:
            preview(stage, route)
        manifest.update(schema=SCHEMA, crop_mode="foreground", base_plan_sha256=base_plan_hash,
                        selection="visually screened fixed-height boxes; fitted binary vehicle masks",
                        segmentation=dict(pack_sha256=sha(archive), **meta))
        write_json(stage / "dataset_manifest.json", manifest)
        plan["dataset_sha256"] = sha(stage / "dataset_manifest.json")
        _, _, _, coverage = audit_plan(plan, plan["dataset_sha256"])
        if coverage != plan["coverage"]:
            raise ValueError("cutouts changed the original question coverage")
        write_json(stage / "plan.json", plan)
        stage.rename(output)
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_cutouts(args.source, args.output), indent=2))
