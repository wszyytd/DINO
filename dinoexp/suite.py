import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from .core import checked_path
from .scene import validate_scene


def build_cases(base, spec):
    result, names = [], set()
    for case in spec["cases"]:
        name = case["name"]
        if not re.fullmatch("[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("case names must be unique safe identifiers")
        names.add(name)
        cfg = copy.deepcopy(base)
        validate_scene(cfg)
        cfg["candidates"].sort(key=lambda c: c["identity"] != cfg["target_identity"])
        channels = case["distractor_color"].split(",")
        if len(channels) != 3 or any(not c.isdigit() or not 0 <= int(c) <= 255 for c in channels):
            raise ValueError("distractor_color must be three RGB channels in [0,255]")
        cfg["candidates"][1]["color"] = case["distractor_color"]
        if not isinstance(case["swap"], bool):
            raise ValueError("swap must be a boolean")
        if case["swap"]:
            cfg["candidates"].reverse()
        validate_scene(cfg)
        result.append((name, cfg))
    if not result:
        raise ValueError("empty scene suite")
    return result


def reuse_reference(source, destination):
    """Keep one reference across conditions, with original metadata and local backups."""
    source, destination = Path(source), Path(destination)

    def read_rows(root):
        with (root / "metadata.csv").open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)

    _, source_rows = read_rows(source)
    fields, rows = read_rows(destination)
    ref = next(r for r in source_rows if r["role"] == "reference")
    dest_ref = next(r for r in rows if r["role"] == "reference")
    if ref["identity"] != dest_ref["identity"] or ref["blueprint"] != dest_ref["blueprint"]:
        raise ValueError("shared reference must represent the same target identity and blueprint")
    if source.resolve() != destination.resolve():
        for key in ("image", "crop"):
            old = checked_path(destination, dest_ref[key])
            backup = old.with_name("local_reference" + old.suffix)
            if backup.exists():
                raise ValueError("reference already shared; refusing to overwrite backup")
            shutil.copyfile(old, backup)
            target = checked_path(destination, ref[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked_path(source, ref[key]), target)
        rows = [dict(ref) if r["role"] == "reference" else r for r in rows]
        with (destination / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    run_path = destination / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["reference_source"] = dict(
        scene=source.name,
        crop_sha256=hashlib.sha256(checked_path(source, ref["crop"]).read_bytes()).hexdigest(),
        note="reference frame/pose belong to source scene; query frames belong to this scene",
    )
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
