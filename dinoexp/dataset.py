import csv
import json
from pathlib import Path

from PIL import Image

from .core import checked_path, image_quality


def load_dataset(root):
    root = Path(root)
    if (root / "run.json").exists():
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if run.get("status") != "complete" or run.get("cleanup_errors"):
            raise ValueError("capture was incomplete or cleanup failed; inspect run.json")
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "identity", "role", "status", "crop"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"metadata.csv needs fields {sorted(required)}")
        rows = list(reader)
    ids = [r["sample_id"] for r in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("sample_id must be nonempty and unique")
    if any(
        r["role"] not in ("reference", "query")
        or not r["identity"]
        or r["status"] not in ("captured", "rejected")
        for r in rows
    ):
        raise ValueError("invalid identity, role or status in metadata")
    refs = [r for r in rows if r["role"] == "reference"]
    if len(refs) != 1 or refs[0]["status"] != "captured":
        raise ValueError("need exactly one valid reference")
    queries = [r for r in rows if r["role"] == "query" and r["status"] == "captured"]
    if not queries:
        raise ValueError("no valid queries")
    for r in [refs[0]] + queries:
        if not r["crop"] or not checked_path(root, r["crop"]).is_file():
            raise FileNotFoundError(f"missing crop for {r['sample_id']}: {r['crop']}")
        with Image.open(checked_path(root, r["crop"])) as image:
            if image_quality(image)["near_black"]:
                raise ValueError(f"near_black crop for {r['sample_id']}; recapture before matching")
    return refs[0], queries, [r for r in rows if r["status"] == "rejected"]
