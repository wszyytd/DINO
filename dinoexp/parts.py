"""Native-coordinate vehicle-part exports; no identity or exclusion inference."""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import checked_path
from .dataset import load_dataset


def file_token(text):
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("._") or "item"
    if value != text:
        value += "_" + hashlib.sha256(str(text).encode()).hexdigest()[:8]
    return value[:110]


def discover_scenes(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"data directory does not exist: {root}")
    if (root / "metadata.csv").is_file():
        paths = [root]
    elif (root / "suite.json").is_file():
        suite = json.loads((root / "suite.json").read_text(encoding="utf-8-sig"))
        conditions = suite.get("conditions", [])
        if suite.get("status") != "complete" or not conditions:
            raise ValueError("suite was incomplete; inspect suite.json")
        if any(c.get("status") != "complete" for c in conditions):
            raise ValueError("suite contains incomplete conditions")
        paths = [checked_path(root, c["name"]) for c in conditions]
    else:
        paths = sorted(p.parent for p in root.rglob("metadata.csv"))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(
            "need a scene metadata.csv or a suite of unique capture directories"
        )
    scenes = []
    for path in paths:
        ref, queries, rejected = load_dataset(path)
        name = path.name if path == root else path.relative_to(root).as_posix()
        scenes.append(
            dict(
                name=name,
                root=path,
                reference=ref,
                rows=[ref] + queries,
                rejected=rejected,
                metadata_sha256=hashlib.sha256(
                    (path / "metadata.csv").read_bytes()
                ).hexdigest(),
            )
        )
    return scenes


def export_prediction(image, detections, output, prefix):
    """detections: class_name, confidence, boolean mask in original crop HxW."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = file_token(prefix)
    image = image.convert("RGB")
    w, h = image.size
    source_name = prefix + "_source.png"
    image.save(output / source_name)
    overlay = np.asarray(image).copy()
    records = []
    for index, detection in enumerate(detections):
        confidence = float(detection["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("nonfinite or out-of-range model confidence")
        raw = np.asarray(detection["mask"])
        if raw.shape != (h, w) or not np.isfinite(raw).all():
            raise ValueError(
                f"mask must use native crop coordinates {(h, w)}; got {raw.shape}"
            )
        mask = raw > 0.5
        name = str(detection["class_name"])
        record = dict(
            class_name=name,
            confidence=confidence,
            status="detected",
            source=source_name,
            native_width=w,
            native_height=h,
            testability_validated=False,
            mask_pixels=int(mask.sum()),
            bbox_xyxy=None,
            bbox_normalized=None,
            mask=None,
            crop=None,
            masked_crop=None,
            crop_width=None,
            crop_height=None,
        )
        if not mask.any():
            record["status"] = "empty_mask"
            records.append(record)
            continue
        yy, xx = np.where(mask)
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        x0, y0, x1, y1 = box
        base = f"{prefix}_{index:02d}_{file_token(name)}"
        mask_name, crop_name, masked_name = (
            base + suffix for suffix in ["_mask.png", "_crop.png", "_masked.png"]
        )
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / mask_name)
        image.crop(box).save(output / crop_name)
        neutral = np.full_like(overlay, (124, 116, 104))
        neutral[mask] = np.asarray(image)[mask]
        Image.fromarray(neutral).crop(box).save(output / masked_name)
        record.update(
            bbox_xyxy=box,
            bbox_normalized=[x0 / w, y0 / h, x1 / w, y1 / h],
            mask=mask_name,
            crop=crop_name,
            masked_crop=masked_name,
            crop_width=x1 - x0,
            crop_height=y1 - y0,
        )
        color = np.array(list(hashlib.sha256(name.encode()).digest()[:3]))
        color = 70 + color % 170
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        records.append(record)
    if not records:
        records = [
            dict(
                class_name=None,
                confidence=None,
                status="no_detection",
                source=source_name,
                native_width=w,
                native_height=h,
                testability_validated=False,
                mask_pixels=0,
                bbox_xyxy=None,
                bbox_normalized=None,
                mask=None,
                crop=None,
                masked_crop=None,
                crop_width=None,
                crop_height=None,
            )
        ]
    # Overlay is enlarged ONLY for human viewing; measurements remain native.
    view = Image.fromarray(overlay)
    scale = min(8, max(1, 720 / w))
    view = view.resize((round(w * scale), round(h * scale)), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(view)
    for record in records:
        if record["bbox_xyxy"] is None:
            continue
        box = [round(v * scale) for v in record["bbox_xyxy"]]
        draw.rectangle(box, outline="yellow", width=2)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{record['class_name']} {record['confidence']:.2f}",
            fill="black",
            stroke_width=1,
            stroke_fill="white",
            font=ImageFont.load_default(),
        )
    overlay_name = prefix + "_overlay.png"
    view.save(output / overlay_name)
    for record in records:
        record["overlay"] = overlay_name
    return records


def make_roi_annotations(records, reference_id, sample_ids, part):
    """Legacy match.py ROI bridge, exact class only; not a visibility guarantee."""
    result = {}
    for sample_id in sample_ids:
        matches = [
            r
            for r in records
            if r["sample_id"] == sample_id
            and r["class_name"] == part
            and r["status"] == "detected"
        ]
        annotation = dict(
            part=part,
            visible=False,
            detection_source="vehicle_part_model",
            testability_validated=False,
            needs_review=True,
        )
        if len(matches) == 1:
            match = matches[0]
            if match.get("crop_width", 2) < 2 or match.get("crop_height", 2) < 2:
                annotation["reason"] = "roi_too_small_for_existing_matcher"
            else:
                annotation.update(
                    visible=True,
                    box=match["bbox_normalized"],
                    reason="single_exact_class_detection_not_validated_testability",
                )
        else:
            annotation["reason"] = (
                "ambiguous_multiple_detections"
                if len(matches) > 1
                else "no_exact_class_detection"
            )
        result[sample_id] = annotation
    if reference_id not in result:
        raise ValueError("reference missing from ROI sample list")
    return result
