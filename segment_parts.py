"""Segment vehicle parts in CARLA captures. Outputs are diagnostic, never exclusion decisions."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

from PIL import Image

from dinoexp.core import checked_path
from dinoexp.parts import (
    discover_scenes,
    export_prediction,
    file_token,
    file_sha256,
    make_roi_annotations,
)


DEFAULT_REPO = "mitbersh/car-parts-segmentation"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_classes(names, requested):
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    # Catch accidental COCO or generic whole-vehicle checkpoints early.
    if not all(
        any(family in n.lower() for n in names.values())
        for family in ["door", "window", "roof"]
    ):
        raise ValueError(
            "checkpoint lacks vehicle door/window/roof classes; use vehicle-part segmentation weights"
        )
    ids = [
        k
        for k, name in names.items()
        if "all" in requested or any(p in name.lower() for p in requested)
    ]
    if not ids:
        raise ValueError("requested part families are absent from checkpoint labels")
    return names, ids


def decode_result(result, names, native_size):
    w, h = native_size
    if tuple(result.orig_shape) != (h, w):
        raise ValueError("prediction source shape differs from native crop")
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None:
        raise ValueError(
            "model returned boxes without masks; a segmentation checkpoint is required"
        )
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    if len(masks) != len(classes) or len(classes) != len(confidences):
        raise ValueError("model returned inconsistent masks/boxes")
    if masks.shape[1:] != (h, w):
        raise ValueError(
            "retina_masks did not return native masks; refusing to stretch letterbox masks"
        )
    return [
        dict(class_name=names[int(c)], confidence=float(conf), mask=mask)
        for c, conf, mask in zip(classes, confidences, masks, strict=True)
    ]


def preview(output, samples, run):
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for sample in samples:
        rows = []
        for record in sample["records"]:
            if record["status"] != "detected":
                rows.append(f'<tr><td colspan="4">{esc(record["status"])}</td></tr>')
                continue
            rows.append(
                f"<tr><td>{esc(record['class_name'])}</td>"
                f"<td>{record['confidence']:.3f}</td><td>{record['mask_pixels']}</td>"
                f"<td>{record['crop_width']}×{record['crop_height']}</td></tr>"
            )
        links = []
        for record in sample["records"]:
            if record["status"] == "detected":
                links.append(
                    f'<a href="{esc(record["masked_crop"])}">'
                    f"{esc(record['class_name'])} masked crop</a>"
                )
        record = sample["records"][0]
        cards.append(
            f"<article><h2>{esc(sample['scene'])} / {esc(sample['sample_id'])}</h2>"
            f"<p>{esc(sample['role'])} · candidate {esc(sample['identity'])} · "
            f"native {record['native_width']}×{record['native_height']}</p>"
            f'<a href="{esc(record["source"])}"><img src="{esc(record["overlay"])}"></a>'
            "<table><tr><th>Part</th><th>Model confidence</th><th>Native mask pixels</th>"
            "<th>Native ROI size</th></tr>"
            + "".join(rows)
            + "</table><p>"
            + " · ".join(links)
            + "</p></article>"
        )
    roi_links = "".join(
        f"<li>{esc(s['name'])}: "
        + " · ".join(
            f'<a href="{esc(p)}">{esc(Path(p).stem)}</a>'
            for p in s.get("roi_files", [])
        )
        + "</li>"
        for s in run["scenes"]
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CARLA vehicle parts</title>
<style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:30px auto;padding:0 18px;color:#243247}aside{background:#fff2d8;padding:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{border:1px solid #ddd;padding:18px;border-radius:8px;overflow-wrap:anywhere}h2{font-size:18px}img{width:100%;height:240px;object-fit:contain;image-rendering:pixelated}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px}a{color:#1459a4}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<h1>CARLA vehicle-part segmentation / 车辆部件分割</h1>
<aside>Model confidence is not testability or identity probability. No detection does not exclude a vehicle.
模型置信度不等于可检验性或身份概率；未检测到部件不能排除候选。叠加图放大仅用于展示，像素统计来自原始裁剪。
本入口使用CARLA已经采集的车辆包围盒裁剪，没有新增整车检测、连续跟踪或自动排除。</aside>
<p><a href="parts.csv">parts.csv</a> · <a href="parts.json">parts.json</a> · <a href="parts_run.json">运行记录</a></p>
<h2>Exact-class ROI exports for existing match.py</h2><p>检查掩膜后再使用。visible=true仅表示唯一同类预测，不保证部位完整可见或足够可比。现有match.py使用外接矩形，不使用分割掩膜。</p><ul>"""
    (output / "index.html").write_text(
        page
        + roi_links
        + '</ul><main class="grid">'
        + "".join(cards)
        + "</main></html>",
        encoding="utf-8",
    )


def run_inference(args, scenes, model, weights, versions):
    native_predict = getattr(model, "predict_parts", None)
    if callable(native_predict):
        names = model.names
        class_ids = list(names)
    else:
        names, class_ids = selected_classes(model.names, args.parts)
    if getattr(model, "task", None) != "segment":
        raise ValueError("checkpoint task must be segment")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        warning="No visibility/identity probability or exclusion decision",
        data=str(args.data.resolve()),
        weights=str(weights.resolve()),
        checkpoint_sha256=file_sha256(weights),
        model_repo=None if args.weights else args.model_repo,
        requested_revision=None if args.weights else args.revision,
        versions=versions,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=names,
        selected_classes={i: names[i] for i in class_ids},
        mask_coordinates="native vehicle crop; xyxy uses exclusive right/bottom",
        model_input="RGB pixels only; identity/role labels used for reporting only",
        scenes=[],
    )
    manifest["backend"] = getattr(model, "backend", "ultralytics")
    if callable(native_predict):
        manifest.update(
            prompts=model.prompts,
            class_semantics="text-prompt-conditioned proposals; side/front/rear not verified",
            nms="none across prompts; overlapping and duplicate masks retained for review",
            preprocess="official SAM3 processor: RGB resize to 1008x1008, normalization [-1,1]",
        )
    write_json(args.output / "parts_run.json", manifest)
    samples, flat = [], []
    try:
        for scene_index, scene in enumerate(scenes):
            key = f"{scene_index:03d}_{file_token(scene['name'])}"
            base = Path("scenes") / key
            destination = args.output / base
            destination.mkdir(parents=True)
            entry = dict(
                name=scene["name"],
                source=str(scene["root"]),
                metadata_sha256=scene["metadata_sha256"],
                rejected_captures=scene["rejected"],
                roi_files=[],
                processed=0,
            )
            manifest["scenes"].append(entry)
            scene_records = []
            for index, row in enumerate(scene["rows"]):
                source = checked_path(scene["root"], row["crop"])
                with Image.open(source) as original:
                    image = original.convert("RGB")
                print(
                    f"[parts] {scene['name']} / {row['sample_id']} {image.width}x{image.height}",
                    flush=True,
                )
                if callable(native_predict):
                    detections = native_predict(image)
                else:
                    predictions = model.predict(
                        source=image,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        device=args.device,
                        classes=class_ids,
                        retina_masks=True,
                        verbose=False,
                        save=False,
                    )
                    if len(predictions) != 1:
                        raise ValueError("expected exactly one prediction for one crop")
                    detections = decode_result(predictions[0], names, image.size)
                records = export_prediction(
                    image, detections, destination, f"s{index:04d}"
                )
                for record in records:
                    for field in ["source", "overlay", "mask", "crop", "masked_crop"]:
                        if record[field]:
                            record[field] = (base / record[field]).as_posix()
                    record.update(
                        scene=scene["name"],
                        sample_id=row["sample_id"],
                        identity=row["identity"],
                        role=row["role"],
                        view=row.get("view", ""),
                        frame=row.get("frame", ""),
                        source_crop=str(source),
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    )
                    if callable(native_predict):
                        record["text_prompt"] = model.prompt_texts.get(record["class_name"])
                sample = dict(
                    scene=scene["name"],
                    sample_id=row["sample_id"],
                    identity=row["identity"],
                    role=row["role"],
                    capture_metadata=row,
                    records=records,
                )
                samples.append(sample)
                flat.extend(records)
                scene_records.extend(records)
                entry["processed"] += 1
                write_json(destination / f"s{index:04d}.json", sample)
            roi_dir = destination / "roi"
            roi_dir.mkdir()
            ids = [row["sample_id"] for row in scene["rows"]]
            for part in sorted(
                {r["class_name"] for r in scene_records if r["status"] == "detected"}
            ):
                annotations = make_roi_annotations(
                    scene_records, scene["reference"]["sample_id"], ids, part
                )
                if not annotations[scene["reference"]["sample_id"]]["visible"]:
                    continue
                path = base / "roi" / (file_token(part) + ".json")
                write_json(args.output / path, annotations)
                entry["roi_files"].append(path.as_posix())
            write_json(args.output / "parts_run.json", manifest)
        write_json(args.output / "parts.json", samples)
        fields = list(flat[0])
        with (args.output / "parts.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for record in flat:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in record.items()
                    }
                )
        manifest.update(
            status="complete",
            samples=len(samples),
            detected_instances=sum(r["status"] == "detected" for r in flat),
            no_detection_samples=sum(r["status"] == "no_detection" for r in flat),
        )
        preview(args.output, samples, manifest)
        write_json(args.output / "parts_run.json", manifest)
    except BaseException as error:
        manifest.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "parts_run.json", manifest)
        raise
    print(f"Saved {len(samples)} samples to {args.output / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="CARLA scene or suite directory"
    )
    parser.add_argument("--output", type=Path, default=Path("out/parts01"))
    parser.add_argument(
        "--weights", type=Path, help="existing local vehicle-part .pt checkpoint"
    )
    parser.add_argument("--model-repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--revision", default="main", help="HF revision; checkpoint SHA256 is recorded"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models/vehicle_parts"))
    parser.add_argument(
        "--device", default="auto", help="auto, cpu or GPU index such as 0"
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["door", "window", "roof", "all"],
        default=["door", "window", "roof"],
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate inputs; no model or download"
    )
    args = parser.parse_args()
    if args.imgsz < 32 or not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        raise ValueError("imgsz >=32 and conf/iou in (0,1] are required")
    scenes = discover_scenes(args.data)
    plan = dict(
        data=str(args.data.resolve()),
        scenes=[s["name"] for s in scenes],
        total_crops=sum(len(s["rows"]) for s in scenes),
        rejected_captures=sum(len(s["rejected"]) for s in scenes),
        weights=str(args.weights) if args.weights else args.model_repo,
        parts=args.parts,
        warning="diagnostic only; no exclusion",
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return
    if args.output.exists():
        raise ValueError("output already exists; choose a new run directory")
    if args.output.resolve().is_relative_to(args.data.resolve()):
        raise ValueError("output must be outside the input capture directory")
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if args.weights:
        weights = args.weights.resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"local weights not found: {weights}")
    else:
        from huggingface_hub import hf_hub_download

        weights = Path(
            hf_hub_download(
                args.model_repo,
                "parts_segmentation.pt",
                revision=args.revision,
                cache_dir=str(args.cache_dir),
            )
        )
    model = YOLO(str(weights))
    run_inference(
        args,
        scenes,
        model,
        weights,
        dict(torch=torch.__version__, ultralytics=ultralytics.__version__),
    )


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
