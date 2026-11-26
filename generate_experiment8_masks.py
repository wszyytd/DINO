"""Optional regeneration of the reviewed-car masks. The server uses the shipped pack."""

import argparse
import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.transforms.functional import to_tensor


def trim_reviewed_boundary(mask, scene, sample):
    # Parking lines/stops touch the white Charger's roof/trunk in these validation views.
    # Boundary vertices were marked on a 400 x 290 display of the source RGB crop.
    if scene not in ("pair0_location1_refA", "pair1_location1_refA") or not sample["identity"].endswith(":A"):
        return mask
    edges = {
        13: [(184, 65), (218, 65), (243, 66), (258, 70), (270, 78)],
        17: [(304, 83), (320, 91), (345, 96), (356, 101), (364, 114), (374, 132), (400, 132)],
        20: [(168, 73), (185, 65), (207, 59), (234, 55), (248, 52), (271, 56),
             (287, 60), (299, 59), (317, 54), (328, 54), (344, 65), (357, 79)],
        21: [(218, 58), (239, 54), (256, 53), (277, 55), (289, 54), (305, 48),
             (317, 46), (331, 50), (347, 58), (358, 65)],
    }
    edge = edges.get(sample["camera_index"])
    if edge:
        polygon = [(edge[0][0], 0), *edge, (edge[-1][0], 0)]
        ImageDraw.Draw(mask).polygon([(round(x*mask.width/400), round(y*mask.height/290))
                                      for x, y in polygon], fill=0)
    return mask


@torch.inference_mode()
def segment(model, scene, box):
    x, y, right, bottom = box
    w, h = right-x, bottom-y
    pad = round(0.25 * max(w, h))
    left, top = max(0, x-pad), max(0, y-pad)
    context = scene.crop((left, top, min(scene.width, right+pad), min(scene.height, bottom+pad)))
    rect = torch.tensor([[x-left, y-top, right-left, bottom-top]], dtype=torch.float32)
    # Use the known vehicle box as the mask head proposal. A second car can be in the context.
    images, targets = model.transform([to_tensor(context)], [{"boxes": rect}])
    heads = model.roi_heads
    pooled = heads.mask_roi_pool(model.backbone(images.tensors), [targets[0]["boxes"]], images.image_sizes)
    logits = heads.mask_predictor(heads.mask_head(pooled))
    car = logits[:, 3:4].sigmoid()  # torchvision's COCO category 3 is car
    scores = paste_masks_in_image(car, rect, (context.height, context.width))[0, 0].numpy()
    scores = scores[y-top:bottom-top, x-left:right-left]
    rgb = np.array(scene.crop(box))
    rough = (scores >= 0.5).astype(np.uint8)
    radius = max(2, round(min(w, h) * 0.023))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    core = cv2.erode(rough, kernel).astype(bool)
    expanded = cv2.dilate(rough, kernel).astype(bool)
    if not core.any() or expanded.all():
        raise ValueError(f"unusable initial mask at {box}")
    trimap = np.zeros_like(rough)
    trimap[expanded] = cv2.GC_PR_BGD
    trimap[rough.astype(bool)] = cv2.GC_PR_FGD
    trimap[core] = cv2.GC_FGD
    cv2.setRNGSeed(7)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), trimap, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    mask = (trimap & 1).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return Image.fromarray(np.uint8(labels == largest)*255)


def generate(args):
    weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if not weights_hash.startswith("73cbd019"):
        raise ValueError("expected torchvision COCO Mask R-CNN ResNet-50 FPN V2 weights")
    plan = json.loads((args.reviewed / "plan.json").read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=448, max_size=768).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    records = {}
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as pack:
        for route in plan["routes"]:
            folder = args.source / route["scene"]
            with (folder / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = {r["crop"]: r for r in csv.DictReader(stream)}
            for key, sample in route["samples"].items():
                row = rows[sample["source_crop"]]
                box = json.loads(row["crop_box"])
                path = args.reviewed / sample["image"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != sample["rgb_sha256"]:
                    raise ValueError(f"unexpected RGB bytes: {key}")
                with Image.open(folder / row["image"]) as image, Image.open(path) as reviewed:
                    scene = image.convert("RGB")
                    if scene.crop(box).tobytes() != reviewed.convert("RGB").tobytes():
                        raise ValueError(f"wrong crop coordinates: {key}")
                    mask = trim_reviewed_boundary(segment(model, scene, box), route["scene"], sample)
                buf = BytesIO()
                mask.save(buf, format="PNG")
                member = sample["image"][:-4]+".mask.png"
                pack.writestr(member, buf.getvalue())
                records[key] = dict(mask=member, rgb_sha256=sample["rgb_sha256"],
                                    mask_sha256=hashlib.sha256(buf.getvalue()).hexdigest())
            print(f"Segmented {route['id']}: {len(records)}", flush=True)
        pack.writestr("index.json", json.dumps(dict(weights_sha256=weights_hash,
                      method="torchvision Mask R-CNN ResNet50 FPN V2, known box/car class, GrabCut boundary, reviewed edge trims",
                      records=records), sort_keys=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/fixed_height_similar_20261104"))
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    generate(parser.parse_args())
