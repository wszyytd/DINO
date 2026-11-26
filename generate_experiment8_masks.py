"""Generate automatic vehicle masks from the source images on the experiment machine."""

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
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.transforms.functional import to_tensor

from dinoexp.region_ab_data import sha


WEIGHTS_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"


def resolve_weights(path):
    if path is None:
        url = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url
        path = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path), hash_prefix=WEIGHTS_SHA256)
    if sha(path) != WEIGHTS_SHA256:
        raise ValueError(f"incorrect COCO Mask R-CNN ResNet-50 FPN V2 weights: {path}")
    return path


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


def generate(source, reviewed, output, weights=None, threads=4):
    weights = resolve_weights(weights)
    plan = json.loads((reviewed / "plan.json").read_text(encoding="utf-8"))
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=448, max_size=768).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    records = {}
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as pack:
        for route in plan["routes"]:
            folder = source / route["scene"]
            with (folder / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = {r["crop"]: r for r in csv.DictReader(stream)}
            for key, sample in route["samples"].items():
                row = rows[sample["source_crop"]]
                box = json.loads(row["crop_box"])
                path = reviewed / sample["image"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != sample["rgb_sha256"]:
                    raise ValueError(f"unexpected RGB bytes: {key}")
                with Image.open(folder / row["image"]) as image, Image.open(path) as crop:
                    scene = image.convert("RGB")
                    if scene.crop(box).tobytes() != crop.convert("RGB").tobytes():
                        raise ValueError(f"wrong crop coordinates: {key}")
                    mask = segment(model, scene, box)
                buf = BytesIO()
                mask.save(buf, format="PNG")
                member = sample["image"][:-4]+".mask.png"
                pack.writestr(member, buf.getvalue())
                records[key] = dict(mask=member, rgb_sha256=sample["rgb_sha256"],
                                    mask_sha256=hashlib.sha256(buf.getvalue()).hexdigest())
            print(f"Segmented {route['id']}: {len(records)}", flush=True)
        pack.writestr("index.json", json.dumps(dict(schema="automatic_vehicle_masks_v1", weights_sha256=WEIGHTS_SHA256,
                      method="torchvision Mask R-CNN ResNet50 FPN V2, known box/car class, GrabCut boundary and largest component",
                      records=records), sort_keys=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="use an existing official COCO weight file")
    parser.add_argument("--source", type=Path, default=Path("data/fixed_height_similar_20261104"))
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    generate(**vars(parser.parse_args()))
