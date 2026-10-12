"""Frozen DINOv2 cosine matching; scores are not probabilities or exclusion decisions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from dinoexp.core import checked_path, cosine_scores, local_box, prepare_image
from dinoexp.dataset import load_dataset


def encode(images, model, torch, device, size, batch_size, pooling):
    features = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = []
            for im in images[start : start + batch_size]:
                array = np.asarray(prepare_image(im, size), dtype=np.float32) / 255
                batch.append(((array - mean) / std).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch)).to(device)
            if pooling == "cls":
                feature = model(x)
            else:
                feature = model.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
            features.append(feature.float().cpu().numpy())
            print(f"Encoded {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return np.concatenate(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("out/probe01"))
    parser.add_argument(
        "--model", choices=["dinov2_vits14", "dinov2_vitb14"], default="dinov2_vits14"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pooling", choices=["cls", "patch_mean"], default="cls")
    parser.add_argument(
        "--roi", type=Path, help="manual normalized ROI annotations on vehicle crops"
    )
    parser.add_argument("--repo", default="facebookresearch/dinov2:main", help="Hub repo/ref")
    parser.add_argument("--model-repo", type=Path, help="local DINOv2 checkout for offline loading")
    parser.add_argument(
        "--checkpoint", type=Path, help="local official backbone weights (requires --model-repo)"
    )
    args = parser.parse_args()
    if args.size <= 0 or args.size % 14 or args.batch_size < 1:
        raise ValueError("size must be divisible by 14; batch size must be positive")
    if args.checkpoint and not args.model_repo:
        raise ValueError("--checkpoint requires --model-repo")
    ref, queries, rejected = load_dataset(args.data)
    if args.output.exists():
        raise ValueError("output already exists; choose a new run directory")
    rows = [ref] + queries
    images = []
    for row in rows:
        with Image.open(checked_path(args.data, row["crop"])) as im:
            images.append(im.convert("RGB"))
    annotations = json.loads(args.roi.read_text(encoding="utf-8-sig")) if args.roi else {}
    if annotations and set(annotations) - {r["sample_id"] for r in rows} - {
        r["sample_id"] for r in rejected
    }:
        raise ValueError("ROI contains unknown sample IDs")
    local_images, local_indices = [], []
    if args.roi:
        ref_ann = annotations.get(ref["sample_id"], {})
        ref_box = local_box(ref_ann, ref_ann, images[0].size)
        if ref_box is None:
            raise ValueError("local comparison needs a visible, named reference ROI")
        local_images.append(images[0].crop(ref_box))
        for i, row in enumerate(queries, start=1):
            box = local_box(annotations.get(row["sample_id"], {}), ref_ann, images[i].size)
            if box is not None:
                local_indices.append(i - 1)
                local_images.append(images[i].crop(box))
    import torch

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu or install matching PyTorch")
    if args.model_repo:
        model = torch.hub.load(
            str(args.model_repo), args.model, source="local", pretrained=not bool(args.checkpoint)
        )
    else:
        model = torch.hub.load(args.repo, args.model, trust_repo=True)
    checkpoint_hash = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_hash = digest.hexdigest()
    model = model.to(device).eval()
    features = encode(images, model, torch, device, args.size, args.batch_size, args.pooling)
    scores = cosine_scores(features[0], features[1:])
    local_features = None
    local_scores = {}
    if len(local_images) > 1:
        local_features = encode(
            local_images, model, torch, device, args.size, args.batch_size, args.pooling
        )
        values = cosine_scores(local_features[0], local_features[1:])
        local_scores = dict(zip(local_indices, values, strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "features.npz",
        features=features,
        sample_ids=np.array([r["sample_id"] for r in rows]),
    )
    if local_features is not None:
        np.savez_compressed(
            args.output / "local_features.npz",
            features=local_features,
            sample_ids=np.array(
                [ref["sample_id"]] + [queries[i]["sample_id"] for i in local_indices]
            ),
        )
    fields = list(queries[0]) + ["whole_score", "local_score", "local_status", "is_target"]
    with (args.output / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(queries):
            writer.writerow(
                dict(
                    row,
                    whole_score=float(scores[i]),
                    local_score=float(local_scores[i]) if i in local_scores else "",
                    local_status="compared" if i in local_scores else "unavailable",
                    is_target=str(row["identity"] == ref["identity"]).lower(),
                )
            )
    run = {
        "model": args.model,
        "repo": str(args.model_repo or args.repo),
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "torch": torch.__version__,
        "size": args.size,
        "pooling": args.pooling,
        "preprocess": "aspect-preserving letterbox + ImageNet normalization",
        "reference": ref,
        "data": str(args.data.resolve()),
        "queries": len(queries),
        "rejected_views": len(rejected),
        "local_compared": len(local_scores),
        "metadata_sha256": hashlib.sha256((args.data / "metadata.csv").read_bytes()).hexdigest(),
        "roi": annotations,
        "warning": "cosine similarity is not probability; no exclusion rule",
    }
    (args.output / "match_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"Saved {len(queries)} scores to {args.output / 'scores.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
