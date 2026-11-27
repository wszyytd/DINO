"""Compare vehicle memory updates on replayed crops and independent photos."""

import math
import os
import random

from PIL import Image
import torch
from torch.nn import functional as F

from .core import checked_path
from .evidence_query_prepare import encode_crop
from .memory_replay import PREPOST_VARIANTS, audit_plan, paired_independent_scores
from .memory_retention import open_crop, summarize
from .memory_retention_eval import encode_route_features, evaluate_route, frozen_runtime, load_checkpoints
from .region_ab_data import read_json, sha, write_json
from .region_ab_prepare import cuda_device
from .region_sequence_training import advance
from .replay_matcher import build_replay_matcher, initialize_slot_ema
from .writer_experiment import evaluate_writer, writer_report


def _fixed_features(runtime, root, train, manifest, *, foreground=False):
    cases = {case["scene_id"]: case for case in manifest["cases"]}
    route_features = {}
    for route in train:
        case = cases[route["scene"]]
        if case["split"] != "train" or case["group"] != route["group"]:
            raise ValueError(f"capture split mismatch for {route['id']}")
        scene = read_json(checked_path(root, case["path"]) / "scene.json")
        queries = {}
        for label, uid in enumerate(route["identities"]):
            record = scene["references"][uid]
            relative = record["object"]
            path = checked_path(root, relative)
            if record["target_uid"] != uid or sha(path) != record["hashes"][relative]:
                raise ValueError(f"fixed reference provenance mismatch: {path}")
            if foreground:
                rgb = record["audit"]["rgb"]
                photo, mask = record["full"], record["mask"]
                sample = dict(image=photo, rgb_sha256=record["hashes"][photo],
                              size=[rgb["width"], rgb["height"]], bbox=record["crop_xyxy"],
                              mask=dict(path=mask, sha256=record["hashes"][mask]))
                crop = open_crop(root, sample, foreground=True)
            else:
                with Image.open(path) as image:
                    crop = image.convert("RGB")
            tokens, valid = encode_crop(runtime, crop, "queries")
            if tokens.shape != (64, 256) or valid.shape != (64,) or not valid.any():
                raise ValueError(f"invalid fixed reference features: {path}")
            queries[label] = runtime.tensor(tokens), runtime.tensor(valid)
        route_features[route["id"]] = queries
    return route_features


def _query_loss(model, state, cache, items):
    tensors, masks = zip(*(cache[key] for key, _ in items), strict=True)
    tokens = torch.stack(tensors)
    expanded = {name: value.expand(len(items), *value.shape[1:]) for name, value in state.items()}
    logits = model.score(tokens, torch.stack(masks), expanded)
    targets = torch.tensor([label for _, label in items], device=tokens.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


def route_objective(model, schedule, features, fixed, variant, independent_weight, postwrite_weight):
    """Keep the memory graph connected across all writes within a route."""
    writes, queries = features
    state = model.initial_state(1, 2, next(model.parameters()).device, torch.float32)
    device = state["slots"].device
    main_losses, pre_losses, post_losses = [], [], []
    supervise_post = variant in PREPOST_VARIANTS
    for step in schedule:
        if (supervise_post or variant == "replay_independent") and step["prewrite"]:
            pre_losses.append(_query_loss(model, state, queries, step["prewrite"]))
        observation = torch.zeros((1, 2, 16, 256), device=device)
        valid = torch.zeros((1, 2, 16), dtype=torch.bool, device=device)
        for key, label in step["writes"]:
            observation[0, label], valid[0, label] = writes[key]
        state = advance(model, "memory", observation, valid, state)
        if supervise_post and step["postwrite"]:
            post_losses.append(_query_loss(model, state, queries, step["postwrite"]))
        if variant == "fixed":
            main_losses.append(_query_loss(model, state, fixed, list(enumerate(step["fixed_labels"]))))
        else:
            main_losses.append(_query_loss(model, state, queries, step["replay"]))
    main = torch.stack(main_losses).mean()
    pre = torch.stack(pre_losses).mean() if pre_losses else main.new_zeros(())
    post = torch.stack(post_losses).mean() if post_losses else main.new_zeros(())
    return main + independent_weight * pre + postwrite_weight * post, main, pre, post


def route_loss(model, schedule, features, fixed, variant, independent_weight):
    """Original three-value loss interface used by existing replay callers."""
    return route_objective(model, schedule, features, fixed, variant, independent_weight, 1.)[:3]


def _fit(model, variant, args, seed, train, schedules, features, fixed, folder, *, after_epoch=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log = []
    updates = 0
    for epoch in range(args.epochs):
        model.train()
        order = list(train)
        random.Random(seed * 100_000 + epoch).shuffle(order)
        sums = torch.zeros(4, device=next(model.parameters()).device)
        by_route = {}
        for offset in range(0, len(order), args.batch_routes):
            batch = order[offset:offset + args.batch_routes]
            optimizer.zero_grad(set_to_none=True)
            for route in batch:
                key = route["id"]
                terms = route_objective(model, schedules[key], features[key], fixed.get(key), variant,
                                       args.independent_weight, args.postwrite_weight)
                (terms[0] / len(batch)).backward()
                detached = torch.stack([term.detach() for term in terms])
                sums += detached
                if after_epoch is not None:
                    by_route[key] = detached
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not torch.isfinite(sums).all() or not torch.isfinite(gradient):
                raise ValueError("nonfinite replay training loss or gradient")
            optimizer.step()
            updates += 1
        losses = (sums / len(order)).tolist()
        log.append(dict(epoch=epoch + 1, updates=updates, loss=losses[0],
                        main_ce=losses[1], independent_ce=losses[2], independent_post_ce=losses[3]))
        if after_epoch is not None:
            fields = ("loss", "main_ce", "independent_ce", "independent_post_ce")
            log[-1]["by_route"] = {key: dict(zip(fields, values.tolist(), strict=True))
                                   for key, values in by_route.items()}
        write_json(folder / "train.json", log)
        if after_epoch is not None:
            after_epoch(model, log[-1])
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"{folder.name} epoch {epoch + 1}/{args.epochs}: loss={losses[0]:.5f}", flush=True)
    return updates


def _metrics(rows):
    groups = summarize(rows)["val"]
    independent = [row for row in rows if row.get("before_frame", {}).get("margin") is not None]
    first = [row for row in rows if row["kind"] == "self" and row["prior_writes"] == 0]
    return dict(groups=groups, paired_independent=paired_independent_scores(rows), independent_prewrite=dict(
        total=len(independent), correct=sum(r["before_frame"]["decision"] == "target" for r in independent)),
        first_exact=dict(total=len(first),
                         correct=sum(r["after"]["decision"] == "target" for r in first),
                         other=sum(r["after"]["decision"] == "other" for r in first),
                         null=sum(r["after"]["decision"] == "null" for r in first)))


@torch.inference_mode()
def _validate(model, routes, features, novelty):
    model.eval()
    rows = []
    by_route = {}
    for route in routes:
        readings, _ = evaluate_route(model, "memory", route, features[route["id"]], novelty,
                                      repeat=False, prewrite=True)
        by_route[route["id"]] = _metrics(readings)
        rows.extend(readings)
    return dict(aggregate=_metrics(rows), by_route=by_route, rows=rows)


def _report(results):
    lines = ["# Vehicle memory replay",
             "",
             "Frozen initialization and all fine-tuned variants use identical held-out routes and queries.",
             "Fine-tuning uses the final epoch. Reported pre-write accuracy requires both candidates in memory.",
             "",
             "| Model | First exact | Old exact correct | Initially right → later wrong | Independent pre-write | Independent post-write |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in results.items():
        old = metrics["groups"]["old_exact"]
        new = metrics["groups"]["new_view"]
        first = metrics["first_exact"]
        pre = metrics["independent_prewrite"]
        lines.append(f"| {name} | {first['correct']}/{first['total']} | {old['correct_after']}/{old['total']} | "
                     f"{old['later_wrong']}/{old['initially_correct']} | {pre['correct']}/{pre['total']} | "
                     f"{new['correct_after']}/{new['total']} |")
    lines.extend(["", "Same independent photos before/after the entire frame; both candidates available.",
                  "Old-view probes are separate photos from previously queried angles. Repeated reads count.",
                  "", "| Model | Paired reads | Pre correct | Post correct | Help | Harm | Balanced post | Old-view correct |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, metrics in results.items():
        paired = metrics["paired_independent"]
        balanced = paired["balanced_post_accuracy"]
        balanced_text = "—" if balanced is None else f"{balanced:.1%}"
        old_view = metrics["groups"].get("old_view", {})
        lines.append(f"| {name} | {paired['total']} | {paired['pre_correct']} | {paired['post_correct']} | "
                     f"{paired['help']} | {paired['harm']} | {balanced_text} | "
                     f"{old_view.get('correct_after', 0)}/{old_view.get('total', 0)} |")
    return "\n".join(lines) + "\n"


def run(args):
    if (min(args.epochs, args.batch_routes, args.threads) < 1 or not math.isfinite(args.lr)
            or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not 0 <= args.ema_decay <= 1 or not math.isfinite(args.postwrite_weight)
            or args.postwrite_weight <= 0
            or not math.isfinite(args.independent_weight) or args.independent_weight <= 0):
        raise ValueError("invalid replay training hyperparameters")
    if args.output.exists():
        raise ValueError("output already exists; choose a fresh experiment directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plan_hash = sha(args.plan)
    plan = read_json(args.plan)
    manifest_path = args.data / "dataset_manifest.json"
    train, val, schedules, coverage = audit_plan(plan, sha(manifest_path))
    sources = load_checkpoints(args.source_run, ["memory"], args.seed)
    device = cuda_device(args.device)
    torch.set_num_threads(args.threads)
    manifest = read_json(manifest_path)
    capture_cases = {case["scene_id"]: (case["split"], case["group"]) for case in manifest["cases"]}
    if any(capture_cases.get(r["scene"]) != (r["split"], r["group"]) for r in plan["routes"]):
        raise ValueError("route split/background differs from the capture manifest")
    variants = list(dict.fromkeys(args.variant))
    foreground = args.crop_mode == "foreground"
    with torch.no_grad():
        runtime, source_plan, backbone_hash = frozen_runtime(args)
        features = {}
        for route in plan["routes"]:
            features[route["id"]] = encode_route_features(runtime, args.data, route, foreground=foreground)
            print(f"Encoded {route['id']}", flush=True)
        fixed = _fixed_features(runtime, args.data, train, manifest, foreground=foreground) if "fixed" in variants else {}
    del runtime
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    info = dict(kind="memory_replay_training_v1", status="running", plan_sha256=plan_hash,
                dataset_sha256=plan["dataset_sha256"], teacher_sha256=source_plan["teacher_sha256"],
                backbone_sha256=backbone_hash, coverage=coverage, excluded=plan["excluded"],
                epochs=args.epochs, batch_routes=args.batch_routes, lr=args.lr,
                independent_weight=args.independent_weight, weight_decay=args.weight_decay,
                crop_mode=args.crop_mode, ema_decay=args.ema_decay, postwrite_weight=args.postwrite_weight,
                source_checkpoints={r["seed"]: r["checkpoint_sha256"] for r, _, _ in sources},
                variants=variants, results={})
    write_json(args.output / "summary.json", info)
    writer_diagnostics, validation_routes = {}, {}
    for record, model_config, weights in sources:
        seed = record["seed"]
        for variant in ("initial", *variants):
            name = f"{variant}-s{seed}"
            folder = args.output / name
            folder.mkdir()
            if variant == "ema_prepost":
                model = initialize_slot_ema(model_config, weights, args.ema_decay).to(device)
            else:
                config = dict(model_config)
                if variant == "competitive_prepost":
                    config["writer_attention"] = "competitive"
                model = build_replay_matcher(config).to(device)
                model.load_state_dict(weights, strict=True)
            if variant != "initial":
                updates = _fit(model, variant, args, seed, train, schedules, features, fixed, folder)
                torch.save(dict(kind="memory_replay_matcher_v1", variant=variant, seed=seed,
                               config=model.config, model_state=model.state_dict(), updates=updates,
                               crop_mode=args.crop_mode, postwrite_weight=args.postwrite_weight,
                               independent_weight=args.independent_weight,
                               plan_sha256=plan_hash, source_checkpoint_sha256=record["checkpoint_sha256"]),
                           folder / "last.pt")
            result = _validate(model, val, features, plan["novelty_angle"])
            write_json(folder / "validation.json", result)
            info["results"][name] = result["aggregate"]
            if "competitive_prepost" in variants:
                writer_diagnostics[name] = evaluate_writer(model, val, features, plan, folder)
                validation_routes[name] = result["by_route"]
                info["writer_diagnostics"] = writer_diagnostics
                (args.output / "writer_report.md").write_text(
                    writer_report(writer_diagnostics, validation_routes), encoding="utf-8")
            write_json(args.output / "summary.json", info)
            del model
    if sha(args.plan) != plan_hash or sha(manifest_path) != plan["dataset_sha256"]:
        raise ValueError("capture or replay plan changed during training")
    (args.output / "report.md").write_text(_report(info["results"]), encoding="utf-8")
    info["status"] = "complete"
    write_json(args.output / "summary.json", info)
    return dict(status="complete", report=str(args.output / "report.md"), coverage=coverage)
