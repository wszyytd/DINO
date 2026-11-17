"""CUDA inference over a frozen plan and existing region-sequence checkpoints."""

import os
from pathlib import Path

import torch

from .core import checked_path
from .evidence_memory import EvidenceMatcher
from .evidence_query_prepare import encode_crop
from .evidence_query_training import clone_state
from .memory_retention import open_crop, score_pair, summarize
from .region_ab_data import read_json, sha, write_json
from .region_ab_prepare import FrozenRuntime, cuda_device
from .region_sequence_training import advance


def load_checkpoints(source, methods, seeds):
    run = read_json(source / "training/run.json")
    if run["kind"] != "region_sequence_training_v1" or run["status"] != "complete":
        raise ValueError("source region-sequence training is incomplete")
    hashes = {key: sha(source / f"{key}.json") for key in ("plan", "prepared")}
    if any(run[f"{key}_sha256"] != digest for key, digest in hashes.items()):
        raise ValueError("source training plan/prepared hash mismatch")
    records = {(row["method"], row["seed"]): row for row in run["results"]}
    checkpoints = []
    for pair in dict.fromkeys((method, seed) for method in methods for seed in seeds):
        record = records.get(pair)
        if record is None:
            raise ValueError(f"checkpoint not in source run: {pair}")
        path = checked_path(source, record["checkpoint"])
        if sha(path) != record["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if (cp["kind"] != "region_sequence_matcher_v1" or (cp["method"], cp["seed"]) != pair
                or any(cp[f"{k}_sha256"] != digest for k, digest in hashes.items())):
            raise ValueError(f"checkpoint provenance mismatch: {path}")
        checkpoints.append((record, cp["config"], cp["model_state"]))
    return checkpoints


def encode_route_features(runtime, root, route):
    write_keys = {write["sample"] for step in route["steps"] for write in step["writes"].values()}
    writes, queries = {}, {}
    for key, sample in route["samples"].items():
        crop = open_crop(root, sample)
        for role, cache, count in (("observations", writes, 16), ("queries", queries, 64)):
            if role == "observations" and key not in write_keys:
                continue
            tokens, valid = encode_crop(runtime, crop, role)
            if (tokens.shape != (count, 256) or valid.shape != (count,) or not valid.any()
                    or not torch.isfinite(torch.from_numpy(tokens)).all()):
                raise ValueError(f"invalid {role} feature: {route['id']}/{key}")
            cache[key] = runtime.tensor(tokens), runtime.tensor(valid)
    return writes, queries


def _read(model, state, queries, items, target):
    if not items:
        return []
    n = len(items)
    tokens, valid = zip(*(queries[item["probe"]] for item in items), strict=True)
    tokens, valid = torch.stack(tokens), torch.stack(valid)
    expanded = {key: value.expand(n, *value.shape[1:]) for key, value in state.items()}
    logits = model.score(tokens, valid, expanded).cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite matcher scores")
    seen = state["seen"][0].tolist()
    return [score_pair(row, seen, target) for row in logits.tolist()]


def _item(kind, probe, step, **fields):
    return dict(kind=kind, probe=probe, anchor_step=step, **fields)


def evaluate_route(model, method, route, features, novelty, *, repeat=True, prewrite=False):
    writes, queries = features
    device = next(model.parameters()).device
    state = model.initial_state(1, 2, device, torch.float32)
    rows, replay, interventions = [], [[], []], []
    reference = dict(route=route["id"], split=route["split"], group=route["group"])
    for step in route["steps"]:
        t = step["step"]
        obs = torch.zeros((1, 2, 16, 256), device=device)
        valid = torch.zeros((1, 2, 16), dtype=torch.bool, device=device)
        for i, uid in enumerate(route["identities"]):
            if uid in step["writes"]:
                obs[0, i], valid[0, i] = writes[step["writes"][uid]["sample"]]
        before = state
        state = advance(model, method, obs, valid, clone_state(before)) if step["writes"] else before

        counts = before["count"][0].tolist()
        for i, uid in enumerate(route["identities"]):
            write = step["writes"].get(uid)
            items = list(replay[i])
            if write:
                items.append(_item("self", write["sample"], t))
                if write["independent"]:
                    probe_gap = write["probe_previous_view_gap"]
                    items.append(_item("new_view", write["independent"], t,
                        previous_view_gap=write["previous_view_gap"], probe_previous_view_gap=probe_gap,
                        novel=(probe_gap or 0) >= novelty and (write["previous_view_gap"] or 0) >= novelty))
            if not items:
                continue

            base = dict(reference, step=t, frame=step["frame"], identity=uid, intervened=bool(write),
                        prior_writes=int(counts[i]),
                        observation=write["sample"] if write else None)
            if write:
                skip_valid = valid.clone()
                skip_valid[:, i] = False
                skipped = advance(model, method, obs, skip_valid, clone_state(before))
                if before["seen"][0, i] and not torch.equal(state["seen"], skipped["seen"]):
                    raise ValueError("skip changed candidate availability after first write")
            else:
                skipped = state
            normal_scores = _read(model, state, queries, items, i)
            skip_scores = _read(model, skipped, queries, items, i) if write else normal_scores
            prospective = next((item for item in items if item["kind"] == "new_view"), None)
            pre_frame = (_read(model, before, queries, [prospective], i)[0]
                         if prewrite and prospective and counts[i] else None)
            for item, normal, skip in zip(items, normal_scores, skip_scores, strict=True):
                row = dict(base, **item, before=skip, after=normal)
                if item is prospective and pre_frame is not None:
                    row["before_frame"] = pre_frame
                rows.append(row)
                if item["kind"] == "self":
                    replay[i].append(_item("old_exact", item["probe"], t, origin=normal))
                elif item["kind"] == "new_view" and all(p["probe"] != item["probe"] for p in replay[i]):
                    replay[i].append(_item("old_view", item["probe"], t, origin=normal))

            if not write or not repeat:
                continue
            repeat_valid = torch.zeros_like(valid)
            repeat_valid[:, i] = valid[:, i]
            repeated = clone_state(state)
            start_slots = state["slots"][0, i]
            for count in range(1, 9):
                repeated = advance(model, method, obs, repeat_valid, repeated)
                if count not in (1, 8):
                    continue
                end_slots = repeated["slots"][0, i]
                drift = float((end_slots - start_slots).norm() / start_slots.norm().clamp_min(1e-12))
                interventions.append(dict(base, repetitions=count, relative_slot_drift=drift))
                repeated_scores = _read(model, repeated, queries, items, i)
                for item, normal, again in zip(items, normal_scores, repeated_scores, strict=True):
                    rows.append(dict(base, kind=f"repeat_{count}", probe_kind=item["kind"], probe=item["probe"],
                                     anchor_step=item["anchor_step"], before=normal, after=again))
    return rows, interventions


def _markdown(summary, plan):
    lines = ["# Vehicle memory retention", "",
             "Ground-truth crop/identity routing isolates the existing matcher's memory. "
             "All checkpoints are frozen. Validation is by background for the same two CARLA assets.", "",
             "Self is the first read of the exact crop. Old exact/view probes are fixed after their first read. "
             "New view is a separate crop near the current view. Novel view is the subset far from earlier writes. "
             "The write-count rows group those new-view probes by evidence available after the write.", "",
             "Help/harm use identical candidate availability and include Null. "
             "Margin is target logit minus other-car logit. Correct is measured after the intervention.", "",
             "Forced repeats start from the normal post-write state and never enter the next step.", ""]
    for case in plan["excluded"]:
        lines.append(f"- Excluded `{case['scene']}` ({case['split']}) with {case['bad_file_count']} missing/empty files.")
    for name, result in summary.items():
        lines.extend(["", f"## {name}", "",
            "| Split | Probe | Correct / total | Distinct probes | Paired | Help | Harm | Decision flips | Mean margin change | Mean absolute change | Init. right → later wrong |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for split, groups in result["scores"].items():
            for kind, m in groups.items():
                change = m["mean_margin_delta"]
                formatted = "—" if change is None else f"{change:.4f}"
                absolute = m["mean_abs_margin_delta"]
                abs_formatted = "—" if absolute is None else f"{absolute:.4f}"
                lines.append(f"| {split} | {kind} | {m['correct_after']}/{m['total']} | {m['unique_probes']} | "
                             f"{m['paired']} | {m['help']} | {m['harm']} | {m['changed']} | {formatted} | {abs_formatted} | "
                             f"{m['later_wrong']}/{m['initially_correct']} |")
        lines.extend(["", "| Split | Extra writes | Writes | Mean relative slot drift |",
                      "| --- | ---: | ---: | ---: |"])
        for m in result["slot_drift"]:
            lines.append(f"| {m['split']} | {m['repetitions']} | {m['writes']} | {m['mean']:.6f} |")
    return "\n".join(lines) + "\n"


def frozen_runtime(args):
    args.dense_cache = None
    args.view_batch = 1
    source = read_json(args.source_run / "plan.json")
    runtime = FrozenRuntime(args)
    backbone_hash = sha(runtime.backbone_path)
    if (runtime.teacher_sha256 != source["teacher_sha256"] or backbone_hash != source["backbone_sha256"]
            or runtime.cfg != source["teacher_feature_config"]):
        raise ValueError("teacher/backbone do not match checkpoint training features")
    from .evidence_query_data import verify_code_sources
    verify_code_sources(Path(__file__).resolve().parents[1], source["code"])
    verify_code_sources(runtime.repo, source["backbone_source"])
    return runtime, source, backbone_hash


@torch.inference_mode()
def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plan_path = args.output / "plan.json"
    plan = read_json(plan_path)
    plan_digest = sha(plan_path)
    if (plan["kind"] != "memory_retention_plan_v1"
            or sha(args.data / "dataset_manifest.json") != plan["dataset_sha256"]):
        raise ValueError("capture manifest differs from frozen plan")
    result_dir = args.output / "evaluation"
    if result_dir.exists():
        raise ValueError("evaluation already exists; preserve it and use a fresh plan directory")
    checkpoints = load_checkpoints(args.source_run, args.method, args.seed)
    device = cuda_device(args.device)
    torch.set_num_threads(args.threads)
    runtime, source, backbone_hash = frozen_runtime(args)
    features = {}
    for route in plan["routes"]:
        features[route["id"]] = encode_route_features(runtime, args.data, route)
        print(f"Encoded {route['id']}", flush=True)
    del runtime
    result_dir.mkdir()
    summary = {}
    for record, config, weights in checkpoints:
        method, seed = record["method"], record["seed"]
        model = EvidenceMatcher(**config).to(device).eval()
        model.load_state_dict(weights, strict=True)
        rows, writes = [], []
        for route in plan["routes"]:
            readings, drift = evaluate_route(model, method, route, features[route["id"]], plan["novelty_angle"])
            rows.extend(readings)
            writes.extend(drift)
        name = f"{method}-s{seed}"
        drift_summary = []
        for split in sorted({row["split"] for row in writes}):
            for count in (1, 8):
                values = [row["relative_slot_drift"] for row in writes
                          if row["split"] == split and row["repetitions"] == count]
                drift_summary.append(dict(split=split, repetitions=count, writes=len(values),
                                          mean=sum(values) / len(values)))
        summary[name] = dict(scores=summarize(rows), slot_drift=drift_summary)
        write_json(result_dir / f"{name}.json", dict(checkpoint=record["checkpoint"],
                   checkpoint_sha256=record["checkpoint_sha256"], rows=rows, repeat_writes=writes))
        print(f"Scored {name}: {len(rows)} rows", flush=True)
    if sha(plan_path) != plan_digest:
        raise ValueError("plan changed during evaluation")
    (result_dir / "report.md").write_text(_markdown(summary, plan), encoding="utf-8")
    write_json(result_dir / "summary.json", dict(status="complete", results=summary, plan_sha256=plan_digest,
               coverage=plan["coverage"], excluded=plan["excluded"], teacher_sha256=source["teacher_sha256"],
               backbone_sha256=backbone_hash))
    return dict(status="complete", report=str(result_dir / "report.md"), checkpoints=list(summary))
