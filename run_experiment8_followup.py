"""Inspect existing experiment 8 results, then run held-out inference on the standard Writer."""

import argparse
import os
from pathlib import Path
from tempfile import mkdtemp

from exp8_case_table import case_table


def report(natural, symmetric, replay, contrasts):
    lines = ["# Experiment 8 follow-up", "",
             f"Original post-write replay vs saved validation: {replay['agree']}/{replay['n']} decisions agree; "
             f"max absolute logit difference {replay['max_logit_error']:.6g}.",
             "", "Natural routes, standard Writer (same memory for original and flipped queries):",
             "group | n | correct orig / flip / mean | flip agrees / helped / harmed | "
             "mean helped / harmed | mean abs margin change (flip)"]
    for name, s in natural.items():
        orig, flip, mean = (s[k] for k in ("original", "flip", "mean"))
        lines.append(f"{name} | {s['n']} | {orig['correct']} / {flip['correct']} / {mean['correct']} | "
                     f"{flip['agree']} / {flip['helped']} / {flip['harmed']} | {mean['helped']} / "
                     f"{mean['harmed']} | {flip['mean_abs_margin_change']:.3f}")
    lines.extend(["", "Symmetric histories, both vehicles receive the same cameras.",
                  "history | method | original correct | front | rear | paired front+rear both correct | "
                  "flip correct / agree / helped / harmed | mean correct | target-other margin"])
    for history, methods in symmetric.items():
        for method, stats in methods.items():
            s = stats["overall"]
            orig, flip = s["original"], s["flip"]
            lines.append(f"{history} | {method} | {orig['correct']}/{s['n']} | "
                         f"{s['front']['original']['correct']}/{s['front']['n']} | "
                         f"{s['rear']['original']['correct']}/{s['rear']['n']} | "
                         f"{orig['both_views_correct']}/{s['paired_views_n']} | {flip['correct']} / "
                         f"{flip['agree']} / {flip['helped']} / {flip['harmed']} | "
                         f"{s['mean']['correct']} | {orig['mean_margin']:+.3f}")
    lines.extend(["", "Per route, original queries: history / method | route | front / rear correct | "
                  "both correct | target-other margin"])
    for history, methods in symmetric.items():
        for method, stats in methods.items():
            for route, s in stats["by_route"].items():
                lines.append(f"{history} / {method} | {route} | {s['front']['original']['correct']}/"
                             f"{s['front']['n']} / {s['rear']['original']['correct']}/{s['rear']['n']} | "
                             f"{s['original']['both_views_correct']}/{s['paired_views_n']} | "
                             f"{s['original']['mean_margin']:+.3f}")
    lines.extend(["", "Same-case contrasts on original photos: baseline -> comparison | questions | "
                  "correct before / after / total | helped / harmed"])
    for c in contrasts:
        lines.append(f"{c['baseline']} -> {c['comparison']} | {c['questions']} | "
                     f"{c['before']} / {c['after']} / {c['n']} | {c['helped']} / {c['harmed']}")
    return "\n".join(lines) + "\n"


def load_writer(folder, config, seed, device):
    import torch
    from dinoexp.replay_matcher import build_replay_matcher

    cp = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    if (cp["kind"] != "memory_replay_matcher_v1" or cp["variant"] != "replay_prepost"
            or cp["seed"] != seed or cp["crop_mode"] != config["crop_mode"]
            or cp["plan_sha256"] != config["plan_sha256"]
            or cp["source_checkpoint_sha256"] != config["source_checkpoints"][str(seed)]):
        raise ValueError(f"checkpoint does not match the ordinary Writer run: {folder}")
    model = build_replay_matcher(cp["config"]).to(device).eval()
    model.load_state_dict(cp["model_state"], strict=True)
    return model


def compare_cutouts(saved, fixed, retrained):
    """Paired, unflipped independent queries. Keep logits and route identity for each case."""
    baseline = {(r["route"], r["probe"]): r["after"] for r in saved if r["kind"] == "new_view"}
    conditions = [baseline]
    for result in (fixed, retrained):
        scores = {(r["route"], r["probe"]): r["scores"]["original"] for r in result if r["kind"] == "new_view"}
        if scores.keys() != baseline.keys():
            raise ValueError("box and cutout questions differ")
        conditions.append(scores)
    names = ("bbox", "same_weights_cutout", "retrained_cutout")
    rows = [dict(route=k[0], probe=k[1], **{name: s[k] for name, s in zip(names, conditions, strict=True)})
            for k in baseline]
    lines = ["Paired natural-route independent queries (unflipped)",
             "group | bbox / same weights on cutouts / retrained on cutouts | n | "
             "same weights helped/harmed vs bbox | retrained helped/harmed vs bbox"]
    for group in ["all", *sorted({r["route"] for r in rows})]:
        sel = [r for r in rows if group == "all" or r["route"] == group]
        ok = [[r[name]["decision"] == "target" for r in sel] for name in names]
        transitions = [(sum(b and not a for a, b in zip(ok[0], cond, strict=True)),
                        sum(a and not b for a, b in zip(ok[0], cond, strict=True))) for cond in ok[1:]]
        lines.append(f"{group} | {sum(ok[0])} / {sum(ok[1])} / {sum(ok[2])} | {len(sel)} | "
                     f"{transitions[0][0]}/{transitions[0][1]} | {transitions[1][0]}/{transitions[1][1]}")
    return rows, "\n".join(lines)


def run(args):
    import torch
    from dinoexp.memory_replay import audit_plan
    from dinoexp.memory_retention_eval import frozen_runtime
    from dinoexp.region_ab_data import read_json, sha, write_json
    from dinoexp.symmetric_memory_eval import (check_angles, encode_inputs, evaluate_route, natural_summary,
                                             paired_contrasts, symmetric_summary)

    bbox = args.bbox
    audit_text = case_table(bbox, args.seed)
    print(audit_text, flush=True)
    data = bbox.parent / "data"
    config = read_json(bbox / "summary.json")
    manifest = read_json(data / "dataset_manifest.json")
    plan_path = data / "plan.json"
    plan = read_json(plan_path)
    modes = {"fixed_height_reviewed_boxes_v1": "bbox", "fixed_height_reviewed_cutouts_v1": "foreground",
             "fixed_height_auto_cutouts_v1": "foreground"}
    if (manifest["schema"] not in modes or config["status"] != "complete"
            or config["crop_mode"] != modes[manifest["schema"]] or config["plan_sha256"] != sha(plan_path)
            or config["dataset_sha256"] != sha(data / "dataset_manifest.json")):
        raise ValueError("expected a completed offline box or cutout run with its matching dataset and plan")
    _, routes, _, _ = audit_plan(plan, config["dataset_sha256"])
    for route in routes:
        check_angles(route, plan["max_query_angle"], plan["novelty_angle"])
    folder = bbox / f"replay_prepost-s{args.seed}"
    checkpoint = folder / "last.pt"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime, _, backbone = frozen_runtime(args)
    if runtime.teacher_sha256 != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("feature encoders differ from original run")
    foreground = config["crop_mode"] == "foreground"
    model = load_writer(folder, config, args.seed, runtime.device)
    comparison = getattr(args, "compare_bbox", None)
    fixed, fixed_symmetric = [], []
    if comparison is not None:
        ref = read_json(comparison / "summary.json")
        if (not foreground or ref["status"] != "complete" or ref["crop_mode"] != "bbox"
                or ref["plan_sha256"] != manifest["base_plan_sha256"]
                or any(ref[k] != config[k] for k in ("teacher_sha256", "backbone_sha256", "source_checkpoints",
                                                       "epochs", "lr", "batch_routes", "postwrite_weight",
                                                       "independent_weight", "weight_decay"))):
            raise ValueError("baseline differs in its data, encoders, initialization, or training settings")
        ref_folder = comparison / folder.name
        bbox_model = load_writer(ref_folder, ref, args.seed, runtime.device)
    natural, symmetric = [], []
    with torch.inference_mode():
        for route in routes:
            inputs = encode_inputs(runtime, data, route, foreground=foreground)
            a, b = evaluate_route(model, route, *inputs)
            natural.extend(a)
            symmetric.extend(b)
            if comparison is not None:
                fixed_a, fixed_b = evaluate_route(bbox_model, route, *inputs)
                fixed.extend(fixed_a)
                fixed_symmetric.extend(fixed_b)
            print(f"Evaluated {route['id']}", flush=True)
    saved = read_json(folder / "validation.json")["rows"]
    old = {(r["route"], r["kind"], r["probe"]): r["after"] for r in saved
           if r["kind"] == "new_view" or (r["kind"] == "self" and r["step"] == 0)}
    if old.keys() != {(r["route"], r["kind"], r["probe"]) for r in natural}:
        raise ValueError("natural-route questions differ from source validation")
    agree, errors = 0, []
    for row in natural:
        baseline = old[row["route"], row["kind"], row["probe"]]
        original = row["scores"]["original"]
        row["saved_original"] = baseline
        agree += baseline["decision"] == original["decision"]
        errors.extend(abs(original[k] - baseline[k]) for k in ("target", "other", "null"))
    replay = dict(agree=agree, n=len(natural), max_logit_error=max(errors))
    ns, ss = natural_summary(natural), symmetric_summary(symmetric)
    if (ns["all_independent"]["n"], ns["first_independent"]["n"], len(symmetric)) != (36, 6, 432):
        raise ValueError("unexpected held-out question coverage")
    contrasts = paired_contrasts(symmetric)
    text = report(ns, ss, replay, contrasts)
    output = Path(mkdtemp(prefix="followup_", dir=bbox.parent))
    if comparison is not None:
        paired, table = compare_cutouts(read_json(ref_folder / "validation.json")["rows"], fixed, natural)
        text += "\n" + table + "\n"
        write_json(output / "cutout_comparison.json", dict(baseline=str(comparison),
                   bbox_checkpoint_sha256=sha(ref_folder / "last.pt"), paired=paired,
                   same_weights_natural=fixed, same_weights_symmetric=fixed_symmetric))
    (output / "cases.txt").write_text(audit_text + "\n", encoding="utf-8")
    (output / "report.txt").write_text(text, encoding="utf-8")
    write_json(output / "questions.json", dict(natural=natural, symmetric=symmetric))
    write_json(output / "summary.json", dict(seed=args.seed, checkpoint_sha256=sha(checkpoint),
               plan_sha256=config["plan_sha256"], crop_mode=config["crop_mode"], source_bbox=str(bbox), replay_check=replay,
               natural=ns, symmetric=ss, contrasts=contrasts))
    print(text, f"Results: {output}", sep="\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bbox", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone")
    parser.add_argument("--model-repo")
    parser.add_argument("--compare-bbox", type=Path, help="original matched bbox run for the cutout ablation")
    run(parser.parse_args())
