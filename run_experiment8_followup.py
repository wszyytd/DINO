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


def run(args):
    import torch
    from dinoexp.memory_replay import audit_plan
    from dinoexp.memory_retention_eval import frozen_runtime
    from dinoexp.region_ab_data import read_json, sha, write_json
    from dinoexp.replay_matcher import build_replay_matcher
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
    if (manifest["schema"] != "fixed_height_reviewed_boxes_v1" or config["status"] != "complete"
            or config["crop_mode"] != "bbox" or config["plan_sha256"] != sha(plan_path)
            or config["dataset_sha256"] != sha(data / "dataset_manifest.json")):
        raise ValueError("expected the completed offline vehicle-box pilot and its original plan")
    _, routes, _, _ = audit_plan(plan, config["dataset_sha256"])
    for route in routes:
        check_angles(route, plan["max_query_angle"], plan["novelty_angle"])
    folder = bbox / f"replay_prepost-s{args.seed}"
    checkpoint = folder / "last.pt"
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (cp["kind"] != "memory_replay_matcher_v1" or cp["variant"] != "replay_prepost"
            or cp["seed"] != args.seed or cp["crop_mode"] != "bbox" or cp["plan_sha256"] != config["plan_sha256"]
            or cp["source_checkpoint_sha256"] != config["source_checkpoints"][str(args.seed)]):
        raise ValueError("checkpoint does not match the offline ordinary Writer run")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime, _, backbone = frozen_runtime(args)
    if runtime.teacher_sha256 != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("feature encoders differ from original run")
    model = build_replay_matcher(cp["config"]).to(runtime.device).eval()
    model.load_state_dict(cp["model_state"], strict=True)
    natural, symmetric = [], []
    with torch.inference_mode():
        for route in routes:
            inputs = encode_inputs(runtime, data, route)
            a, b = evaluate_route(model, route, *inputs)
            natural.extend(a)
            symmetric.extend(b)
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
    (output / "cases.txt").write_text(audit_text + "\n", encoding="utf-8")
    (output / "report.txt").write_text(text, encoding="utf-8")
    write_json(output / "questions.json", dict(natural=natural, symmetric=symmetric))
    write_json(output / "summary.json", dict(seed=args.seed, checkpoint_sha256=sha(checkpoint),
               plan_sha256=cp["plan_sha256"], source_bbox=str(bbox), replay_check=replay,
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
    run(parser.parse_args())
