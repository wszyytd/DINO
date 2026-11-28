"""Evaluate zero-training shared view routing on experiment 8's four-view questions."""

import argparse
import math
import os
from pathlib import Path
from tempfile import mkdtemp

from dinoexp.memory_replay import audit_plan
from dinoexp.region_ab_data import sha, write_json
from run_experiment8_followup import load_writer
from run_experiment8_four_views import experiment_source
from run_experiment8_long import SEED


def run(args):
    import torch
    from dinoexp.experiment8_view_routing import POOLING, encode_route, evaluate_routes, pooling_report, report, summarize
    from dinoexp.four_view_memory import select_views
    from dinoexp.memory_retention_eval import frozen_runtime

    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("routing temperature must be finite and positive")
    compare_pooling = getattr(args, "compare_pooling", False)
    baseline, config, plan = experiment_source(args.checkpoints, args.baseline)
    _, routes, _, _ = audit_plan(plan, config["dataset_sha256"])
    selections = {r["id"]: select_views(r, plan["max_query_angle"]) for r in routes}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    runtime, source, backbone = frozen_runtime(args)
    if source["teacher_sha256"] != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("feature encoders differ from the checkpoint experiment")
    checkpoint = args.checkpoints / "epoch_100"
    model = load_writer(checkpoint, config, SEED, runtime.device, epoch=100)
    if args.output is None:
        Path("out").mkdir(exist_ok=True)
        prefix = "experiment8_view_pooling_" if compare_pooling else "experiment8_view_routing_"
        output = Path(mkdtemp(prefix=prefix, dir="out"))
    else:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {output}", flush=True)
    info = dict(status="running", source_epoch=100, seed=SEED, source_run=str(args.checkpoints),
                baseline=str(baseline), plan_sha256=config["plan_sha256"],
                dataset_sha256=config["dataset_sha256"], teacher_sha256=config["teacher_sha256"],
                backbone_sha256=backbone, checkpoint_sha256=sha(checkpoint/"last.pt"),
                descriptor="L2-normalized mean of non-padding patches from the deepest selected DINOv2 layer",
                descriptor_layer=runtime.cfg["layers"][-1], temperature=args.temperature, selections=selections)
    if compare_pooling:
        info["pooling"] = dict(
            global_mean="non-padding patch mean",
            foreground_mean="area-weighted patch mean from the instance mask",
            foreground_grid="occupied corresponding 4x4 mask-tight cells, mean cosine; foreground_mean fallback")
    write_json(output/"run.json", info)
    encoded = {}
    for route in routes:
        rid = route["id"]
        encoded[rid] = encode_route(runtime, baseline/"data", model, route, selections[rid], compare_pooling=compare_pooling)
        print(f"Encoded {rid}", flush=True)
    del runtime
    results = evaluate_routes(model, encoded, args.temperature, compare_pooling=compare_pooling)
    questions = results if compare_pooling else {POOLING[0]: results}
    summaries = {}
    for method, rows in questions.items():
        folder = output
        if compare_pooling:
            folder = output/method
            folder.mkdir()
        summaries[method] = summary = summarize(rows)
        write_json(folder/"questions.json", rows)
        write_json(folder/"summary.json", summary)
        (folder/"report.md").write_text(report(summary, args.temperature), encoding="utf-8")
        for route, stats in summary.items():
            c, n = stats["correct"], stats["n"]
            print(f"{method} {route}: view {sum(stats['view_hits'].values())}/{n}, hard {c['hard']}/{n}, "
                  f"soft {c['soft']}/{n}, gallery {c['gallery']}/{n}, oracle {c['oracle']}/{n}", flush=True)
    if compare_pooling:
        write_json(output/"summary.json", summaries)
        (output/"report.md").write_text(pooling_report(summaries, questions), encoding="utf-8")
    info.update(status="complete", questions=len(questions[POOLING[0]]))
    write_json(output/"run.json", info)
    print(f"Report: {output/'report.md'}", flush=True)
    return output


if __name__ == "__main__":
    from dinoexp.experiment8_view_routing import DEFAULT_TEMPERATURE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("out/experiment8_long_6e8fu81r"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="new output directory")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--compare-pooling", action="store_true", help="compare global and two foreground descriptors")
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-repo")
    parser.add_argument("--backbone")
    run(parser.parse_args())
