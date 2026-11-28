"""Run experiment 8's no-training fixed-capacity view memory evaluation."""

import argparse
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
    from dinoexp.memory_retention_eval import frozen_runtime
    from dinoexp.prototype_memory import (
        AFTER_FOUR_DIRECTIONS, CAPACITY, SEQUENCES, calibrate, encode_stream, replay, report, select_stream, summarize,
    )

    baseline, config, plan = experiment_source(args.checkpoints, args.baseline)
    train, val, _, _ = audit_plan(plan, config["dataset_sha256"])
    streams = {r["id"]: select_stream(r, plan["max_query_angle"]) for r in val}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    runtime, source, backbone = frozen_runtime(args)
    if source["teacher_sha256"] != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("feature encoders differ from the checkpoint experiment")
    checkpoint = args.checkpoints / "epoch_100"
    model = load_writer(checkpoint, config, SEED, runtime.device, epoch=100)
    if args.output is None:
        Path("out").mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="experiment8_prototype_memory_", dir="out"))
    else:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {output}", flush=True)
    info = dict(status="running", capacity=CAPACITY, source_epoch=100, seed=SEED, baseline=str(baseline),
                source_run=str(args.checkpoints), plan_sha256=config["plan_sha256"], dataset_sha256=config["dataset_sha256"],
                teacher_sha256=config["teacher_sha256"], backbone_sha256=backbone,
                checkpoint_sha256=sha(checkpoint/"last.pt"), descriptor_layer=runtime.cfg["layers"][-1],
                sequences=SEQUENCES, after_four_directions=AFTER_FOUR_DIRECTIONS, streams=streams,
                calibration_routes=[r["id"] for r in train])
    write_json(output/"run.json", info)
    root = baseline/"data"
    fitted = calibrate(runtime, root, train, plan["max_query_angle"])
    write_json(output/"calibration.json", fitted)
    print(f"Location 42 calibration: threshold {fitted['threshold']:.5f}, "
          f"duplicates {fitted['duplicate_correct']}/{fitted['duplicate_n']}, "
          f"novel {fitted['novel_correct']}/{fitted['novel_n']}", flush=True)
    questions, events = [], []
    for route in val:
        rid = route["id"]
        writes, probes = streams[rid]
        encoded = encode_stream(runtime, root, model, route, writes, probes)
        rows, writes = replay(model, route, probes, encoded, fitted["threshold"])
        questions.extend(rows)
        events.extend(writes)
        print(f"Evaluated {rid}: {len(writes)} writes, {len(rows)} paired reads across both sequences", flush=True)
    summary = summarize(questions, events)
    write_json(output/"questions.json", questions)
    write_json(output/"writes.json", events)
    write_json(output/"summary.json", summary)
    (output/"report.md").write_text(report(summary, fitted), encoding="utf-8")
    info.update(status="complete", paired_reads=len(questions), write_events=len(events), threshold=fitted["threshold"])
    write_json(output/"run.json", info)
    for row in summary:
        if row["route"].startswith("pair2_"):
            print(f"{row['sequence']} {row['method']}: {row['after_four']}/{row['questions']} -> "
                  f"{row['final']}/{row['questions']}, lost {row['lost_final']}, "
                  f"rejected {row['decisions'].get('rejected_full', 0)}", flush=True)
    print(f"Report: {output/'report.md'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("out/experiment8_long_6e8fu81r"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="new output directory")
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-repo")
    parser.add_argument("--backbone")
    run(parser.parse_args())
