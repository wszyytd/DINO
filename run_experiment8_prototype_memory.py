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
        AFTER_FOUR_DIRECTIONS, CAPACITY, METHODS, SEQUENCES, calibrate, encode_stream, replay, report,
        select_stream, summarize,
    )
    from dinoexp.reader_prototype_memory import (
        METHODS as READER_METHODS,
        calibrate_reader, classification, compare_observations, replay_reader, report_reader,
    )

    reader_dedup = getattr(args, "reader_dedup", False)
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
        prefix = "experiment8_reader_prototype_" if reader_dedup else "experiment8_prototype_memory_"
        output = Path(mkdtemp(prefix=prefix, dir="out"))
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
    if reader_dedup:
        info.update(methods=READER_METHODS, reader_dedup_score="minimum of the two directed identity logits",
                    write_reference="cached per-image reference features used for the reverse direction")
    write_json(output/"run.json", info)
    root = baseline/"data"
    max_angle = plan["max_query_angle"]
    if reader_dedup:
        fitted = calibrate_reader(runtime, root, model, train, max_angle)
    else:
        fitted = calibrate(runtime, root, train, max_angle)
    write_json(output/"calibration.json", fitted)
    fits = fitted["thresholds"] if reader_dedup else {"grid": fitted}
    for kind, fit in fits.items():
        print(f"Location 42 {kind} threshold {fit['threshold']:.5f}, "
              f"duplicates {fit['duplicate_correct']}/{fit['duplicate_n']}, "
              f"novel {fit['novel_correct']}/{fit['novel_n']}", flush=True)
    questions, events, held_out_pairs = [], [], []
    for route in val:
        rid = route["id"]
        writes, probes = streams[rid]
        if reader_dedup:
            write_references = {}
            encoded = encode_stream(runtime, root, model, route, writes, probes, write_references=write_references)
            with torch.inference_mode():
                directed, pairs = compare_observations(model, route, encoded[0], write_references, max_angle)
            held_out_pairs.extend(pairs)
            rows, writes = replay_reader(model, route, probes, encoded, directed, fits)
        else:
            encoded = encode_stream(runtime, root, model, route, writes, probes)
            rows, writes = replay(model, route, probes, encoded, fitted["threshold"])
        questions.extend(rows)
        events.extend(writes)
        print(f"Evaluated {rid}: {len(writes)} writes, {len(rows)} paired reads across both sequences", flush=True)
    summary = summarize(questions, events, methods=READER_METHODS if reader_dedup else METHODS)
    write_json(output/"questions.json", questions)
    write_json(output/"writes.json", events)
    write_json(output/"summary.json", summary)
    if reader_dedup:
        held_out = classification(held_out_pairs, fitted)
        write_json(output/"held_out_pairs.json", dict(metrics=held_out, pairs=held_out_pairs))
        text = report_reader(summary, fitted, held_out)
        info["thresholds"] = fits
    else:
        text = report(summary, fitted)
        info["threshold"] = fitted["threshold"]
    (output/"report.md").write_text(text, encoding="utf-8")
    info.update(status="complete", paired_reads=len(questions), write_events=len(events))
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
    parser.add_argument("--reader-dedup", action="store_true", help="compare max-score reads and bidirectional Reader deduplication")
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-repo")
    parser.add_argument("--backbone")
    run(parser.parse_args())
