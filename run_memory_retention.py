"""Diagnose what each vehicle-memory update adds, preserves or forgets."""

import argparse
import json
from pathlib import Path

from dinoexp.memory_retention import build_plan
from dinoexp.region_ab_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    plan = stages.add_parser("plan", help="validate capture files and choose probes without a model")
    evaluate = stages.add_parser("evaluate", help="encode and score with existing CUDA checkpoints")
    for stage in (plan, evaluate):
        stage.add_argument("--data", type=Path, default=Path("data/multiscene_roundtrip_20261116_v1"))
        stage.add_argument("--output", type=Path, required=True)
    plan.add_argument("--split", choices=("all", "train", "val"), default="all")
    plan.add_argument("--allow-incomplete", action="store_true", help="list and skip incomplete layouts")
    plan.add_argument("--min-visible-pixels", type=int, default=8192)
    plan.add_argument("--max-query-angle", type=float, default=25.0)
    plan.add_argument("--novelty-angle", type=float, default=45.0)
    evaluate.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    evaluate.add_argument("--seed", type=int, nargs="+", default=[7, 17, 27])
    evaluate.add_argument("--method", choices=("memory", "last", "mean"), nargs="+", default=["memory"])
    evaluate.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    evaluate.add_argument("--model-repo")
    evaluate.add_argument("--backbone")
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "plan":
        if args.output.exists():
            parser.error("output exists; choose a new directory")
        result = build_plan(args.data, args.split, args.allow_incomplete, args.min_visible_pixels,
                            args.max_query_angle, args.novelty_angle)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", result)
        result = {key: result[key] for key in ("coverage", "excluded")}
    else:
        from dinoexp.memory_retention_eval import run
        result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
