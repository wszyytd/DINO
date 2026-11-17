"""Audit or fine-tune vehicle memory with all previously seen photos."""

import argparse
import json
from pathlib import Path

from dinoexp.memory_replay import audit_plan
from dinoexp.region_ab_data import read_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    audit = commands.add_parser("audit", help="check query counts and split isolation without CUDA")
    train = commands.add_parser("train", help="fine-tune and evaluate on CUDA")
    for command in (audit, train):
        command.add_argument("--plan", type=Path, required=True, help="memory retention plan.json")
        command.add_argument("--data", type=Path, default=Path("data/multiscene_roundtrip_20261116_v1"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--variant", nargs="+", choices=("fixed", "replay", "replay_independent"),
                       default=["fixed", "replay", "replay_independent"])
    train.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    train.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    train.add_argument("--model-repo")
    train.add_argument("--backbone")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, nargs="+", default=[7])
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-routes", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=.01)
    train.add_argument("--independent-weight", type=float, default=1.)
    args = parser.parse_args()
    if args.stage == "audit":
        plan = read_json(args.plan)
        _, _, _, coverage = audit_plan(plan, sha(args.data / "dataset_manifest.json"))
        result = dict(coverage=coverage, excluded=plan["excluded"])
    else:
        from dinoexp.memory_replay_training import run
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
