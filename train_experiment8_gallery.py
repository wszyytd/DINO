"""Fine-tune only the Reader and match head of the epoch-100 ordinary Writer checkpoint."""

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
    from dinoexp.four_view_memory import VIEWS, encode_sequence, select_views
    from dinoexp.gallery_reader import (TRAIN_VIEWS, cache_route, evaluate, freeze_for_gallery, report,
                                       subset_schedule, train_epoch)
    from dinoexp.memory_retention_eval import frozen_runtime

    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    baseline, config, plan = experiment_source(args.checkpoints, args.baseline)
    train, val, _, _ = audit_plan(plan, config["dataset_sha256"])
    selections = {}
    for routes, split, group, views in ((train, "train", "spawn-42", TRAIN_VIEWS), (val, "val", "spawn-64", VIEWS)):
        for route in routes:
            if route["group"] != group:
                raise ValueError(f"unexpected {split} location: {route['group']}")
            selections[route["id"]] = select_views(route, plan["max_query_angle"], views=views, split=split)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    runtime, source, backbone = frozen_runtime(args)
    if source["teacher_sha256"] != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("feature encoders differ from the checkpoint experiment")
    folder = args.checkpoints / "epoch_100"
    model = load_writer(folder, config, SEED, runtime.device, epoch=100)
    freeze_for_gallery(model)
    if args.output is None:
        Path("out").mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="experiment8_gallery_reader_", dir="out"))
    else:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {output}", flush=True)
    trainable = [k for k, v in model.named_parameters() if v.requires_grad]
    info = dict(status="running", seed=SEED, source_epoch=100, source_checkpoint_sha256=sha(folder/"last.pt"),
                baseline=str(baseline), source_run=str(args.checkpoints), plan_sha256=config["plan_sha256"],
                dataset_sha256=config["dataset_sha256"], teacher_sha256=config["teacher_sha256"], backbone_sha256=backbone,
                epochs=args.epochs, batch_routes=2, lr=1e-4, weight_decay=.01, trainable=trainable,
                selections=selections, train_routes=[r["id"] for r in train], val_routes=[r["id"] for r in val])
    write_json(output/"run.json", info)
    caches = {}
    with torch.no_grad():
        for route in [*train, *val]:
            rid = route["id"]
            pairs = selections[rid]
            features = encode_sequence(runtime, baseline/"data", route, pairs)
            caches[rid] = cache_route(model, route, pairs, *features)
    del runtime
    training = {r["id"]: caches[r["id"]] for r in train}
    validation = {r["id"]: caches[r["id"]] for r in val}
    def save_stage(stage):
        rows = evaluate(model, validation)
        write_json(output/f"{stage}_questions.json", rows)
        return rows
    stages = {"before": save_stage("before")}
    (output/"report.md").write_text(report(stages), encoding="utf-8")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=info["lr"], weight_decay=info["weight_decay"])
    schedules = {rid: subset_schedule(len(TRAIN_VIEWS), args.epochs, SEED*1000+i)
                 for i, rid in enumerate(sorted(training))}
    logs = []
    updates = 0
    for epoch in range(args.epochs):
        subsets = {rid: schedule[epoch] for rid, schedule in schedules.items()}
        stats = train_epoch(model, optimizer, training, subsets, SEED*100_000+epoch)
        updates += stats["updates"]
        histories = {rid: [[TRAIN_VIEWS[i][1] for i in subset] for subset in groups]
                     for rid, groups in subsets.items()}
        logs.append(dict(**stats, epoch=epoch+1, total_updates=updates, write_cameras=histories))
        write_json(output/"train.json", logs)
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == args.epochs:
            print(f"Reader epoch {epoch+1}/{args.epochs}, updates {updates}, CE {stats['mean_ce']:.4f}", flush=True)

    stages["after"] = save_stage("after")
    torch.save(dict(kind="experiment8_gallery_reader_v1", config=model.config, model_state=model.state_dict(),
                    seed=SEED, source_checkpoint_sha256=info["source_checkpoint_sha256"],
                    plan_sha256=info["plan_sha256"], epoch=args.epochs, updates=updates), output/"last.pt")
    (output/"report.md").write_text(report(stages), encoding="utf-8")
    info.update(status="complete", updates=updates)
    write_json(output/"run.json", info)
    for route in val:
        rid = route["id"]
        hits = [sum(r["conditions"]["prefix_4"]["decision"] == "target" for r in rows if r["route"] == rid)
                for rows in stages.values()]
        print(f"{rid}: full gallery {hits[0]}/8 -> {hits[1]}/8", flush=True)
    print(f"Report: {output / 'report.md'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("out/experiment8_long_6e8fu81r"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="new directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-repo")
    parser.add_argument("--backbone")
    run(parser.parse_args())
