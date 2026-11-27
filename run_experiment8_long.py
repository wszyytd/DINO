"""Train the ordinary cutout Writer once and evaluate at epochs 30, 50, 75 and 100."""

import argparse
import os
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

from dinoexp.region_ab_data import read_json, sha, write_json


MILESTONES = (30, 50, 75, 100)
SEED = 7
VARIANT = "replay_prepost"
BASE = Path("out/experiment8_cutouts_20261127_084108_Cmx6Mi")
HYPERPARAMETERS = ("lr", "weight_decay", "batch_routes", "independent_weight", "postwrite_weight")


def read_reference(root):
    data, old = root / "data", root / "foreground"
    config = read_json(old / "summary.json")
    manifest = read_json(data / "dataset_manifest.json")
    if (manifest["schema"] != "fixed_height_auto_cutouts_v1"
            or config["status"] != "complete" or config["crop_mode"] != "foreground"
            or config["epochs"] != 30 or VARIANT not in config["variants"]
            or config["plan_sha256"] != sha(data / "plan.json")
            or config["dataset_sha256"] != sha(data / "dataset_manifest.json")):
        raise ValueError("expected the completed 30-epoch automatic cutout run with its matching data")
    return config, read_json(data / "plan.json")


def report(summaries):
    lines = ["# 实验八普通 Writer 延长训练", "", "单图、分数平均、16 槽和 8 槽都使用同一模型和原方向查询。",
             "FR 表示先写车头再写车尾，RF 表示相反顺序。每行的分差为目标车分数减去另一辆车分数。",
             "", "| 轮数 | 训练更新次数 | 自然路线写后答对 | 同题写前 → 写后 | 改对 / 改错 |",
             "| ---: | ---: | ---: | --- | --- |"]
    for s in summaries:
        n, paired = s["natural"], s["paired_independent"]
        lines.append(f"| {s['epoch']} | {s['updates']} | {n['correct_after']}/{n['total']} | "
                     f"{paired['pre_correct']} → {paired['post_correct']} / {paired['total']} | "
                     f"{paired['help']} / {paired['harm']} |")
    check = summaries[0]["reproduction"]
    lines.extend(["", f"第 30 轮与原第 30 轮比较：旧 {check['reference_correct']}/{check['n']}，"
                  f"新 {check['current_correct']}/{check['n']}，逐题判定相同 {check['agree']}/{check['n']}，"
                  f"三个分数的最大绝对差 {check['max_abs_logit_diff']:.8g}。",
                  f"权重张量逐元素相同：{check['weights_identical']}。"])
    for s in summaries:
        lines.extend(["", f"## 第 {s['epoch']} 轮", "", "下面按同一检查点重算地点 42 的训练损失。训练过程中的各路线损失见 train.json。",
                      "", "| 训练路线 | 总损失 | 历史回问 CE | 写前 CE | 写后 CE |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for route, v in sorted(s["train_at_checkpoint"].items()):
            lines.append(f"| {route} | {v['loss']:.4f} | {v['main_ce']:.4f} | "
                         f"{v['independent_ce']:.4f} | {v['independent_post_ce']:.4f} |")
        lines.extend(["", "previous 为第一张单存，written 为第二张单存，proposal 为门控前，updated 为最终 8 槽。",
                      "score_mean 对两次单存的三个原始分数逐项取平均，gallery 用一个 Reader 读两张独立写出的 16 槽。",
                      "", "| 地点 64 车辆组 | 顺序 | 读法 | 车头答对 | 车头平均分差 | 车尾答对 | 车尾平均分差 | 两面都对 |",
                      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for v in s["views"]:
            f, r = v["front"], v["rear"]
            lines.append(f"| {v['group']} | {v['history']} | {v['method']} | {f['correct']}/{f['n']} | "
                         f"{f['margin_mean']:+.4f} | {r['correct']}/{r['n']} | {r['margin_mean']:+.4f} | "
                         f"{v['both_correct']}/{v['paired_n']} |")
        lines.extend(["", "近色 Lincoln B 车的两个留出机位，数值为目标车减对手的分差。",
                      "", "| 顺序 | 查询机位 | previous | written | proposal | updated |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"])
        for r in s["lincoln_b"]:
            margins = " | ".join(f"{r['scores'][k]['margin']:+.4f}" for k in ("previous", "written", "proposal", "updated"))
            lines.append(f"| {r['history']} | {r['camera']:02} | {margins} |")
        for g in s["lincoln_b_gates"]:
            values = ", ".join(f"{value:.3f}" for value in g["gate"])
            lines.append(f"\n{g['history']} 第二次写入时该车的逐槽门值：{values}。")
    return "\n".join(lines) + "\n"


def run(args):
    import torch
    from dinoexp.experiment8_long_eval import compare_replay, inspect_route, summarize_views
    from dinoexp.memory_replay import audit_plan
    from dinoexp.memory_replay_training import _fit, _validate, route_objective
    from dinoexp.memory_retention_eval import encode_route_features, frozen_runtime, load_checkpoints
    from dinoexp.replay_matcher import build_replay_matcher
    from dinoexp.symmetric_memory_eval import check_angles

    config, plan = read_reference(args.baseline)
    train, val, schedules, coverage = audit_plan(plan, config["dataset_sha256"])
    folder = args.baseline / "foreground" / f"{VARIANT}-s{SEED}"
    reference = read_json(folder / "validation.json")["rows"]
    record, architecture, initial = load_checkpoints(args.source_run, ["memory"], [SEED])[0]
    if record["checkpoint_sha256"] != config["source_checkpoints"][str(SEED)] or coverage != config["coverage"]:
        raise ValueError("initial weights or route coverage differ from the 30-epoch baseline")
    for route in val:
        check_angles(route, plan["max_query_angle"], plan["novelty_angle"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    runtime, source, backbone = frozen_runtime(args)
    if source["teacher_sha256"] != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("frozen encoders differ from the 30-epoch baseline")
    device = runtime.device
    if args.output is None:
        output_root = Path("out")
        output_root.mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="experiment8_long_", dir=output_root))
    else:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {output}", flush=True)
    fit_args = SimpleNamespace(epochs=MILESTONES[-1], **{key: config[key] for key in HYPERPARAMETERS})
    info = dict(status="running", baseline=str(args.baseline), variant=VARIANT, seed=SEED,
                milestones=list(MILESTONES), training=vars(fit_args), coverage=coverage,
                source_checkpoint_sha256=record["checkpoint_sha256"], plan_sha256=config["plan_sha256"],
                dataset_sha256=config["dataset_sha256"], teacher_sha256=config["teacher_sha256"],
                backbone_sha256=backbone, completed_epochs=[])
    write_json(output / "run.json", info)
    features = {}
    # Use no_grad, not inference_mode: cached features also feed autograd during training.
    with torch.no_grad():
        for route in plan["routes"]:
            extra = [k for k, s in route["samples"].items() if route["split"] == "val" and s["camera_index"] in (0, 12)]
            features[route["id"]] = encode_route_features(runtime, args.baseline / "data", route,
                                                         foreground=True, extra_writes=extra)
            print(f"Encoded {route['id']}", flush=True)
    del runtime
    model = build_replay_matcher(architecture).to(device)
    model.load_state_dict(initial, strict=True)
    summaries = []

    @torch.inference_mode()
    def checkpoint(net, log):
        epoch = log["epoch"]
        if epoch not in MILESTONES:
            return
        dst = output / f"epoch_{epoch:03}"
        dst.mkdir()
        temporary = dst / "last.pt.tmp"
        torch.save(dict(kind="memory_replay_matcher_v1", variant=VARIANT, seed=SEED, epoch=epoch,
                        config=net.config, model_state=net.state_dict(), updates=log["updates"], crop_mode="foreground",
                        plan_sha256=config["plan_sha256"], source_checkpoint_sha256=record["checkpoint_sha256"],
                        independent_weight=fit_args.independent_weight, postwrite_weight=fit_args.postwrite_weight), temporary)
        temporary.replace(dst / "last.pt")
        validation = _validate(net, val, features, plan["novelty_angle"])
        train_at_checkpoint = {}
        for route in train:
            key = route["id"]
            terms = route_objective(net, schedules[key], features[key], {}, VARIANT,
                                    fit_args.independent_weight, fit_args.postwrite_weight)
            fields = ("loss", "main_ce", "independent_ce", "independent_post_ce")
            train_at_checkpoint[key] = dict(zip(fields, torch.stack(terms).tolist(), strict=True))
        writes, questions = [], []
        for route in val:
            rows, gates = inspect_route(net, route, features[route["id"]])
            questions.extend(rows)
            writes.extend(gates)
        metrics = validation["aggregate"]
        result = dict(epoch=epoch, updates=log["updates"], train_at_checkpoint=train_at_checkpoint,
                      natural=metrics["groups"]["new_view"], paired_independent=metrics["paired_independent"],
                      views=summarize_views(questions),
                      lincoln_b_gates=[g for g in writes if g["route"].startswith("pair2_") and
                                      g["identity"].endswith(":B")],
                      lincoln_b=[r for r in questions if r["route"].startswith("pair2_") and
                                 r["identity"].endswith(":B") and r["camera"] in (11, 22)])
        if epoch == 30:
            check = compare_replay(reference, validation["rows"])
            check["weights_identical"] = None
            if (folder / "last.pt").is_file():
                old = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)["model_state"]
                state = net.state_dict()
                check["weights_identical"] = (old.keys() == state.keys() and
                                              all(torch.equal(old[k], state[k].cpu()) for k in old))
            result["reproduction"] = check
            print(f"Epoch 30 replay: {check}", flush=True)
        write_json(dst / "validation.json", validation)
        write_json(dst / "questions.json", questions)
        write_json(dst / "gates.json", writes)
        write_json(dst / "summary.json", result)
        summaries.append(result)
        temporary = output / "report.md.tmp"
        temporary.write_text(report(summaries), encoding="utf-8")
        temporary.replace(output / "report.md")
        info["completed_epochs"].append(epoch)
        write_json(output / "run.json", info)
        print(f"Epoch {epoch}: {result['natural']['correct_after']}/{result['natural']['total']} independent. "
              f"Report: {output / 'report.md'}", flush=True)

    _fit(model, VARIANT, fit_args, SEED, train, schedules, features, {}, output, after_epoch=checkpoint)
    info["status"] = "complete"
    write_json(output / "run.json", info)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE, help="completed automatic cutout run root")
    parser.add_argument("--output", type=Path, help="fresh output directory; default: unique directory under out")
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-repo")
    parser.add_argument("--backbone")
    run(parser.parse_args())
