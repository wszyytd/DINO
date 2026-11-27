"""Evaluate the four specified vehicle views using saved ordinary Writer checkpoints."""

import argparse
import os
from pathlib import Path
from tempfile import mkdtemp

from dinoexp.memory_replay import audit_plan
from dinoexp.region_ab_data import read_json, sha, write_json
from run_experiment8_followup import load_writer
from run_experiment8_long import MILESTONES, SEED, VARIANT, read_reference


def report(epochs):
    from dinoexp.four_view_memory import VIEWS
    lines = ["# 实验八四视角顺序写入", "",
             "两辆候选车同步写入 00 → 20 → 16 → 12。每步都问相同的 01、21、15、11 四个机位。",
             "updated 为递归 8 槽，written 为只存最新一张，score_mean 为已写各单图的三个分数逐项平均，"
             "slot_mean 为同编号的槽逐张取平均（每车 8 槽），gallery 为分存所有已写图的槽并用同一 Reader 查询，四步分别保存 8、16、24、32 槽。",
             "旧方向和新方向栏均为与上一步相比的“改对 / 改错（题数）”。第 1 步没有可比的先前记忆。"
             "四列方向成绩包含尚未写入的方向。四面都对以车辆为分母。"]
    for epoch, summary in epochs.items():
        lines.extend(["", f"## 第 {epoch} 轮权重", "",
                      "| 组 | 已写张数 | 读法 | 正车头 | 左前侧 | 左后侧 | 正车尾 | 总答对 | 四面都对 | 旧方向变化 | 新方向变化 |",
                      "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"])
        for r in summary:
            cells = [f"{r['views'][v]['correct']}/{r['views'][v]['n']}" for v, _, _ in VIEWS]
            deltas = [f"{d['helped']}/{d['harmed']} ({d['n']})" if d["n"] else "—"
                      for d in (r["old_changes"], r["new_changes"])]
            lines.append(f"| {r['group']} | {r['step']} | {r['method']} | " + " | ".join(cells) +
                         f" | {r['correct']}/{r['n']} | {r['all_right']}/{r['all_right_n']} | "
                         + " | ".join(deltas) + " |")
    return "\n".join(lines) + "\n"


def experiment_source(checkpoints, baseline):
    long_run = read_json(checkpoints / "run.json")
    baseline = baseline or Path(long_run["baseline"])
    config, plan = read_reference(baseline)
    if (long_run["status"] != "complete" or long_run["variant"] != VARIANT or long_run["seed"] != SEED
            or long_run["completed_epochs"] != list(MILESTONES)
            or long_run["source_checkpoint_sha256"] != config["source_checkpoints"][str(SEED)]
            or any(long_run[k] != config[k] for k in ("plan_sha256", "dataset_sha256", "teacher_sha256", "backbone_sha256"))):
        raise ValueError("long-run checkpoints do not match the completed foreground experiment")
    return baseline, config, plan


def run(args):
    import torch
    from dinoexp.four_view_memory import encode_sequence, inspect_sequence, select_views, summarize
    from dinoexp.memory_retention_eval import frozen_runtime

    baseline, config, plan = experiment_source(args.checkpoints, args.baseline)
    _, routes, _, _ = audit_plan(plan, config["dataset_sha256"])
    selections = {r["id"]: select_views(r, plan["max_query_angle"]) for r in routes}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    runtime, source, backbone = frozen_runtime(args)
    if source["teacher_sha256"] != config["teacher_sha256"] or backbone != config["backbone_sha256"]:
        raise ValueError("feature encoders differ from the checkpoint experiment")
    device = runtime.device
    if args.output is None:
        Path("out").mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="experiment8_four_views_", dir="out"))
    else:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {output}", flush=True)
    info = dict(status="running", seed=SEED, baseline=str(baseline), source_run=str(args.checkpoints),
                plan_sha256=config["plan_sha256"], selections=selections, checkpoints={})
    write_json(output / "run.json", info)
    with torch.inference_mode():
        features = {r["id"]: encode_sequence(runtime, baseline / "data", r, selections[r["id"]]) for r in routes}
    del runtime
    summaries = {}
    for epoch in MILESTONES:
        folder = args.checkpoints / f"epoch_{epoch:03}"
        model = load_writer(folder, config, SEED, device, epoch=epoch)
        questions, gates = [], []
        for r in routes:
            rows, events = inspect_sequence(model, r, selections[r["id"]], *features[r["id"]])
            questions.extend(rows)
            gates.extend(events)
        summaries[epoch] = summarize(questions)
        dst = output / folder.name
        dst.mkdir()
        write_json(dst / "questions.json", questions)
        write_json(dst / "gates.json", gates)
        write_json(dst / "summary.json", summaries[epoch])
        info["checkpoints"][str(epoch)] = sha(folder / "last.pt")
        tmp = output / "report.md.tmp"
        tmp.write_text(report(summaries), encoding="utf-8")
        tmp.replace(output / "report.md")
        write_json(output / "run.json", info)
        print(f"Epoch {epoch}: {len(questions)} paired question reads. See {output / 'report.md'}", flush=True)
        for row in summaries[epoch]:
            if row["group"] == "pair1+pair2" and row["method"] == "updated":
                view_counts = ", ".join(f"{name} {v['correct']}/{v['n']}" for name, v in row["views"].items())
                print(f"  近色车，已写 {row['step']} 张：{view_counts}，四面都对 "
                      f"{row['all_right']}/{row['all_right_n']}", flush=True)
    info["status"] = "complete"
    write_json(output / "run.json", info)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("out/experiment8_long_6e8fu81r"))
    parser.add_argument("--baseline", type=Path, help="override the automatic cutout dataset run location")
    parser.add_argument("--output", type=Path, help="fresh results directory")
    parser.add_argument("--source-run", type=Path, default=Path("out/region_sequence_mask_20261112_152736"))
    parser.add_argument("--teacher", type=Path, default=Path("out/teacher_grid_fit_e20_retry/last.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-repo")
    parser.add_argument("--backbone")
    run(parser.parse_args())
