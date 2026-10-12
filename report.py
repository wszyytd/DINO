"""Per-view scores and an image gallery; no confidence or exclusion claims."""

import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys

from dinoexp.core import checked_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="override moved dataset location")
    args = parser.parse_args()
    run = json.loads((args.results / "match_run.json").read_text(encoding="utf-8"))
    root = args.data or Path(run["data"])
    with (args.results / "scores.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty scores.csv")
    # Each sequence is a set of fixed views, not a flown or fused trajectory.
    groups = {}
    for row in rows:
        key = (row["identity"], row.get("position", ""), row.get("weather", ""))
        groups.setdefault(key, []).append(row)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for key, group in groups.items():
        group.sort(key=lambda r: int(r.get("sequence") or 0))
        x = [int(r.get("sequence") or 0) for r in group]
        label = "/".join(str(k) for k in key)
        axes[0].plot(x, [float(r["whole_score"]) for r in group], "o-", label=label)
        local = [r for r in group if r.get("local_score")]
        if local:
            axes[1].plot(
                [int(r.get("sequence") or 0) for r in local],
                [float(r["local_score"]) for r in local],
                "o-",
                label=label,
            )
    score_values = [float(r["whole_score"]) for r in rows]
    score_values += [float(r["local_score"]) for r in rows if r.get("local_score")]
    lower, upper = max(-1.05, min(score_values) - 0.05), min(1.05, max(score_values) + 0.05)
    for axis, title in zip(
        axes, ["Whole vehicle", "Manual local ROI (visible same part only)"], strict=True
    ):
        axis.set(
            title=title, xlabel="Observation order / fixed view index", ylabel="Cosine similarity"
        )
        axis.set_ylim(lower, upper)
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    if not any(r.get("local_score") for r in rows):
        axes[1].text(
            0.5,
            0.5,
            "Local comparison unavailable; annotate ROI first",
            ha="center",
            transform=axes[1].transAxes,
        )
    fig.savefig(args.results / "scores.png", dpi=160)
    plt.close(fig)
    assets = args.results / "gallery"
    assets.mkdir(exist_ok=True)

    def copied(row, index):
        source = checked_path(root, row["crop"])
        target = assets / f"{index:04d}.png"
        shutil.copyfile(source, target)
        return f"gallery/{target.name}"

    ref = run["reference"]
    ref_link = copied(ref, 0)
    cards = []
    for i, row in enumerate(rows, start=1):
        src = copied(row, i)
        details = {
            k: row.get(k, "")
            for k in [
                "identity",
                "view",
                "weather",
                "position",
                "feature_visibility",
                "whole_score",
                "local_score",
                "local_status",
            ]
        }
        cards.append(
            f'<article><img src="{src}"><b>{html.escape(row["sample_id"])}</b>'
            f"<pre>{html.escape(json.dumps(details, indent=2, ensure_ascii=False))}</pre></article>"
        )
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Vehicle matching probe</title><style>
body{{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}}
article{{border:1px solid #ddd;padding:12px}}img{{max-width:100%}}pre{{white-space:pre-wrap;font-size:12px}}
</style><h1>车辆辨识信号实验</h1>
<p>余弦相似度不是概率；低分不构成排除证据。视角顺序不是实际飞行轨迹，也没有做证据融合。</p>
<p>模型：{html.escape(run["model"])}；整车查询：{len(rows)}；局部可比：{run["local_compared"]}；
采集拒绝视图：{run["rejected_views"]}。未标注可见性应保持 unknown。</p>
<h2>参考图</h2><img src="{ref_link}" style="max-width:400px">
<h2>逐视角分数</h2><img src="scores.png">
<h2>图像与元数据</h2><div class="grid">{"".join(cards)}</div></html>'''
    (args.results / "report.html").write_text(document, encoding="utf-8")
    print(args.results / "report.html")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
