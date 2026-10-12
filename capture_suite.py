"""Controlled two-candidate conditions, paired position swaps and a shared reference."""

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

from capture_scene import preview
from dinoexp.core import checked_path
from dinoexp.dataset import load_dataset
from dinoexp.suite import build_cases, reuse_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/scene_suite.json"))
    parser.add_argument("--output", type=Path, default=Path("data/suite01"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--take-tick-ownership", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.suite.read_text(encoding="utf-8-sig"))
    base = json.loads(
        checked_path(args.suite.parent, spec["base_config"]).read_text(encoding="utf-8-sig")
    )
    cases = build_cases(base, spec)
    if args.dry_run:
        for name, cfg in cases:
            print(name, [(c["identity"], c["color"]) for c in cfg["candidates"]])
        print(
            f"{len(cases)} conditions, {len(cases) * len(base['views'])} shared observations; 1 common reference"
        )
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "_configs").mkdir()
    manifest = dict(
        status="incomplete",
        conditions=[],
        shared_reference=cases[0][0],
        note="same map/location/weather, controlled conditions; not independent scene trials",
    )
    cards = []
    try:
        for name, cfg in cases:
            config = root / "_configs" / f"{name}.json"
            config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            destination = root / name
            command = [
                sys.executable,
                str(Path(__file__).with_name("capture_scene.py").resolve()),
                "--config",
                str(config),
                "--output",
                str(destination),
            ]
            if args.take_tick_ownership:
                command.append("--take-tick-ownership")
            print(f"Capturing condition: {name}", flush=True)
            subprocess.run(command, check=True)
            reuse_reference(root / cases[0][0], destination)
            ref, queries, rejected = load_dataset(destination)
            if rejected:
                raise ValueError(f"{name} contains rejected views; inspect before continuing")
            preview(destination, [ref] + queries, cfg["target_identity"])
            manifest["conditions"].append(dict(name=name, status="complete", config=cfg))
            escaped = html.escape(name)
            cards.append(
                f"<section><h2>{escaped}</h2><p>{html.escape(str([(c['identity'], c['blueprint'], c['color']) for c in cfg['candidates']]))}</p>"
                f'<a href="{escaped}/preview.html"><img width="900" src="{escaped}/images/overview.png"></a>'
                f'<p><a href="{escaped}/preview.html">查看参考图、各视点和候选裁剪</a></p></section>'
            )
            (root / "suite_preview.html").write_text(
                '<!doctype html><meta charset="utf-8"><title>场景对照预览</title>'
                "<style>body{max-width:1050px;margin:30px auto;font:16px system-ui}img{max-width:100%}</style>"
                f"<h1>{html.escape(spec.get('title', '两候选同场：颜色差异与位置交换'))}</h1>"
                f"<p>{html.escape(spec.get('description', '每组 A 始终是白色目标；共用第一组的目标参考图。'))}</p>"
                f"<p>同地点的 {len(cases)} 个对照条件，不是独立场景测试。先检查真实外观与遮挡。</p>"
                + "".join(cards),
                encoding="utf-8",
            )
            (root / "suite.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["error"] = str(error)
        raise
    finally:
        (root / "suite.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {root / 'suite_preview.html'}")


if __name__ == "__main__":
    main()
