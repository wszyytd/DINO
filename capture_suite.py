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
from dinoexp.scene import with_resolution


def check_connection(config):
    """Read-only preflight before any suite output directory is created."""
    try:
        import carla
        from capture import connect

        _, world, versions = connect(carla, config)
        print(f"[preflight] CARLA {versions}; map: {world.get_map().name}", flush=True)
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError(
            f"CARLA preflight failed at {config['host']}:{config['port']}: {error}. "
            "Start the CARLA simulator, wait for its world to load, then retry. "
            "Activating a conda environment does not start CARLA. No suite output was created."
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/scene_suite.json"))
    parser.add_argument("--output", type=Path, default=Path("data/suite01"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-connection", action="store_true",
                        help="read-only CARLA connection check; no capture or output")
    parser.add_argument("--take-tick-ownership", action="store_true")
    parser.add_argument("--width", type=int, help="native RGB sensor width; requires --height")
    parser.add_argument("--height", type=int, help="native RGB sensor height; keeps aspect ratio")
    parser.add_argument(
        "--reference-source", type=Path, help="reuse reference from an existing captured scene"
    )
    args = parser.parse_args()
    if args.dry_run and args.check_connection:
        raise ValueError("choose --dry-run or --check-connection")
    spec = json.loads(args.suite.read_text(encoding="utf-8-sig"))
    base = json.loads(
        checked_path(args.suite.parent, spec["base_config"]).read_text(encoding="utf-8-sig")
    )
    base = with_resolution(base, args.width, args.height)
    cases = build_cases(base, spec)
    if args.dry_run:
        print(f"Native camera: {base['width']}x{base['height']}; FOV: {base['fov']}")
        for name, cfg in cases:
            print(name, [(c["identity"], c["color"]) for c in cfg["candidates"]])
        print(
            f"{len(cases)} conditions, {len(cases) * len(base['views'])} shared observations; 1 common reference"
        )
        return
    if args.check_connection:
        check_connection(base)
        return
    reference_source = args.reference_source.resolve() if args.reference_source else None
    if reference_source:
        load_dataset(reference_source)
    root = args.output.resolve()
    if root.exists():
        raise ValueError(f"output already exists: {root}; choose a new --output, do not overwrite")
    check_connection(base)
    root.mkdir(parents=True, exist_ok=False)
    (root / "_configs").mkdir()
    manifest = dict(
        status="incomplete",
        conditions=[],
        shared_reference=str(reference_source) if reference_source else cases[0][0],
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
            reuse_reference(reference_source or root / cases[0][0], destination)
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
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
