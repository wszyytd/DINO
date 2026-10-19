"""Run capture -> part segmentation -> evidence; stop immediately if any stage fails."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def make_plan(root, name, weights=None, width=2048, height=1536,
              model_repo=None, checkpoint=None, take_tick=False):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('run name must contain only letters, digits, underscore or hyphen')
    if width <= 0 or height <= 0 or width * 3 != height * 4:
        raise ValueError('positive 4:3 native resolution required')
    if checkpoint and not model_repo:
        raise ValueError('--checkpoint requires --model-repo')
    outputs = [root / 'data' / name, root / 'out' / (name + '-parts'),
               root / 'out' / (name + '-evidence')]
    for path in outputs:
        if path.exists():
            raise ValueError(f'output already exists: {path}; omit --name for a fresh run')
    weights = Path(weights) if weights else root / 'models/vehicle_parts_local/parts_segmentation.pt'
    if not weights.is_absolute():
        weights = root / weights

    def command(environment, script, *args):
        return ['conda', 'run', '--no-capture-output', '-n', environment,
                'python', '-u', str(root / script), *map(str, args)]

    capture = command('carla10', 'capture_suite.py',
                      '--suite', root / 'configs/model_suite_no_poles.json',
                      '--width', width, '--height', height,
                      '--reference-source', root / 'data/model_suite02/charger_variant_Aleft',
                      '--output', outputs[0])
    if take_tick:
        capture.append('--take-tick-ownership')
    segment = command('dino-probe', 'segment_parts.py', '--data', outputs[0],
                      '--output', outputs[1], '--weights', weights, '--device', 'cpu')
    evidence = command('dino-probe', 'evidence.py', '--parts-run', outputs[1],
                       '--output', outputs[2], '--compare', '--device', 'cpu')
    if model_repo:
        repo = Path(model_repo)
        evidence += ['--model-repo', str(repo if repo.is_absolute() else root / repo)]
    if checkpoint:
        path = Path(checkpoint)
        evidence += ['--checkpoint', str(path if path.is_absolute() else root / path)]
    return [dict(stage=stage, command=cmd, output=str(output), cwd=str(root))
            for stage, cmd, output in zip(['capture', 'segment', 'evidence'],
                                          [capture, segment, evidence], outputs)]


def run_plan(plan, executor=None):
    executor = executor or subprocess.run
    for step in plan:
        print(f"[{step['stage']}] {shlex.join(step['command'])}", flush=True)
        try:
            executor(step['command'], cwd=step.get('cwd'), check=True)
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(
                f"Stage {step['stage']} failed: {error}. Later stages were not run. "
                "Fix the reported error; retry without --name to use fresh output directories."
            ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--name', help='optional unique run ID; default is a UTC timestamp')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--height', type=int, default=1536)
    parser.add_argument('--model-repo', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--take-tick-ownership', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print plan; no models, conda or CARLA')
    args = parser.parse_args()
    name = args.name or datetime.now(timezone.utc).strftime('probe_%Y%m%d_%H%M%S_%f')
    plan = make_plan(args.root, name, args.weights, args.width, args.height,
                     args.model_repo, args.checkpoint, args.take_tick_ownership)
    print(f'Run: {name}', flush=True)
    for step in plan:
        print(f"{step['stage']} output: {step['output']}", flush=True)
    if args.dry_run:
        for step in plan:
            print(shlex.join(step['command']))
        return
    if not shutil.which('conda'):
        raise ValueError('conda executable not found in PATH; run from your conda terminal')
    weight_path = Path(plan[1]['command'][plan[1]['command'].index('--weights') + 1])
    if not weight_path.is_file():
        raise ValueError(f'vehicle-part weights not found: {weight_path}')
    run_plan(plan)
    print(f"Completed. Download this evidence folder: {plan[-1]['output']}")


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
