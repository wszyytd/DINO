"""Plan, collect, resume or validate the reference-guided CARLA episode pilot."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dinoexp.dense_cache import save_json
from dinoexp.local_pairs import sha256
from dinoexp.scene import validate_scene
from dinoexp.search_capture import capture_collection, preflight, write_preview
from dinoexp.search_episodes import make_plan, smoke_coverage_gaps, validate_dataset, validate_splits


REPO = Path(__file__).resolve().parent
OLD = REPO / 'data' / 'grid_pilot_20261109'


def code_provenance():
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True)
    files = ('capture_search_episodes.py', 'capture.py', 'dinoexp/search_episodes.py',
             'dinoexp/search_capture.py', 'dinoexp/dense_capture.py', 'dinoexp/core.py')
    return dict(git_head=revision.stdout.strip() if revision.returncode == 0 else None,
                sources_sha256={name: sha256(REPO / name) for name in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs' / 'search_episode_pilot.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=('smoke', 'pilot'), default='smoke')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--validate', action='store_true')
    modes.add_argument('--resume', action='store_true')
    modes.add_argument('--record-smoke-review', metavar='NOTE',
                       help='record a successful visual review after inspecting the real observations')
    parser.add_argument('--reviewed-smoke', type=Path,
                        help='verified smoke dataset with a recorded visual review')
    parser.add_argument('--resume-after-code-update', action='store_true',
                        help='resume a compatible capture repair and retain per-episode code provenance')
    args = parser.parse_args()
    if args.resume_after_code_update and not args.resume:
        parser.error('--resume-after-code-update requires --resume')
    if args.output:
        args.output = args.output.resolve()
        if args.output.is_relative_to(OLD.resolve()) or OLD.resolve().is_relative_to(args.output):
            parser.error('choose a new output directory separate from the original grid pilot')
    if args.validate or args.record_smoke_review:
        if args.output is None or (args.output / '.capture.lock').exists():
            parser.error('validation requires an output directory with no active capture lock')
        result = validate_dataset(args.output)
        manifest_file = args.output / 'dataset_manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        cases = manifest['cases']
        write_preview(args.output, cases, result)
        if args.record_smoke_review:
            gaps = smoke_coverage_gaps(result, cases)
            if gaps or manifest['status'] != 'complete':
                parser.error('smoke cannot be marked reviewed: ' + '; '.join(gaps or ['incomplete']))
            save_json(args.output / 'visual_review.json', dict(
                status='passed', episodes=sorted(result['episodes']),
                manifest_sha256=sha256(manifest_file), note=args.record_smoke_review,
            ))
            with (args.output / 'validation_report.md').open('a', encoding='utf-8') as report:
                report.write('\nVisual review recorded: ' + args.record_smoke_review + '\n')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result['errors'] else 2

    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    for layout in cfg['layouts']:
        validate_scene(dict(candidates=cfg['capture']['candidates'], target_identity=layout['target']))
    cases = make_plan(cfg, args.stage)
    for case in cases:
        case['path'] = f'scenes/{case["scene_id"]}/episodes/{case["episode_id"]}'
    validate_splits(cases)
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, seed=cfg['seed'], planned_episodes=len(cases),
                              planned_observations=sum(len(case['actions']) + 1 for case in cases),
                              cases=cases, captured=False), indent=2))
        return 0
    if not args.preflight and args.output is None:
        parser.error('capture requires --output')
    if not args.preflight and not args.resume and args.output.exists():
        parser.error('output already exists; choose a new directory or use --resume')
    if args.stage == 'pilot' and not args.preflight:
        if args.reviewed_smoke is None:
            parser.error('pilot expansion requires --reviewed-smoke')
        smoke = args.reviewed_smoke.resolve()
        if smoke.is_relative_to(OLD.resolve()) or (smoke / '.capture.lock').exists():
            parser.error('invalid or busy smoke dataset')
        report = validate_dataset(smoke)
        review = json.loads((smoke / 'visual_review.json').read_text(encoding='utf-8'))
        smoke_manifest = json.loads((smoke / 'dataset_manifest.json').read_text(encoding='utf-8'))
        if (smoke_coverage_gaps(report, smoke_manifest['cases']) or review['status'] != 'passed' or
            set(review['episodes']) != set(report['episodes']) or
            review['manifest_sha256'] != sha256(smoke / 'dataset_manifest.json') or
            smoke_manifest['config'] != cfg or smoke_manifest['code'] != code_provenance()):
            parser.error('smoke dataset requires a passing review of the same config and code')

    import carla
    from capture import connect
    client, world, versions = connect(carla, cfg['capture'])
    if versions[0] != versions[1]:
        raise RuntimeError(f'exact CARLA version match required: {versions}')
    state = preflight(world, cfg['map'])
    if args.preflight:
        print(json.dumps(dict(versions=versions, **state), indent=2))
        return 0
    result = capture_collection(carla, client, world, versions, cfg, args.output, cases, code_provenance(),
                                args.resume, args.resume_after_code_update)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ImportError, ValueError, KeyError, OSError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(2)
