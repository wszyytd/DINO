"""Collect randomized multi-background CARLA grids; all capture is explicitly requested."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from dinoexp.multiscene_capture import collect, read_json, validate_collection
from dinoexp.multiscene_plan import choose_backgrounds, dry_plan, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/multiscene_roundtrip_v1.json')
    parser.add_argument('--output')
    parser.add_argument('--stage', choices=('dry-run', 'preflight', 'capture', 'validate', 'preview'),
                        default='dry-run')
    parser.add_argument('--resume', action='store_true', help='same config/code/map, preserves old attempts')
    parser.add_argument('--size', type=int, default=480, help='preview/evaluation input short side')
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    if args.resume and args.stage != 'capture':
        parser.error('--resume is only valid for --stage capture')
    if args.stage == 'dry-run':
        print(json.dumps(dry_plan(config), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not args.output and args.stage != 'preflight':
        parser.error('--output is required for this stage')
    if args.stage == 'validate':
        print(json.dumps(validate_collection(args.output), ensure_ascii=False, indent=2))
        return
    if args.stage == 'preview':
        from dinoexp.multiscene_routes import prepare, render_report
        # Collection directory and sequence output are separate, never overwrite source data.
        preview_args = SimpleNamespace(data=args.output, output=str(Path(args.output)/'roundtrip_preview'),
                                       size=args.size)
        prepare(preview_args)
        render_report(preview_args, preview=True)
        return
    # Imported only for preflight/capture. No CARLA server connection on dry-run/validate/preview.
    import carla
    from capture import connect
    from dinoexp.core import pose_values
    from dinoexp.search_capture import preflight
    _, world, versions = connect(carla, config['capture'])
    if len(versions) != 2 or versions[0] != versions[1]:
        raise RuntimeError('exact CARLA client/server version agreement required')
    state = preflight(world, config['map'])
    backgrounds = choose_backgrounds(config, [list(pose_values(p)) for p in world.get_map().get_spawn_points()])
    for item in config['capture']['candidates']:
        world.get_blueprint_library().find(item['blueprint'])
    print(json.dumps(dict(preflight=state, versions=versions, backgrounds=backgrounds),
                     ensure_ascii=False, indent=2))
    if args.stage == 'preflight':
        return
    manifest, report = collect(config, args.output, carla, world, versions, resume=args.resume)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if manifest['collection_status'] == 'partial':
        raise SystemExit(2)  # Accepted subset is published; requested/exhausted counts stay explicit.


if __name__ == '__main__':
    main()
