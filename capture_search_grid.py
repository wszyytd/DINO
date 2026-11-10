"""Collect complete CARLA search grids, derive offline walks and validate the dataset."""

import argparse
import json
from pathlib import Path
import sys

from capture_search_episodes import code_provenance as episode_code_provenance
from dinoexp.dense_cache import save_json
from dinoexp.grid_search_capture import capture_layout, grid_plan, init_collection
from dinoexp.grid_search_preview import publish_preview
from dinoexp.grid_search_validation import read_json, read_layout, save_routes, validate_collection
from dinoexp.local_pairs import sha256
from dinoexp.search_capture import preflight
from dinoexp.search_grid import action_spec

REPO = Path(__file__).resolve().parent
MANIFEST = 'dataset_manifest.json'


def code_provenance():
    code = episode_code_provenance()
    for name in ('capture_search_grid.py', 'dinoexp/search_grid.py', 'dinoexp/grid_search_capture.py',
                 'dinoexp/grid_search_validation.py', 'dinoexp/grid_search_preview.py', 'dinoexp/scene.py'):
        code['sources_sha256'][name] = sha256(REPO / name)
    return code


def new_collection(args, config, cases):
    import carla
    from capture import connect, find_blueprint

    _, world, versions = connect(carla, config['capture'])
    if versions[0] != versions[1]:
        raise RuntimeError(f'exact CARLA version match required: {versions}')
    state = preflight(world, config['map'])
    library = world.get_blueprint_library()
    for candidate in config['capture']['candidates']:
        bp = find_blueprint(library, candidate['blueprint'])
        if not bp.has_attribute('color') or not bp.get_attribute('color').is_modifiable:
            raise ValueError(f'{bp.id} does not have a modifiable color attribute')
    if args.preflight:
        return dict(**state, carla_versions=versions)
    root = args.output
    manifest = init_collection(root, config, cases, args.stage, code_provenance())
    try:
        for case in cases:
            capture_layout(carla, world, versions, config, root, case)
            scene, graph, frames = read_layout(root, config, case)
            save_routes(root, case, scene, graph, frames, args.stage)
        result = validate_collection(root, manifest, publishing=True)
        publish_preview(root, manifest, result)
        save_json(root / 'validation.json', result)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['failure'] = str(error)
        raise
    finally:
        save_json(root / MANIFEST, manifest)
    return result


def require_reviewable(manifest, result):
    if manifest['stage'] != 'smoke' or len(manifest['cases']) != 1 or result['offline_coverage_gaps']:
        raise ValueError(f'need one smoke grid with passing coverage: {result["offline_coverage_gaps"]}')
    for scene in result['scenes'].values():
        if any(asset['actor_color'] != asset['configured_color'] for asset in scene['assets']):
            raise ValueError('actor-reported color differs from config; investigate raw images and attributes')


def review_index(root):
    manifest = read_json(root / MANIFEST)
    files = [MANIFEST]
    for case in manifest['cases']:
        base = case['path']
        files.extend(f'{base}/{name}' for name in
                     ('scene.json', 'graph.json', 'frames.jsonl', 'audit/observations.jsonl',
                      'episodes/index.json'))
        for path in read_json(root / base / 'episodes/index.json')['paths']:
            files.extend(f'{path}/{name}' for name in ('episode.json', 'frames.jsonl', 'inputs.jsonl'))
    return {name: sha256(root / name) for name in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs' / 'search_grid_v1.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=('smoke', 'pilot'), default='smoke')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--validate', action='store_true')
    modes.add_argument('--record-review', metavar='NOTE',
                       help='after manual review, record state IDs and actual appearance/view findings')
    parser.add_argument('--reviewed-smoke', type=Path)
    args = parser.parse_args()
    if args.output:
        args.output = args.output.resolve()
        for old in ('grid_pilot_20261109', 'search_episodes_smoke_v1'):
            protected = (REPO / 'data' / old).resolve()
            if args.output.is_relative_to(protected) or protected.is_relative_to(args.output):
                parser.error('choose a separate grid dataset directory')
    if args.validate or args.record_review is not None:
        if args.output is None:
            parser.error('provide the dataset path via --output')
        manifest = read_json(args.output / MANIFEST)
        result = validate_collection(args.output, manifest)
        if args.record_review is not None:
            require_reviewable(manifest, result)
            note = args.record_review.strip()
            if not note:
                parser.error('provide visual findings and reviewed state IDs')
            with (args.output / 'visual_review.json').open('x', encoding='utf-8') as file:
                json.dump(dict(status='passed', index_sha256=review_index(args.output), note=note),
                          file, ensure_ascii=False, indent=2)
    else:
        config = read_json(args.config)
        cases = grid_plan(config, args.stage)
        if args.dry_run:
            count = sum((config['grid']['size']**2 - len(c['layout'].get('excluded_points', ()))) * 8
                        for c in cases)
            result = dict(captured=False, source_images_planned=count, cases=cases,
                          derived_sequences_requested=8 if args.stage == 'smoke' else 4*len(cases),
                          action_spec=action_spec(config['grid']))
        else:
            if not args.preflight:
                if args.output is None or args.output.exists():
                    parser.error('capture requires a new output directory')
                if args.stage == 'pilot':
                    if args.reviewed_smoke is None:
                        parser.error('pilot requires a visually reviewed smoke dataset')
                    smoke = args.reviewed_smoke.resolve()
                    manifest = read_json(smoke / MANIFEST)
                    require_reviewable(manifest, validate_collection(smoke, manifest))
                    review = read_json(smoke / 'visual_review.json')
                    if (review['status'] != 'passed' or review['index_sha256'] != review_index(smoke)
                        or manifest['config'] != config or manifest['code'] != code_provenance()):
                        parser.error('pilot requires a visual review of the same config and code')
            result = new_collection(args, config, cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
