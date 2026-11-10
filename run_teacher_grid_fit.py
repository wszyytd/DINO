"""Fit the teacher-preset grid scorer on a single captured layout for diagnosis."""
import argparse
from pathlib import Path
import sys

import numpy as np

from dinoexp.core import checked_path
from dinoexp.dense_cache import DenseFeatureCache, extract_dense_features, save_json
from dinoexp.dense_data import make_targets, read_dense_manifest
from dinoexp.grid_search_validation import read_json, read_layout
from dinoexp.search_episodes import json_rows


DEFAULT_FIT = ('r00c00_d01', 'r01c01_d02', 'r02c03_d06', 'r00c00_d03')
DEFAULT_PROBE = ('r04c02_d04', 'r04c02_d01')
SCOPE = '单布局拟合诊断。留图不参与参数更新，与拟合图共享布局和参考。'
ROOT = Path(__file__).resolve().parent


def prepare(root, fit, probe):
    manifest = read_json(root/'dataset_manifest.json')
    if manifest['status'] != 'complete' or manifest['mode'] != 'discrete_view_graph' or len(manifest['cases']) != 1:
        raise ValueError('this diagnostic requires one complete grid layout')
    case, = manifest['cases']
    scene, _, frames = read_layout(root, manifest['config'], case)
    states = (*fit, *probe)
    if len(set(states)) != len(states) or set(states) - frames.keys():
        raise ValueError('fit and probe states must be distinct existing state IDs')
    directory = checked_path(root, case['path'])
    audit = {row['state_id']: row['rgb'] for row in json_rows(directory/'audit'/'observations.jsonl')}
    if {audit[sid]['file_sha256'] for sid in fit} & {audit[sid]['file_sha256'] for sid in probe}:
        raise ValueError('probe RGB duplicates a fitting image')
    references = scene['references']
    samples, visibility = [], {}
    for sid in states:
        frame = frames[sid]
        visibility[sid] = {uid.rsplit(':', 1)[-1]: label['quality']['visibility_present']
                           for uid, label in frame['objects'].items()}
        for uid, ref in references.items():
            mask = frame['objects'][uid]['mask']
            other, = (item['mask'] for key, item in frame['objects'].items() if key != uid)
            meta = audit[sid]
            samples.append(dict(
                id=sid+'_'+uid.rsplit(':', 1)[-1], split='train', group=case['group'],
                reference=ref['object'], image=frame['image_path'],
                reference_sha256=ref['hashes'][ref['object']], image_sha256=meta['file_sha256'],
                positive_masks=[mask['path']] if mask else [], negative_masks=[other['path']] if other else [],
                presence='present' if mask else 'absent', supervision='simulation',
                mask_source='carla_instance_verified',
                mask_sha256={item['path']: item['sha256'] for item in (mask, other) if item},
                camera_pose=meta['pose'], fov=meta['intrinsics']['horizontal_fov_deg']))
    return dict(format_version=1, task='reference_guided_dense_search', samples=samples), visibility


def fit_grid(args, manifest, visibility):
    import torch

    from dinoexp.dense_loss import dense_loss
    from dinoexp.dense_model import DenseSearchHead
    from dinoexp.dense_training import _metrics, _predict, sample_tensors, write_dense_report
    from dinoexp.local_features import resolve_device

    device = resolve_device(args.device, torch)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True)
    path = args.output/'manifest.json'
    save_json(path, manifest)
    data = read_dense_manifest(path, args.data_root)
    root, rows = data['_root'], data['samples']
    fit_count = 2*len(args.fit_states)
    fit_rows, probe_rows = rows[:fit_count], rows[fit_count:]
    run = dict(status='running', scope=SCOPE, epochs=args.epochs, lr=args.lr, seed=args.seed,
               device=str(device), fit_states=list(args.fit_states), probe_states=list(args.probe_states),
               visibility=visibility, fit_pairs=len(fit_rows), probe_pairs=len(probe_rows))
    save_json(args.output/'run.json', run)
    extract_dense_features(path, args.cache, data_root=root, model_repo=args.model_repo,
                           checkpoint=args.backbone, device=str(device))
    cache = DenseFeatureCache(args.cache)
    cache.preflight(rows, root)
    grid = cache.config['grid_size']
    labels = [make_targets(row, root, grid) for row in fit_rows]
    model = DenseSearchHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)

    def report(name):
        output, loss = _predict(model, rows, cache, root, device)
        both_absent = {sid for sid, seen in visibility.items() if not any(seen.values())}
        result = dict(scope=SCOPE, loss=loss, **_metrics(rows, output, root, grid),
                      fit=_metrics(fit_rows, output[:fit_count], root, grid),
                      probe=_metrics(probe_rows, output[fit_count:], root, grid),
                      both_absent_null={row['id']: float(p[-1])
                                        for row, p in zip(rows, output, strict=True)
                                        if row['id'].rsplit('_', 1)[0] in both_absent})
        write_dense_report(args.output/name, rows, output, root, grid, result)
        return result

    report('before')
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.
        for index in torch.randperm(len(fit_rows)).tolist():
            optimizer.zero_grad(set_to_none=True)
            out = model(**sample_tensors([fit_rows[index]], cache, device))
            loss = dense_loss(out['probabilities'], [labels[index]])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite grid fit loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        history.append(dict(epoch=epoch, fit_loss=total/len(fit_rows)))
        save_json(args.output/'history.json', history)
        print(f'Epoch {epoch}/{args.epochs} | fit loss {history[-1]["fit_loss"]:.6f}', flush=True)
        temp = args.output/'last.pt.tmp'
        torch.save(dict(kind='dense_reference_search_v1', stage='simulation', epoch=epoch,
                        feature_config=cache.config, model_config=model.config, model_state=model.state_dict(),
                        manifest_sha256=data['_manifest_sha256'],
                        training_images=sorted({row[key] for row in fit_rows for key in
                                                ('reference_sha256', 'image_sha256')}),
                        training_groups=sorted({row['group'] for row in fit_rows}), scope=SCOPE), temp)
        temp.replace(args.output/'last.pt')
    final = report('after')
    run.update(status='complete', after=final)
    save_json(args.output/'run.json', run)
    print('Final heatmaps:', args.output/'after/index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/search_grid_smoke_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache', type=Path, help='reuse frozen features for the exact same selection')
    parser.add_argument('--model-repo', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--fit-states', nargs='+', default=DEFAULT_FIT)
    parser.add_argument('--probe-states', nargs='+', default=DEFAULT_PROBE)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    if not args.dry_run:
        if args.output is None or args.output.exists():
            parser.error('choose a new --output directory')
        args.output = args.output.resolve()
        args.cache = (args.cache or args.output/'features').resolve()
        if any(p.is_relative_to(args.data_root) for p in (args.output, args.cache)):
            parser.error('output and cache must be separate from the captured grid')
    if args.epochs < 1 or args.threads < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, threads and learning rate must be positive')
    manifest, visibility = prepare(args.data_root, args.fit_states, args.probe_states)
    for name, states in [('fit', args.fit_states), ('probe', args.probe_states)]:
        for sid in states:
            print(name, sid, visibility[sid])
    if not args.dry_run:
        fit_grid(args, manifest, visibility)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
