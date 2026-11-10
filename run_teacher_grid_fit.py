"""Fit the teacher-preset grid scorer on a single captured layout for diagnosis."""
import argparse
from pathlib import Path
import sys

import numpy as np

from dinoexp.core import checked_path
from dinoexp.dense_cache import DenseFeatureCache, extract_dense_features, save_json
from dinoexp.dense_data import make_targets, read_dense_manifest
from dinoexp.grid_search_validation import read_json, read_layout
from dinoexp.local_pairs import sha256
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
    from dinoexp.dense_training import (_metrics, _predict, inspect_checkpoint, sample_tensors,
                                        write_dense_report)
    from dinoexp.local_features import resolve_device

    device = resolve_device(args.device, torch)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    initial_manifest = args.init_checkpoint.with_name('manifest.json') if args.init_checkpoint else None
    if initial_manifest and initial_manifest.is_file() and read_json(initial_manifest) != manifest:
        raise ValueError('checkpoint manifest does not match the selected grid pairs')
    args.output.mkdir(parents=True)
    path = args.output/'manifest.json'
    if initial_manifest and initial_manifest.is_file():
        path.write_bytes(initial_manifest.read_bytes())
    else:
        save_json(path, manifest)
    data = read_dense_manifest(path, args.data_root)
    root, rows = data['_root'], data['samples']
    fit_count = 2*len(args.fit_states)
    fit_rows, probe_rows = rows[:fit_count], rows[fit_count:]
    run = dict(status='running', scope=SCOPE, epochs=args.epochs, lr=args.lr, seed=args.seed,
               patience=args.patience, eval_every=args.eval_every, min_delta=args.min_delta,
               device=str(device), fit_states=list(args.fit_states), probe_states=list(args.probe_states),
               visibility=visibility, fit_pairs=len(fit_rows), probe_pairs=len(probe_rows))
    save_json(args.output/'run.json', run)
    extract_dense_features(path, args.cache, data_root=root, model_repo=args.model_repo,
                           checkpoint=args.backbone, device=str(device))
    cache = DenseFeatureCache(args.cache)
    cache.preflight(rows, root)
    grid = cache.config['grid_size']
    labels = [make_targets(row, root, grid) for row in fit_rows]
    model = DenseSearchHead()
    provenance = dict(
        kind='dense_reference_search_v1', stage='simulation', feature_config=cache.config,
        model_config=model.config, manifest_sha256=data['_manifest_sha256'],
        training_images=sorted({row[key] for row in fit_rows for key in
                                ('reference_sha256', 'image_sha256')}),
        training_groups=sorted({row['group'] for row in fit_rows}), scope=SCOPE)
    start_epoch = 0
    if args.init_checkpoint:
        model, checkpoint = inspect_checkpoint(args.init_checkpoint, cache.config, model.config)
        if any(checkpoint.get(key) != provenance[key] for key in
               ('stage', 'manifest_sha256', 'training_images', 'training_groups')):
            raise ValueError('checkpoint must come from the same simulation fit samples and manifest')
        start_epoch = checkpoint['epoch']
        if type(start_epoch) is not int or start_epoch < 0:
            raise ValueError('checkpoint must record a nonnegative epoch')
        run['initial_checkpoint'] = dict(path=str(args.init_checkpoint),
                                         sha256=sha256(args.init_checkpoint), epoch=start_epoch,
                                         optimizer='new_adamw')
        del checkpoint
        print(f'Loaded weights at epoch {start_epoch}; starting a new AdamW optimizer.', flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    save_json(args.output/'run.json', run)

    def evaluate():
        fit_scores, fit_loss = _predict(model, fit_rows, cache, root, device)
        probe_scores, probe_loss = _predict(model, probe_rows, cache, root, device)
        if not np.isfinite([fit_loss, probe_loss]).all():
            raise RuntimeError('nonfinite evaluation loss')
        return fit_scores + probe_scores, fit_loss, probe_loss

    def report(name, evaluation):
        output, fit_loss, probe_loss = evaluation
        loss = (fit_loss*len(fit_rows) + probe_loss*len(probe_rows))/len(rows)
        both_absent = {sid for sid, seen in visibility.items() if not any(seen.values())}
        result = dict(scope=SCOPE, loss=loss, **_metrics(rows, output, root, grid),
                      fit=dict(loss=fit_loss, **_metrics(fit_rows, output[:fit_count], root, grid)),
                      probe=dict(loss=probe_loss, **_metrics(probe_rows, output[fit_count:], root, grid)),
                      both_absent_null={row['id']: float(p[-1])
                                        for row, p in zip(rows, output, strict=True)
                                        if row['id'].rsplit('_', 1)[0] in both_absent})
        write_dense_report(args.output/name, rows, output, root, grid, result)
        return result

    def save_checkpoint(name, epoch):
        file = args.output/name
        temp = file.with_suffix('.pt.tmp')
        torch.save(dict(provenance, epoch=epoch, model_state=model.state_dict()), temp)
        temp.replace(file)

    evaluation = evaluate()
    report('before', evaluation)
    best_evaluation = evaluation
    best_epoch = start_epoch
    plateau_reference = evaluation[1]
    stale_checks = 0
    stop_reason = 'epoch_limit'
    if args.patience:
        save_checkpoint('best_fit.pt', start_epoch)
    history = []
    for step in range(1, args.epochs+1):
        epoch = start_epoch+step
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
        entry = dict(epoch=epoch, fit_loss=total/len(fit_rows))
        checked = args.patience and (step % args.eval_every == 0 or step == args.epochs)
        if checked:
            evaluation = evaluate()
            scores, fit_loss, probe_loss = evaluation
            probe = _metrics(probe_rows, scores[fit_count:], root, grid)
            if fit_loss < best_evaluation[1]:
                best_evaluation, best_epoch = evaluation, epoch
                save_checkpoint('best_fit.pt', epoch)
            if plateau_reference - fit_loss >= args.min_delta:
                plateau_reference, stale_checks = fit_loss, 0
            else:
                stale_checks += 1
            entry.update(fit_eval_loss=fit_loss, probe_eval_loss=probe_loss,
                         probe_top5=probe['spatial_top5'], stale_checks=stale_checks)
            print(f'Check {epoch} | fit {fit_loss:.6f} | probe {probe_loss:.6f} | '
                  f'probe Top-5 {probe["spatial_top5"]} | plateau {stale_checks}/{args.patience}', flush=True)
        history.append(entry)
        save_json(args.output/'history.json', history)
        print(f'Epoch {epoch} (+{step}/{args.epochs}) | fit loss {entry["fit_loss"]:.6f}', flush=True)
        save_checkpoint('last.pt', epoch)
        if args.patience and stale_checks >= args.patience:
            stop_reason = 'fit_loss_plateau'
            break
    final = report('after', evaluation if args.patience else evaluate())
    if args.patience:
        run['best_fit'] = dict(epoch=best_epoch, evaluation=report('best_fit', best_evaluation))
    run.update(status='complete', start_epoch=start_epoch, final_epoch=epoch,
               stop_reason=stop_reason, after=final)
    save_json(args.output/'run.json', run)
    print(f'Stopped at epoch {epoch}: {stop_reason}. Final heatmaps:', args.output/'after/index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/search_grid_smoke_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache', type=Path, help='reuse frozen features for the exact same selection')
    parser.add_argument('--model-repo', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--fit-states', nargs='+', default=DEFAULT_FIT)
    parser.add_argument('--probe-states', nargs='+', default=DEFAULT_PROBE)
    parser.add_argument('--init-checkpoint', type=Path, help='start from saved weights with a new optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='maximum number of additional epochs')
    parser.add_argument('--patience', type=int, default=0, help='stop after this many stale fit checks (0 disables)')
    parser.add_argument('--eval-every', type=int, default=5, help='check plateau every N epochs')
    parser.add_argument('--min-delta', type=float, default=.005, help='minimum meaningful fit loss decrease')
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
    if args.patience < 0 or args.eval_every < 1 or not np.isfinite(args.min_delta) or args.min_delta <= 0:
        parser.error('invalid patience, check interval, or minimum improvement')
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        parser.error('--init-checkpoint must point to an existing checkpoint file')
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
