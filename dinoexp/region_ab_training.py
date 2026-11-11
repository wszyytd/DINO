"""Paired current-only training. Neural entry points require CUDA explicitly."""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .evidence_memory import EvidenceMatcher
from .region_geometry import remap_positives
from .region_ab_metrics import teacher_metrics, metric_counts
from .region_ab_data import load_bundle, write_json, sha, code_record, digest_json
from .region_ab_prepare import cuda_device


def initialization_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_model(cfg, seed, device):
    seed_all(seed)
    return EvidenceMatcher(dim=256, heads=cfg['heads'], slots=cfg['slots'], mode='current').to(device)


def make_batch(rows, labels, refs, features, device, permutations=None):
    """Allow-list network input: RGB-derived observations, validity and reference only."""
    capacity = max(1, max(len(r['candidates']) for r in rows))
    x = np.zeros((len(rows), capacity, 16, 256), np.float32)
    valid = np.zeros((len(rows), capacity, 16), bool)
    positive = np.zeros((len(rows), capacity+1), bool)
    references, refvalid = [], []
    for i, r in enumerate(rows):
        k, label = len(r['candidates']), labels[r['id']]
        order = permutations[i] if permutations is not None else list(range(k))
        remapped = remap_positives(label['positives'], order)
        f = features[r['id']]
        if k:
            x[i, :k], valid[i, :k] = f['observation'][order], f['valid'][order]
        positive[i, remapped if remapped else [capacity]] = True
        reference = refs[r['reference_sha256']]
        references.append(reference['tokens']); refvalid.append(reference['valid'])
    tensor = lambda a: torch.from_numpy(np.asarray(a)).to(device)
    inputs = dict(observation=tensor(x), valid=tensor(valid),
                  reference=tensor(np.stack(references)), reference_valid=tensor(np.stack(refvalid)))
    return inputs, tensor(positive)


def forward_current(model, inputs):
    state = model.update(inputs['observation'], inputs['valid'], state=None)
    logits = model.score(inputs['reference'], inputs['reference_valid'], state)
    available = torch.cat((state['seen'], torch.ones(len(logits), 1, dtype=torch.bool, device=logits.device)), 1)
    return logits.masked_fill(~available, torch.finfo(logits.dtype).min)


def set_loss(logits, positive):
    if positive.shape != logits.shape or not positive.any(1).all():
        raise ValueError('each sample needs positive candidates or Null')
    return (torch.logsumexp(logits, 1)-torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 1)).mean()


def complete_features(row, feature):
    return not feature['errors'] and (len(row['candidates']) == 0 or bool(feature['valid'].any(1).all()))


@torch.inference_mode()
def evaluate(model, rows, labels, refs, features, device, batch_size):
    model.eval(); predictions = []; forward_seconds = 0.
    start_total = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        inputs, _ = make_batch(batch, labels, refs, features, device)
        torch.cuda.synchronize(device); start_forward = time.perf_counter()
        logits = forward_current(model, inputs)
        torch.cuda.synchronize(device); forward_seconds += time.perf_counter()-start_forward
        values = logits.cpu().numpy()
        for i, r in enumerate(batch):
            k = len(r['candidates'])
            score = np.concatenate([values[i, :k], values[i, -1:]])
            probability = np.exp(score-score.max()); probability /= probability.sum()
            choice = int(np.argmax(score))
            # Unavailable candidates cannot be selected or win the ranking.
            available = features[r['id']]['valid'].any(1)
            ranked = np.where(available, score[:k], -np.inf)
            rank = int(np.argmax(ranked)) if available.any() else None
            predictions.append(dict(id=r['id'], scores=score.tolist(), probabilities=probability.tolist(),
                                    choice=None if choice == k else choice, rank_choice=rank,
                                    extraction_failure=not complete_features(r, features[r['id']])))
    return dict(predictions=predictions, metrics=metric_counts(predictions, labels),
                timing=dict(cached_matcher_forward_seconds=forward_seconds,
                            cached_evaluation_wall_seconds=time.perf_counter()-start_total,
                            decisions=len(rows), batch_size=batch_size,
                            scope='precomputed features; forward time excludes loading, batching and CPU transfer'))


def server_checks(device, cfg):
    """Synthetic CUDA forward/backward and permutation checks, never a performance claim."""
    seed_all(12345)
    model = make_model(cfg, 12345, device)
    x = torch.randn(2, 3, 16, 256, device=device)
    valid = torch.ones(2, 3, 16, dtype=torch.bool, device=device); valid[1] = False
    inputs = dict(observation=x, valid=valid, reference=torch.randn(2, 64, 256, device=device),
                  reference_valid=torch.ones(2, 64, dtype=torch.bool, device=device))
    before = forward_current(model, inputs)
    positive = torch.zeros_like(before, dtype=torch.bool); positive[0, [0, 2]] = True; positive[1, -1] = True
    loss = set_loss(before, positive)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError('nonfinite synthetic multi-positive/empty gradients')
    permutation = [2, 0, 1]
    moved = dict(inputs, observation=x[:, permutation], valid=valid[:, permutation])
    after = forward_current(model, moved)
    torch.testing.assert_close(after, before[:, [2, 0, 1, 3]], atol=2e-5, rtol=1e-5)
    if before[1].argmax().item() != 3:
        raise AssertionError('empty/invalid candidates must choose Null')
    # Zero physical candidates: one masked storage slot, only Null is a real output.
    empty_inputs = dict(inputs, observation=x[:, :1], valid=torch.zeros_like(valid[:, :1]))
    empty_scores = forward_current(model, empty_inputs)
    if not (empty_scores.argmax(1) == 1).all():
        raise AssertionError('zero-candidate storage padding selected')
    return dict(status='passed', multi_positive_loss=float(loss.detach()), permutation_atol=2e-5,
                checks=['multipositive backward', 'invalid/empty candidate Null', 'candidate permutation', 'fresh current state'])


def train(args):
    output = Path(args.output).resolve(); directory = output/'training'
    if directory.exists():
        raise ValueError('training exists; do not overwrite checkpoints or results')
    device = cuda_device(args.device); torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    meta, rows, labels, refs, features = load_bundle(output, args.data)
    cfg = meta['config']['training']
    training = [r for r in rows if r['split'] == 'train']
    partitions = {s: [r for r in rows if r['split'] == s] for s in sorted({r['split'] for r in rows})}
    joint_eligible = {r['id']: all(complete_features(r, features[v][r['id']]) for v in ('A', 'B')) for r in rows}
    if not any(joint_eligible[r['id']] for r in training):
        raise ValueError('no paired training samples have complete features; inspect preparation failures')
    directory.mkdir()
    checks = server_checks(device, cfg)
    # A few real-data updates, throw away all temporary weights before seeded runs.
    smoke = make_model(cfg, 12345, device)
    optimizer = torch.optim.AdamW(smoke.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    real_rows = [r for r in training if joint_eligible[r['id']]][:cfg['batch_size']]
    checks['real_smoke_losses'] = []
    for v in ('A', 'B'):
        inputs, positive = make_batch(real_rows, labels, refs, features[v], device)
        optimizer.zero_grad(set_to_none=True)
        loss = set_loss(forward_current(smoke, inputs), positive)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite real smoke loss')
        loss.backward(); optimizer.step(); checks['real_smoke_losses'].append(float(loss.detach()))
    del smoke, optimizer
    write_json(directory/'checks.json', checks)
    plan = dict(config=cfg, code=code_record(), prepared_sha256=sha(output/'prepared.json'),
                proposal_sha256=sha(output/'proposals/records.json'), labels_sha256=sha(output/'labels.json'),
                split=meta['split'], sampling='each unique frame-reference pair once per epoch; fresh current state',
                planned_batches_per_epoch=math.ceil(len(training)/cfg['batch_size']),
                planned_pairs_per_epoch=len(training),
                excluded_training_pairs=[r['id'] for r in training if not joint_eligible[r['id']]],
                failure_policy='same paired samples excluded from loss only; retained in all evaluation denominators; any extraction failure counts as chain failure',
                selection='fixed final epoch, no validation/probe/test selection', results=[])
    write_json(directory/'run.json', plan)
    for seed in cfg['seeds']:
        initial_hash = None; schedule_fingerprints = []
        for v in ('A', 'B'):
            folder = directory/f'{v}_seed{seed}'; folder.mkdir()
            model = make_model(cfg, seed, device)
            init_hash = initialization_hash(model)
            if initial_hash is not None and init_hash != initial_hash:
                raise AssertionError('A/B initialization mismatch')
            initial_hash = init_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            updates = 0; schedule = hashlib.sha256()
            torch.cuda.reset_peak_memory_stats(device); start_training = time.perf_counter()
            for epoch in range(cfg['epochs']):
                model.train(); rng = np.random.default_rng(seed*100000+epoch)
                order = rng.permutation(len(training)).tolist()
                permutations = {j: rng.permutation(len(training[j]['candidates'])).tolist() for j in order}
                schedule.update(digest_json([order, permutations]).encode())
                losses, samples, empty_batches = 0., 0, 0
                for start in range(0, len(order), cfg['batch_size']):
                    ids = [j for j in order[start:start+cfg['batch_size']] if joint_eligible[training[j]['id']]]
                    if not ids:
                        empty_batches += 1; continue
                    batch = [training[j] for j in ids]
                    inputs, positive = make_batch(batch, labels, refs, features[v], device, [permutations[j] for j in ids])
                    optimizer.zero_grad(set_to_none=True)
                    loss = set_loss(forward_current(model, inputs), positive)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite training loss; preserve run for diagnosis')
                    loss.backward(); optimizer.step(); updates += 1
                    losses += float(loss.detach())*len(ids); samples += len(ids)
                entry = dict(epoch=epoch+1, mean_loss=losses/max(1, samples), pairs_used=samples,
                             optimizer_updates=updates, empty_batches=empty_batches)
                with (folder/'train.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry)+'\n')
                if epoch == 0 or (epoch+1) % 10 == 0:
                    print(f'{v} seed={seed} epoch={epoch+1}/{cfg["epochs"]} loss={entry["mean_loss"]:.6f}', flush=True)
            train_seconds = time.perf_counter()-start_training
            fingerprint = (schedule.hexdigest(), updates)
            schedule_fingerprints.append(fingerprint)
            if len(schedule_fingerprints) == 2 and schedule_fingerprints[0] != fingerprint:
                raise AssertionError('A/B sampling schedule or optimizer budget differs')
            checkpoint = dict(kind='region_current_matcher_v1', variant=v, seed=seed, epoch=cfg['epochs'],
                              config=model.config, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                              prepared_sha256=plan['prepared_sha256'], proposal_sha256=plan['proposal_sha256'],
                              labels_sha256=plan['labels_sha256'], initial_parameters_sha256=init_hash,
                              schedule_sha256=schedule.hexdigest(), optimizer_updates=updates, code=plan['code'])
            torch.save(checkpoint, folder/'last.pt')
            result = dict(variant=v, seed=seed, checkpoint=str((folder/'last.pt').relative_to(output)).replace('\\', '/'),
                          checkpoint_sha256=sha(folder/'last.pt'), initial_parameters_sha256=init_hash,
                          schedule_sha256=schedule.hexdigest(), optimizer_updates=updates, train_seconds=train_seconds,
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device), splits={})
            for name, selected in partitions.items():
                evaluation = evaluate(model, selected, labels, refs, features[v], device, cfg['batch_size'])
                write_json(folder/(name+'.json'), evaluation)
                result['splits'][name] = dict(metrics=evaluation['metrics'], timing=evaluation['timing'],
                                             evaluation_sha256=sha(folder/(name+'.json')))
            # Reload saved weights, replay every held-out pair, not just one easy sample.
            saved = torch.load(folder/'last.pt', map_location=device, weights_only=True)
            replay_model = EvidenceMatcher(**saved['config']).to(device)
            replay_model.load_state_dict(saved['model_state'], strict=True)
            max_error = 0.
            from .region_ab_data import read_json
            for name, selected in partitions.items():
                if name == 'train':
                    continue
                before = read_json(folder/(name+'.json'))['predictions']
                after = evaluate(replay_model, selected, labels, refs, features[v], device, cfg['batch_size'])['predictions']
                for p, q in zip(before, after, strict=True):
                    error = float(np.max(np.abs(np.array(p['scores'])-np.array(q['scores']))))
                    max_error = max(max_error, error)
                    if p['choice'] != q['choice'] or error > 2e-5:
                        raise AssertionError('saved checkpoint does not reproduce evaluation')
            result['reload_max_abs_score_error'] = max_error
            plan['results'].append(result); write_json(directory/'run.json', plan)
            print(v, seed, {s: r['metrics']['end_to_end_accuracy'] for s, r in result['splits'].items()}, flush=True)
            del model, replay_model, optimizer, checkpoint, saved
    plan['status'] = 'complete'; write_json(directory/'run.json', plan)
    write_json(directory/'teacher_metrics.json', {s: teacher_metrics(rs, labels) for s, rs in partitions.items()})
    return plan
