"""Controlled all-state fits; future observations are diagnostic-only privileges."""

from collections import defaultdict
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .evidence_diagnostics import ranking_diagnostics
from .evidence_labels import fingerprint
from .evidence_model import ActionValue
from .evidence_training import save_json, seed_all
from .evidence_update_diagnostics import pack_records, phase_weights, state_losses


CONDITIONS = ('direct_single', 'direct_joint', 'observation_zero_joint', 'observation_real_joint')
BASE_INPUTS = {'q', 'memories', 'scores', 'counts', 'subjects', 'poses'}


class ObservationValue(ActionValue):
    """Same current-state branch plus a zero-initialized real-observation residual."""
    def __init__(self, dim=128, heads=4):
        super().__init__(dim, heads)
        self.observation_encoder = nn.Sequential(nn.LayerNorm(5*dim), nn.Linear(5*dim, dim), nn.GELU())
        self.observation_output = nn.Sequential(nn.Linear(5*dim+1, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.observation_output[-1].weight)
        nn.init.zeros_(self.observation_output[-1].bias)

    def forward(self, q, memories, scores, counts, subjects, poses, future, future_valid):
        combined = self.action_features(q, memories, scores, counts, subjects, poses)
        if future.shape != (*subjects.shape, 5, self.config['dim']) or future_valid.shape != subjects.shape:
            raise ValueError('future observations must align with actions')
        clean = future.masked_fill(~future_valid[..., None, None].bool(), 0)
        observed = self.observation_encoder(clean.flatten(2))
        residual = self.observation_output(torch.cat(
            (combined, observed, future_valid[..., None].to(combined.dtype)), -1))
        return 4*torch.tanh((self.output(combined)+residual).squeeze(-1))


def prepare_records(records, condition):
    if condition not in CONDITIONS:
        raise ValueError('unknown diagnostic condition')
    output = []
    for row in records:
        if not BASE_INPUTS <= row['inputs'].keys() or set(row['inputs'])-BASE_INPUTS-{'future', 'future_valid'}:
            raise ValueError('unexpected diagnostic input keys')
        inputs = {k: row['inputs'][k].detach().cpu() for k in BASE_INPUTS}
        if condition.startswith('observation_'):
            for key in ('future', 'future_valid'):
                value = row['inputs'][key].detach().cpu()
                inputs[key] = torch.zeros_like(value) if condition == 'observation_zero_joint' else value
        output.append({**row, 'inputs': inputs, 'targets': row['targets'].detach().cpu()})
    return output


def pack_observations(records):
    inputs, targets, valid = pack_records(records)
    if 'future' in records[0]['inputs']:
        for key in ('future', 'future_valid'):
            parts = []
            for row in records:
                value = row['inputs'][key]
                pad = value.new_zeros((1, targets.shape[1]-value.shape[1], *value.shape[2:]))
                parts.append(torch.cat((value, pad), 1))
            inputs[key] = torch.cat(parts)
    return inputs, targets, valid


def backward_chunks(model, records, weights, chunk_size):
    """Accumulate globally weighted gradients; the caller clips/steps only once."""
    if chunk_size < 1 or len(weights) != len(records):
        raise ValueError('invalid gradient chunks/weights')
    total = 0.
    for start in range(0, len(records), chunk_size):
        rows = records[start:start+chunk_size]
        inputs, target, valid = pack_observations(rows)
        loss = state_losses(model(**inputs), target, valid) @ weights[start:start+len(rows)]
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite diagnostic loss')
        loss.backward()
        total += float(loss.detach())
    return total


def summarize_states(rows):
    if not rows:
        return None
    pairs = sum(r['meaningful_pair_count'] for r in rows)
    return dict(states=len(rows), mse=float(np.average([r['mse'] for r in rows],
                                                     weights=[len(r['target']) for r in rows])),
                mean_action_regret=float(np.mean([r['regret'] for r in rows])),
                random_mean_regret=float(np.mean([r['random_regret'] for r in rows])),
                selected_mean_gain=float(np.mean([r['selected_gain'] for r in rows])),
                near_best_rate=float(np.mean([r['near_best'] for r in rows])),
                meaningful_pair_accuracy=sum(r['meaningful_pair_correct'] for r in rows)/pairs
                if pairs else None, meaningful_pair_count=pairs)


@torch.no_grad()
def evaluate_records(model, records, chunk_size=32, weights=None, detail=False):
    was_training = model.training
    model.eval()
    rows = []
    for start in range(0, len(records), chunk_size):
        batch = records[start:start+chunk_size]
        inputs, target, valid = pack_observations(batch)
        prediction = model(**inputs)
        if not torch.isfinite(prediction[valid]).all():
            raise RuntimeError('nonfinite diagnostic prediction')
        for row, p, y, mask in zip(batch, prediction, target, valid, strict=True):
            p, y = p[mask], y[mask]
            rows.append(dict(episode_id=row['episode_id'], capture_group=row.get('capture_group', row['episode_id']),
                             action_ids=row['action_ids'], history_ids=row['history_ids'],
                             prediction=p.tolist(), target=y.tolist(), mse=float((p-y).square().mean()),
                             zero_mse=float(y.square().mean()), selected_gain=float(y[p.argmax()]),
                             random_regret=float(y.max()-y.mean()), **ranking_diagnostics(p, y)))
    model.train(was_training)
    result = summarize_states(rows)
    if result is None:
        return None
    result['weighted_mse'] = float(np.average([r['mse'] for r in rows], weights=weights))
    result['zero_weighted_mse'] = float(np.average([r['zero_mse'] for r in rows], weights=weights))
    result['informative'] = summarize_states([r for r in rows if r['target_range'] >= .05])
    if detail:
        groups = defaultdict(list)
        for row in rows:
            groups[row['capture_group']].append(row)
        result['by_capture_group'] = {k: summarize_states(v) for k, v in groups.items()}
        result['rows'] = rows
    return result


def fit_condition(records, development, config, output, condition, steps=2000, seed=7,
                  lr=.0001, chunk_size=32, evaluate_every=50):
    if condition not in CONDITIONS or not records or min(steps, chunk_size, evaluate_every) < 1 or not (
            math.isfinite(lr) and lr > 0):
        raise ValueError('invalid diagnostic fit configuration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    seed_all(seed)
    model = (ObservationValue if condition.startswith('observation_') else ActionValue)(**config)
    initial = fingerprint(model)
    fixed = prepare_records(records, condition)
    dev = prepare_records(development, condition)
    gaps = [float(torch.topk(r['inputs']['scores'][0], 2).values.diff().abs()) for r in fixed]
    difficult = [i for i, gap in enumerate(gaps) if gap <= float(np.median(gaps))]
    average = (phase_weights(len(fixed), difficult, 1)+phase_weights(len(fixed), difficult, 2))/2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    rng, forwards, actions, active = random.Random(seed), 0, 0, 0
    joint_step = (steps//len(fixed))//2*2
    matched_step = joint_step*len(fixed) if condition == 'direct_single' else joint_step
    curve, updates, matched = [], [], None

    def snapshot(step):
        return dict(step=step, training_state_forwards=forwards, training_actions=actions,
                    active_loss_states=active,
                    **evaluate_records(model, fixed, chunk_size, average.tolist()))

    curve.append(snapshot(0))
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        if condition == 'direct_single':
            index = rng.choice(difficult if step % 2 else range(len(fixed)))
            loss = backward_chunks(model, [fixed[index]], torch.ones(1), 1)
            forwards += 1
            actions += fixed[index]['targets'].numel()
            active += 1
        else:
            weights = phase_weights(len(fixed), difficult, step)
            loss = backward_chunks(model, fixed, weights, chunk_size)
            forwards += len(fixed)
            actions += sum(r['targets'].numel() for r in fixed)
            active += int((weights > 0).sum())
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        if not math.isfinite(norm):
            raise RuntimeError('nonfinite diagnostic gradient')
        optimizer.step()
        updates.append(dict(step=step, scaled_mse=loss, gradient_norm=norm))
        if step == 1 or step % evaluate_every == 0 or step in (matched_step, steps):
            current = snapshot(step)
            curve.append(current)
            if step == matched_step:
                matched = current
        if step % 250 == 0 or step == steps:
            print(f'{output.parent.name}/{condition} {step}/{steps}: '
                  f'mse={curve[-1]["weighted_mse"]:.6g}, regret={curve[-1]["mean_action_regret"]:.6g}', flush=True)
    train_metrics = evaluate_records(model, fixed, chunk_size, average.tolist(), detail=True)
    dev_metrics = evaluate_records(model, dev, chunk_size, detail=True) if dev else None
    result = dict(status='complete', condition=condition, diagnostic_only=True, states=len(fixed),
                  steps=steps, seed=seed, lr=lr, chunk_size=chunk_size, model_config=config,
                  parameters=sum(p.numel() for p in model.parameters()), initial_fingerprint=initial,
                  final_fingerprint=fingerprint(model), hard_indices=difficult,
                  average_state_weights=average.tolist(), seconds=time.monotonic()-start,
                  curve=curve, matched_work=matched, training=train_metrics, development=dev_metrics,
                  selection='final fixed-step checkpoint; no development checkpoint selection',
                  scope='all unique training states; optional old development location, identity overlap',
                  work_note='counts exclude evaluation; joint forwards all states including zero phase weights')
    torch.save(dict(state_dict=model.state_dict(), config=config, diagnostic_only=True,
                    condition=condition), output/'last.pt')
    save_json(output/'fit.json', result)
    save_json(output/'updates.json', updates)
    return result
