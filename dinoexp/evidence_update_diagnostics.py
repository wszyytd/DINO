"""Controlled single-state versus exact phase-expected joint update experiments."""

import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evidence_diagnostics import ranking_diagnostics
from .evidence_labels import fingerprint
from .evidence_model import ActionValue
from .evidence_training import save_json, seed_all


def pack_records(records):
    if not records or any(r['targets'].shape[1] == 0 for r in records):
        raise ValueError('need nonempty states with actions')
    lengths = [r['targets'].shape[1] for r in records]
    maximum = max(lengths)
    packed = {}
    for key in ('q', 'memories', 'scores', 'counts'):
        packed[key] = torch.cat([r['inputs'][key].detach() for r in records], 0)
    for key in ('subjects', 'poses'):
        pieces = []
        for r, length in zip(records, lengths, strict=True):
            value = r['inputs'][key].detach()
            pad = torch.zeros((1, maximum-length, *value.shape[2:]), dtype=value.dtype,
                              device=value.device)
            pieces.append(torch.cat((value, pad), 1))
        packed[key] = torch.cat(pieces, 0)
    targets = torch.zeros(len(records), maximum, dtype=records[0]['targets'].dtype)
    valid = torch.zeros_like(targets, dtype=torch.bool)
    for i, (r, length) in enumerate(zip(records, lengths, strict=True)):
        targets[i, :length] = r['targets'].detach()[0]
        valid[i, :length] = True
    return packed, targets, valid


def state_losses(prediction, targets, valid):
    """Mean squared error /16 per state, excluding padding BEFORE squaring."""
    difference = (prediction-targets).masked_fill(~valid, 0)/4.
    return difference.square().sum(1)/valid.sum(1)


def phase_weights(count, difficult, step):
    if count < 1 or not difficult or step < 1:
        raise ValueError('positive count/step and a nonempty hard subset required')
    weights = torch.zeros(count)
    if step % 2:
        weights[difficult] = 1./len(difficult)
    else:
        weights.fill_(1./count)
    return weights


def gradient_geometry(gradients, weights):
    gradients, weights = gradients.double(), weights.double()
    norms = gradients.norm(dim=1)
    nonzero = norms > 1e-12
    active = gradients[nonzero]/norms[nonzero, None]
    cosine = active @ active.T
    mask = torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)
    pairs = cosine[mask]
    average = (weights[:, None]*gradients).sum(0)
    expected_squared_norm = (weights*norms.square()).sum()
    return dict(nonzero_gradient_states=int(nonzero.sum()),
                negative_cosine_fraction=float((pairs < -1e-8).double().mean()) if len(pairs) else None,
                mean_pair_cosine=float(pairs.mean()) if len(pairs) else None,
                cosine_matrix=cosine.tolist(), gradient_norms=norms.tolist(),
                mean_gradient_norm=float(average.norm()),
                mean_to_individual_squared_norm_ratio=float(average.square().sum()/expected_squared_norm)
                if expected_squared_norm > 1e-24 else None)


def gradient_probe(model, records, weights):
    params = tuple(model.parameters())
    gradients = []
    for row in records:
        loss = F.mse_loss(model(**row['inputs'])/4., row['targets']/4.)
        values = torch.autograd.grad(loss, params)
        gradients.append(torch.cat([g.detach().flatten() for g in values]))
    return gradient_geometry(torch.stack(gradients), weights)


@torch.no_grad()
def measure(model, packed, targets, valid, average_weights):
    prediction = model(**packed)
    losses = state_losses(prediction, targets, valid)*16.
    lengths = valid.sum(1)
    ranks = [ranking_diagnostics(p[m], y[m]) for p, y, m in zip(prediction, targets, valid, strict=True)]
    pairs = sum(r['meaningful_pair_count'] for r in ranks)
    return dict(state_mse=losses.tolist(), weighted_mse=float(losses @ average_weights),
                mse=float((losses*lengths).sum()/lengths.sum()),
                mean_action_regret=float(np.mean([r['regret'] for r in ranks])),
                near_best_rate=float(np.mean([r['near_best'] for r in ranks])),
                meaningful_pair_accuracy=sum(r['meaningful_pair_correct'] for r in ranks)/pairs
                if pairs else None)


def fit_updates(records, config, output, mode, steps=2000, seed=7, lr=.0001,
                probe_steps=(0, 125, 500, 1000, 1500, 2000)):
    if mode not in ('single', 'joint') or steps < 1 or lr <= 0 or not records:
        raise ValueError('invalid update experiment configuration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    seed_all(seed)
    model = ActionValue(**config).train()
    initial_fingerprint = fingerprint(model)
    fixed = [{**r, 'inputs': {k: v.detach() for k, v in r['inputs'].items()},
              'targets': r['targets'].detach()} for r in records]
    packed, targets, valid = pack_records(fixed)
    gaps = [float(torch.topk(r['inputs']['scores'][0], 2).values.diff().abs()) for r in fixed]
    difficult = [i for i, gap in enumerate(gaps) if gap <= float(np.median(gaps))]
    average_weights = (phase_weights(len(fixed), difficult, 1)+phase_weights(len(fixed), difficult, 2))/2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    rng = random.Random(seed)
    curve, per_step, probes = [], [], []
    forwards, active_states, actions = 0, 0, 0
    previous = measure(model, packed, targets, valid, average_weights)

    def snapshot(step, metrics):
        curve.append(dict(step=step, training_state_forwards=forwards,
                          active_loss_states=active_states, training_actions=actions, **metrics))

    snapshot(0, previous)
    per_step.append(dict(step=0, **previous))
    if 0 in probe_steps:
        probes.append(dict(step=0, **gradient_probe(model, fixed, average_weights)))
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        chosen = None
        if mode == 'single':
            chosen = rng.choice(difficult if step % 2 else range(len(fixed)))
            row = fixed[chosen]
            loss = F.mse_loss(model(**row['inputs'])/4., row['targets']/4.)
            forwards += 1
            active_states += 1
            actions += row['targets'].numel()
        else:
            weights = phase_weights(len(fixed), difficult, step)
            loss = state_losses(model(**packed), targets, valid) @ weights
            forwards += len(fixed)
            active_states += int((weights > 0).sum())
            actions += int(valid.sum())
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        if not math.isfinite(norm) or not torch.isfinite(loss):
            raise RuntimeError('nonfinite update')
        optimizer.step()
        current = measure(model, packed, targets, valid, average_weights)
        per_step.append(dict(step=step, chosen=chosen, gradient_norm=norm,
                             delta_weighted_mse=current['weighted_mse']-previous['weighted_mse'],
                             delta_chosen_mse=current['state_mse'][chosen]-previous['state_mse'][chosen]
                             if chosen is not None else None, **current))
        if step == 1 or step % 50 == 0 or step in probe_steps or step == steps:
            snapshot(step, current)
        if step in probe_steps:
            probes.append(dict(step=step, **gradient_probe(model, fixed, average_weights)))
        previous = current
        if step % 500 == 0 or step == steps:
            print(f'{output.parent.name}/{output.name} {mode} {step}/{steps}: '
                  f'weighted_mse={current["weighted_mse"]:.6g} '
                  f'regret={current["mean_action_regret"]:.6g}', flush=True)
    tail = per_step[max(1, len(per_step)-500):]
    weighted = np.array([r['weighted_mse'] for r in tail])
    regrets = np.array([r['mean_action_regret'] for r in tail])
    increases = [r for r in tail if r['delta_weighted_mse'] > 1e-6]
    interference = [r for r in increases if r['delta_chosen_mse'] is not None
                    and r['delta_chosen_mse'] < -1e-6]
    stability = dict(updates=len(tail), mean_weighted_mse=float(weighted.mean()),
                     std_weighted_mse=float(weighted.std()), worst_weighted_mse=float(weighted.max()),
                     mean_regret=float(regrets.mean()), worst_regret=float(regrets.max()),
                     objective_increase_updates=len(increases),
                     sampled_improves_but_objective_worsens=len(interference) if mode == 'single' else None)
    result = dict(status='complete', scope='fixed training-state dynamics; no held-out evaluation',
                  mode=mode, seed=seed, lr=lr, steps=steps, states=len(records), model_config=config,
                  initial_fingerprint=initial_fingerprint, final_fingerprint=fingerprint(model),
                  hard_indices=difficult, average_state_weights=average_weights.tolist(),
                  sample_weights='odd=uniform hard half, even=uniform all; identical phase expectation',
                  zero_mse=float(targets[valid].square().mean()),
                  random_mean_regret=float(np.mean([float(y[m].max()-y[m].mean())
                                                   for y, m in zip(targets, valid, strict=True)])),
                  seconds=time.monotonic()-start, curve=curve, per_step=per_step,
                  gradient_probes=probes, tail_500=stability,
                  budget_note='training_state_forwards excludes common evaluation/gradient probes; '
                              'joint includes zero-weight states forwarded on hard-only phases')
    save_json(output/'updates.json', result)
    torch.save({'state_dict': model.state_dict(), 'config': config,
                'diagnostic_only': True}, output/'last.pt')
    return result
