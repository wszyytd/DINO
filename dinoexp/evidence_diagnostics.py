"""Training-only, frozen-input diagnostics; no changes to the deployed policy."""

from collections import defaultdict
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .evidence_model import ActionValue
from .evidence_state import apply_action, legal_actions, true_margin
from .evidence_training import evaluate_value, save_json, seed_all


def state_key(record):
    return record['episode_id'], tuple(tuple(sorted(h)) for h in record['history_ids'])


def unique_record_indices(records):
    """Set-equivalent histories with identical available actions are one state."""
    seen, indices = {}, []
    for i, row in enumerate(records):
        key = (state_key(row), tuple(sorted(row['action_ids'])))
        if key not in seen:
            seen[key] = i
            indices.append(i)
            continue
        previous = records[seen[key]]
        a = dict(zip(previous['action_ids'], previous['targets'][0].tolist(), strict=True))
        b = dict(zip(row['action_ids'], row['targets'][0].tolist(), strict=True))
        if any(abs(a[k]-b[k]) > 1e-5 for k in a):
            raise ValueError('conflicting labels for a repeated state')
    return indices, len(records)-len(indices)


def paired_selection(branches, sizes=(1, 4, 16), seed=7, min_range=.05):
    if not branches or not sizes or min(sizes) < 1 or min_range <= 0:
        raise ValueError('need branches, positive sizes and meaningful target range')
    first = branches[0]
    unique, duplicates = unique_record_indices(first)
    for branch in branches[1:]:
        unique_record_indices(branch)
        if len(branch) != len(first) or any(
                state_key(a) != state_key(b) or a['action_ids'] != b['action_ids']
                for a, b in zip(first, branch, strict=True)):
            raise ValueError('branches must have aligned states and action lists')
    eligible = [i for i in unique if all(
        float(branch[i]['targets'].max()-branch[i]['targets'].min()) >= min_range
        for branch in branches)]
    groups = defaultdict(list)
    for i in eligible:
        groups[first[i]['episode_id']].append(i)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        rng.shuffle(groups[name])
    ordered = []
    while any(groups.values()):
        for name in names:
            if groups[name]:
                ordered.append(groups[name].pop())
    if len(ordered) < max(sizes):
        raise ValueError('not enough unique informative states for requested fit sizes')
    return dict(raw_states=len(first), unique_states=len(unique), duplicates=duplicates,
                informative_states=len(eligible), min_range=min_range, seed=seed,
                eligible_episodes=sorted(names), unique_indices=unique,
                selection='training-label range filter, seeded episode round robin; nested',
                subsets={str(size): ordered[:size] for size in sizes})


def reconstruct_state(episode, record):
    if record['episode_id'] != episode.id or len(record['history_ids']) != len(episode.candidate_ids):
        raise ValueError('record belongs to another episode or candidate layout')
    by_id = {a.id: a for a in episode.actions}
    remaining = record['action_ids']
    if len(set(remaining)) != len(remaining) or not set(remaining) <= by_id.keys():
        raise ValueError('invalid remaining action ids')
    attempted = set(by_id)-set(remaining)
    state = episode.initial
    for candidate, ids in enumerate(record['history_ids']):
        initial = [o.id for o in state.histories[candidate]]
        if ids[:len(initial)] != initial or len(set(ids)) != len(ids):
            raise ValueError('history must retain its initial observations without duplicates')
        for sample in ids[len(initial):]:
            if sample not in attempted or by_id[sample].candidate != candidate:
                raise ValueError('history contains a future or wrong-candidate observation')
            state = apply_action(state, by_id[sample], episode.outcomes[sample])
    # A failed capture (or exact replay) consumed an action without adding a history item.
    for action in episode.actions:
        if action.id in attempted and action.id not in state.attempted:
            state = apply_action(state, action, episode.outcomes[action.id])
    if [[o.id for o in h] for h in state.histories] != record['history_ids']:
        raise ValueError('saved history cannot be reconstructed from actual observations')
    if [a.id for a in legal_actions(episode, state)] != remaining:
        raise ValueError('reconstructed legal action order differs')
    return state


def transition_result(before, after, target):
    def prediction(scores):
        top = max(scores)
        winners = [i for i, v in enumerate(scores) if abs(v-top) <= 1e-8]
        return winners[0] if len(winners) == 1 else None
    pre, post = true_margin(before, target), true_margin(after, target)
    correct_pre, correct_post = pre > 1e-8, post > 1e-8
    kind = ('stayed_correct' if correct_post else 'introduced_error') if correct_pre else (
        'corrected' if correct_post else 'stayed_incorrect')
    return dict(before_scores=list(before), after_scores=list(after),
                before_margin=pre, after_margin=post, gain=post-pre,
                before_correct=correct_pre, after_correct=correct_post,
                before_prediction=prediction(before), after_prediction=prediction(after),
                transition=kind)


def ranking_diagnostics(prediction, target, gap=.01):
    prediction, target = prediction.detach().cpu(), target.detach().cpu()
    delta = target[:, None]-target[None, :]
    comparable = torch.triu(delta.abs() > gap, diagonal=1)
    pd = prediction[:, None]-prediction[None, :]
    count = int(comparable.sum())
    correct = int(((delta*pd > 0) & comparable).sum())
    chosen = int(prediction.argmax())
    return dict(meaningful_pair_count=count, meaningful_pair_correct=correct,
                meaningful_pair_accuracy=correct/count if count else None,
                regret=float(target.max()-target[chosen]),
                near_best=float(target.max()-target[chosen]) <= gap,
                prediction_range=float(prediction.max()-prediction.min()),
                target_range=float(target.max()-target.min()))


def input_audit(records):
    """Exact collisions are contradictions; similar vectors alone are not proof of a defect."""
    seen, conflicts = {}, []
    q_norm, memory_norm, pose_norm = [], [], []
    for i, row in enumerate(records):
        x = row['inputs']
        q_norm.append(float(x['q'].mean(1).norm()))
        memory_norm.extend(x['memories'].mean(2)[0].norm(dim=-1).tolist())
        pose_norm.extend(x['poses'][0].norm(dim=-1).tolist())
        effective = (x['q'].mean(1), x['memories'].mean(2), x['scores'], x['counts'])
        common = b''.join(v.contiguous().numpy().tobytes() for v in effective)
        for j in range(len(row['action_ids'])):
            key = (common, int(x['subjects'][0, j]), x['poses'][0, j].numpy().tobytes())
            target = float(row['targets'][0, j])
            if key in seen and abs(seen[key][2]-target) > 1e-5:
                conflicts.append({'first': seen[key][:2], 'second': [i, j],
                                  'target_difference': abs(seen[key][2]-target)})
            else:
                seen[key] = (i, j, target)
    return dict(exact_input_label_conflicts=conflicts,
                mean_reference_norm=float(np.mean(q_norm)),
                mean_memory_norm=float(np.mean(memory_norm)),
                mean_pose_norm=float(np.mean(pose_norm)),
                collision_scope='means actually consumed by ActionValue, scores/counts, subject and pose',
                note='Norm differences are descriptive, not a causal diagnosis.')


def fixed_fit(records, config, output, steps=2000, evaluate_every=50,
              seed=7, lr=.001, device='cpu', gap=.01):
    """Original architecture/optimizer/sampler on fixed records; evaluate the full set."""
    if not records or steps < 1 or evaluate_every < 1 or lr <= 0:
        raise ValueError('invalid fixed-fit settings')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    model = ActionValue(**config).to(device)
    original = {k: v.detach().clone() for k, v in model.named_parameters()}
    fixed = [{**r, 'inputs': {k: v.detach().to(device) for k, v in r['inputs'].items()},
              'targets': r['targets'].detach().to(device)} for r in records]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    rng = random.Random(seed)
    gaps = [float(torch.topk(r['inputs']['scores'][0], 2).values.diff().abs()) for r in fixed]
    hard = [i for i, value in enumerate(gaps) if value <= float(np.median(gaps))]
    curve, updates = [], []

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        result = evaluate_value(model, records, device)
        extras = [ranking_diagnostics(model(**r['inputs'])[0], r['targets'][0], gap) for r in fixed]
        pairs = sum(r['meaningful_pair_count'] for r in extras)
        result.update(step=step, meaningful_pair_count=pairs,
                      meaningful_pair_accuracy=sum(r['meaningful_pair_correct'] for r in extras)/pairs
                      if pairs else None, near_best_rate=float(np.mean([r['near_best'] for r in extras])),
                      mean_prediction_range=float(np.mean([r['prediction_range'] for r in extras])))
        curve.append(result)
        model.train()

    evaluate(0)
    for step in range(1, steps+1):
        index = rng.choice(hard if (step-1) % 2 == 0 else range(len(fixed)))
        row = fixed[index]
        optimizer.zero_grad(set_to_none=True)
        predicted = model(**row['inputs'])
        loss = F.mse_loss(predicted/4., row['targets']/4.)
        loss.backward()
        module_norms = defaultdict(float)
        for name, param in model.named_parameters():
            if param.grad is not None:
                module_norms[name.split('.')[0]] += float(param.grad.square().sum())
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        if not math.isfinite(norm) or not torch.isfinite(loss):
            raise RuntimeError('nonfinite diagnostic loss/gradient')
        optimizer.step()
        updates.append(dict(step=step, record=index, scaled_mse=float(loss.detach()),
                            gradient_norm=norm, modules={k: math.sqrt(v) for k, v in module_norms.items()}))
        if step == 1 or step % evaluate_every == 0 or step == steps:
            evaluate(step)
        if step % 500 == 0 or step == steps:
            print(f'Fit {output.name} step {step}/{steps}: mse={curve[-1]["mse"]:.6g} '
                  f'regret={curve[-1]["mean_action_regret"]:.6g}', flush=True)
    parameter_delta = math.sqrt(sum(float((p.detach()-original[k]).square().sum())
                                   for k, p in model.named_parameters()))
    model.eval()
    with torch.no_grad():
        predictions = [model(**r['inputs']).cpu().tolist() for r in fixed]
    result = dict(status='complete', scope='fixed training set fitting, not generalization',
                  states=len(records), steps=steps, seed=seed, lr=lr, weight_decay=.01,
                  gradient_clip=5., meaningful_gap=gap, model_config=config,
                  selection='final checkpoint, no best-checkpoint selection',
                  sampler='alternating lower-half current score gaps and all fixed states',
                  parameter_delta_l2=parameter_delta, input_audit=input_audit(records),
                  curve=curve, updates=updates, predictions=predictions,
                  targets=[r['targets'].tolist() for r in records],
                  state_ids=[{'episode': r['episode_id'], 'histories': r['history_ids'],
                              'actions': r['action_ids']} for r in records])
    save_json(output/'fit.json', result)
    torch.save({'config': config, 'state_dict': model.cpu().state_dict(),
                'diagnostic_only': True}, output/'last.pt')
    return result
