"""Separate representation/value optimization with versioned frozen checkpoints."""

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evidence_labels import fingerprint, represent_state, sample_state
from .evidence_model import ActionValue, EvidenceModel, prediction_loss
from .evidence_state import legal_actions


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_memory(model, bank, episodes, output, steps=200, alpha=.1, seed=7, lr=.0003,
                 max_history_steps=6):
    if not episodes or steps < 1 or alpha < 0:
        raise ValueError('need episodes, positive steps and nonnegative alpha')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    rng = random.Random(seed)
    target_rng = random.Random(seed + 103)
    model.train().requires_grad_(True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    history, start = [], time.monotonic()
    for step in range(steps):
        episode = episodes[rng.randrange(len(episodes))]
        state = sample_state(episode, rng, max_history_steps)
        optimizer.zero_grad(set_to_none=True)
        q, memory, _ = represent_state(model, bank, episode, state)
        logits = model.match(q, memory)
        match_loss = F.cross_entropy(logits, torch.tensor([episode.target], device=device))
        pred_loss = match_loss * 0.
        candidates = []
        seen = {o.image_hash for h in state.histories for o in h}
        for action in legal_actions(episode, state):
            result = episode.outcomes[action.id]
            if result.valid and result.image_hash not in seen:
                candidates.append(action)
        if alpha and candidates:
            action = target_rng.choice(candidates)
            target = bank.get(episode.outcomes[action.id].image_hash)
            prediction = model.predict(memory[:, action.candidate],
                                       torch.tensor([action.pose], device=device))
            pred_loss = prediction_loss(prediction, target['teacher'][None].to(device),
                                        target['teacher_valid'][None].to(device))
        loss = match_loss + alpha*pred_loss
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite representation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        history.append({'step': step+1, 'episode_id': episode.id,
                        'history_ids': [[o.id for o in h] for h in state.histories],
                        'match_loss': float(match_loss.detach()),
                        'prediction_loss': float(pred_loss.detach()),
                        'loss': float(loss.detach())})
        if step == 0 or (step+1) % 20 == 0 or step+1 == steps:
            recent = history[-20:]
            print(f'Memory alpha={alpha:g} step {step+1}/{steps}: '
                  f'match={np.mean([h["match_loss"] for h in recent]):.5f} '
                  f'pred={np.mean([h["prediction_loss"] for h in recent]):.5f}', flush=True)
    metadata = dict(status='complete', stage='representation', steps=steps, alpha=alpha,
                    seed=seed, lr=lr, seconds=time.monotonic()-start,
                    feature_config=bank.config, model_config=model.config,
                    training_episodes=[e.id for e in episodes],
                    training_groups=sorted({e.group for e in episodes}),
                    model_fingerprint=fingerprint(model), selection='fixed_steps_last_checkpoint')
    torch.save({'metadata': metadata, 'state_dict': model.cpu().state_dict()}, output/'last.pt')
    model.to(device).eval()
    save_json(output/'training.json', metadata)
    save_json(output/'history.json', history)
    return metadata


def load_memory(path, feature_config, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    if meta['feature_config'] != feature_config:
        raise ValueError('feature configuration differs from checkpoint')
    model = EvidenceModel(**meta['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if fingerprint(model) != meta['model_fingerprint']:
        raise ValueError('model checkpoint fingerprint mismatch')
    return model.to(device).eval(), meta


def train_value(runtime, records, output, steps=400, seed=7, lr=.001):
    if not records or steps < 1:
        raise ValueError('need nonempty labels and positive steps')
    if any(r['upstream_fingerprint'] != runtime.fingerprint for r in records):
        raise ValueError('value labels have wrong upstream fingerprint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    value = ActionValue(dim=runtime.model.dim, heads=runtime.model.config['heads']).to(runtime.device)
    optimizer = torch.optim.AdamW(value.parameters(), lr=lr, weight_decay=.01)
    rng = random.Random(seed)
    gaps = [float(torch.topk(r['inputs']['scores'][0], 2).values.diff().abs()) for r in records]
    boundary = float(np.median(gaps))
    difficult = [r for r, gap in zip(records, gaps, strict=True) if gap <= boundary]
    losses, start = [], time.monotonic()
    for step in range(steps):
        record = rng.choice(difficult if step % 2 == 0 else records)
        inputs = {k: v.to(runtime.device) for k, v in record['inputs'].items()}
        target = record['targets'].to(runtime.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = value(**inputs)
        loss = F.mse_loss(prediction/4., target/4.)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite action value loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            print(f'Value step {step+1}/{steps}: scaled_mse={np.mean(losses[-100:]):.7f}',
                  flush=True)
    meta = dict(status='complete', stage='value', steps=steps, seed=seed, lr=lr,
                upstream_fingerprint=runtime.fingerprint, model_config=value.config,
                seconds=time.monotonic()-start, states=len(records),
                training_episodes=sorted({r['episode_id'] for r in records}),
                training_groups=sorted({r['group'] for r in records}),
                model_fingerprint=fingerprint(value))
    torch.save({'metadata': meta, 'state_dict': value.cpu().state_dict()}, output/'last.pt')
    value.to(runtime.device).eval()
    save_json(output/'training.json', meta)
    save_json(output/'history.json', {'scaled_mse': losses})
    torch.save(records, output/'labels.pt')
    return value, meta


def load_value(path, upstream_fingerprint, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    if meta['upstream_fingerprint'] != upstream_fingerprint:
        raise ValueError('value checkpoint belongs to another upstream model')
    value = ActionValue(**meta['model_config'])
    value.load_state_dict(checkpoint['state_dict'], strict=True)
    if fingerprint(value) != meta['model_fingerprint']:
        raise ValueError('value checkpoint fingerprint mismatch')
    return value.to(device).eval()


@torch.no_grad()
def evaluate_value(value, records, device='cpu'):
    errors, zero_errors, regrets, hits = [], [], [], []
    nonconstant = 0
    for record in records:
        prediction = value(**{k: v.to(device) for k, v in record['inputs'].items()})[0].cpu()
        target = record['targets'][0]
        errors.extend((prediction-target).square().tolist())
        zero_errors.extend(target.square().tolist())
        chosen = int(prediction.argmax())
        regrets.append(float(target.max()-target[chosen]))
        if float(target.max()-target.min()) > 1e-6:
            nonconstant += 1
            hits.append(float(target[chosen] >= target.max()-1e-6))
    return {'states': len(records), 'nonconstant_states': nonconstant,
            'mse': float(np.mean(errors)) if errors else None,
            'zero_predictor_mse': float(np.mean(zero_errors)) if errors else None,
            'mean_action_regret': float(np.mean(regrets)) if regrets else None,
            'best_action_rate_nonconstant': float(np.mean(hits)) if hits else None}
