"""Frozen runtime and counterfactual labels from actual recorded observations."""

import hashlib
import json
import random

import torch

from .evidence_state import apply_action, legal_actions, true_margin


def fingerprint(model):
    digest = hashlib.sha256(json.dumps(model.config, sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def represent_state(model, bank, episode, state, encoder=None):
    """Differentiable construction; an optional frozen encoder only caches real images."""
    device = next(model.parameters()).device
    if encoder is None:
        reference = bank.get(episode.reference_hash)
        q = model.encode_reference(reference['tokens'][None].to(device),
                                   reference['valid'][None].to(device))
        digests = list(dict.fromkeys(o.image_hash for h in state.histories for o in h))
        if digests:
            values = [bank.get(d) for d in digests]
            z = model.encode_observations(
                torch.stack([v['tokens'] for v in values]).to(device),
                torch.stack([v['valid'] for v in values]).to(device))
            encoded = dict(zip(digests, z, strict=True))
        else:
            encoded = {}
    else:
        q, encoded = encoder(episode, state)
    candidates, length = len(state.histories), max(1, max(map(len, state.histories)))
    z_grid = torch.zeros(candidates, length, 5, model.dim, device=device)
    poses = torch.zeros(candidates, length, 5, device=device)
    quality = torch.zeros(candidates, length, 3, device=device)
    valid = torch.zeros(candidates, length, dtype=torch.bool, device=device)
    for candidate, history in enumerate(state.histories):
        for i, observation in enumerate(history):
            z_grid[candidate, i] = encoded[observation.image_hash]
            poses[candidate, i] = torch.tensor(observation.pose, device=device)
            quality[candidate, i] = torch.tensor(observation.quality, device=device)
            valid[candidate, i] = True
    memory = model.aggregate(z_grid, poses, quality, valid)[None]
    counts = valid.sum(1).float()[None]
    return q, memory, counts


class FrozenRuntime:
    def __init__(self, model, bank, score_temperature=1.):
        if not 0 < score_temperature < float('inf'):
            raise ValueError('score temperature must be positive and finite')
        self.model = model.eval().requires_grad_(False)
        for parameter in self.model.parameters():
            parameter.grad = None
        self.bank = bank
        self.temperature = float(score_temperature)
        payload = {'model': fingerprint(model), 'features': bank.config,
                   'score_temperature': self.temperature, 'updater_version': 1}
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.device = next(model.parameters()).device
        self.references, self.observations = {}, {}

    def _encode(self, episode, state):
        if episode.reference_hash not in self.references:
            item = self.bank.get(episode.reference_hash)
            self.references[episode.reference_hash] = self.model.encode_reference(
                item['tokens'][None].to(self.device), item['valid'][None].to(self.device))
        digests = list(dict.fromkeys(o.image_hash for h in state.histories for o in h))
        missing = [d for d in digests if d not in self.observations]
        if missing:
            items = [self.bank.get(d) for d in missing]
            z = self.model.encode_observations(
                torch.stack([i['tokens'] for i in items]).to(self.device),
                torch.stack([i['valid'] for i in items]).to(self.device))
            self.observations.update(dict(zip(missing, z, strict=True)))
        return self.references[episode.reference_hash], self.observations

    @torch.no_grad()
    def encode_state(self, episode, state):
        q, memories, counts = represent_state(self.model, self.bank, episode, state, self._encode)
        scores = torch.tanh(self.model.match(q, memories) / self.temperature)
        return q, memories, scores, counts


class CLSRuntime:
    """No trained representation: cosine of the mean normalized real CLS observations."""
    def __init__(self, bank):
        self.bank = bank
        self.device = torch.device('cpu')

    @torch.no_grad()
    def encode_state(self, episode, state):
        q = torch.nn.functional.normalize(self.bank.get(episode.reference_hash)['cls'], dim=0)
        means = []
        for history in state.histories:
            if history:
                values = torch.stack([torch.nn.functional.normalize(
                    self.bank.get(o.image_hash)['cls'], dim=0) for o in history])
                means.append(torch.nn.functional.normalize(values.mean(0), dim=0))
            else:
                means.append(torch.zeros_like(q))
        memories = torch.stack(means)[None, :, None]
        scores = (memories[:, :, 0] * q).sum(-1)
        counts = torch.tensor([[len(h) for h in state.histories]], dtype=torch.float32)
        return q[None, None], memories, scores, counts


def policy_inputs(q, memories, scores, counts, actions):
    device = memories.device
    return dict(q=q, memories=memories, scores=scores, counts=counts,
                subjects=torch.tensor([[a.candidate for a in actions]], device=device),
                poses=torch.tensor([[a.pose for a in actions]], dtype=torch.float32, device=device))


def sample_state(episode, rng, max_steps=8):
    state = episode.initial
    actions = list(episode.actions)
    rng.shuffle(actions)
    for action in actions[:rng.randrange(min(max_steps, len(actions))+1)]:
        state = apply_action(state, action, episode.outcomes[action.id])
    return state


def generate_labels(runtime, episodes, states_per_episode=16, seed=7, max_history_steps=8):
    if states_per_episode < 1:
        raise ValueError('states_per_episode must be positive')
    rng, records = random.Random(seed), []
    for episode in episodes:
        for index in range(states_per_episode):
            state = episode.initial if index == 0 else sample_state(
                episode, rng, max_history_steps)
            actions = legal_actions(episode, state)
            if not actions:
                continue
            q, memories, scores, counts = runtime.encode_state(episode, state)
            before = true_margin(scores[0], episode.target)
            gains = []
            for action in actions:
                after = apply_action(state, action, episode.outcomes[action.id])
                after_scores = runtime.encode_state(episode, after)[2]
                gains.append(true_margin(after_scores[0], episode.target) - before)
            inputs = policy_inputs(q, memories, scores, counts, actions)
            records.append({'inputs': {k: v.detach().cpu() for k, v in inputs.items()},
                            'targets': torch.tensor([gains], dtype=torch.float32),
                            'episode_id': episode.id, 'group': episode.group,
                            'action_ids': [a.id for a in actions],
                            'history_ids': [[o.id for o in h] for h in state.histories],
                            'upstream_fingerprint': runtime.fingerprint})
        print(f'Value labels: {episode.id} ({len(records)} states)', flush=True)
    return records
