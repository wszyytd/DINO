"""Privileged observation construction, confined to the diagnostic entry point."""

import torch

from .evidence_diagnostics import reconstruct_state, state_key, unique_record_indices
from .evidence_labels import policy_inputs
from .evidence_state import apply_action, legal_actions, true_margin


def paired_unique(branches):
    first = next(iter(branches.values()))
    indices, duplicates = unique_record_indices(first)
    for rows in branches.values():
        other, _ = unique_record_indices(rows)
        if other != indices or len(rows) != len(first) or any(
                state_key(a) != state_key(b) or a['action_ids'] != b['action_ids']
                for a, b in zip(first, rows, strict=True)):
            raise ValueError('branches must have aligned states and actions')
    return {name: [rows[i] for i in indices] for name, rows in branches.items()}, dict(
        raw_states=len(first), unique_states=len(indices), duplicates=duplicates, indices=indices)


@torch.no_grad()
def attach_observations(runtime, episodes, records):
    """Verify original inputs/labels and attach frozen real features in exact action order."""
    by_episode = {e.id: e for e in episodes}
    enriched, maximum_input, maximum_label = [], 0., 0.
    for row in records:
        if row['episode_id'] not in by_episode or row['upstream_fingerprint'] != runtime.fingerprint:
            raise ValueError('record outside the allowed episode role or upstream fingerprint')
        episode = by_episode[row['episode_id']]
        state = reconstruct_state(episode, row)
        actions = legal_actions(episode, state)
        encoded = runtime.encode_state(episode, state)
        inputs = policy_inputs(*encoded, actions)
        if set(inputs) != set(row['inputs']):
            raise ValueError('unexpected saved policy input keys')
        for key, value in inputs.items():
            saved = row['inputs'][key]
            if value.shape != saved.shape or not torch.allclose(value.cpu(), saved, atol=5e-4, rtol=5e-4):
                raise ValueError('reconstructed policy input differs: '+key)
            maximum_input = max(maximum_input, float((value.cpu()-saved).abs().max()))
        future = torch.zeros(1, len(actions), 5, runtime.model.dim)
        valid = torch.zeros(1, len(actions), dtype=torch.bool)
        gains = []
        before = true_margin(encoded[2][0], episode.target)
        for j, action in enumerate(actions):
            observation = episode.outcomes[action.id]
            after = apply_action(state, action, observation)
            after_scores = runtime.encode_state(episode, after)[2][0]
            gains.append(true_margin(after_scores, episode.target)-before)
            if observation.valid:
                # An exact duplicate may already be cached; a new observation was encoded above.
                future[0, j] = runtime.observations[observation.image_hash].detach().cpu()
                valid[0, j] = True
        actual = torch.tensor([gains], dtype=row['targets'].dtype)
        if actual.shape != row['targets'].shape or not torch.allclose(
                actual, row['targets'], atol=1e-4, rtol=1e-4):
            raise ValueError('reconstructed label differs')
        maximum_label = max(maximum_label, float((actual-row['targets']).abs().max()))
        enriched.append({**row, 'capture_group': episode.capture_group or episode.group,
                         'inputs': {**row['inputs'], 'future': future, 'future_valid': valid}})
    return enriched, dict(states=len(enriched), max_input_error=maximum_input,
                          max_label_error=maximum_label,
                          feature_space='frozen encode_observations: five ordered tokens',
                          privileged_inputs=['future', 'future_valid'])
