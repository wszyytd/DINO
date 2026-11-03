"""Equal-cost replay policies; truth access is explicit and confined to the oracle/evaluator."""

import random

import numpy as np
import torch

from .evidence_labels import policy_inputs
from .evidence_state import apply_action, legal_actions, true_margin


def coverage_action(actions, state):
    def distance(action):
        history = state.histories[action.candidate]
        if not history:
            return float('inf')
        return min(float(np.linalg.norm(np.asarray(action.pose)-np.asarray(o.pose)))
                   for o in history)
    return max(actions, key=distance)


@torch.no_grad()
def rollout(runtime, episode, policy, budget=10, seed=7, value=None,
            stop_gap=None, stop_score=-1., cost_weight=0., stop_on_nonpositive=False):
    if budget < 0 or policy not in ('random', 'fixed', 'coverage', 'learned', 'oracle'):
        raise ValueError('invalid budget or policy')
    if policy == 'learned' and value is None:
        raise ValueError('learned policy needs a value network')
    if stop_gap is not None and stop_gap <= 0:
        raise ValueError('confirmation gap must be positive')
    rng, state = random.Random(seed), episode.initial
    steps, failures, cost = [], 0, 0.
    status, reason, last_action = 'UNRESOLVED', 'budget', None
    for step in range(budget+1):
        q, memories, scores, counts = runtime.encode_state(episode, state)
        scores_list = scores[0].cpu().tolist()
        ordered = sorted(scores_list, reverse=True)
        margin = true_margin(scores_list, episode.target)
        steps.append({'attempt': step, 'action': last_action, 'scores': scores_list,
                      'correct_rank': margin > 1e-8, 'true_margin': margin,
                      'counts': counts[0].cpu().tolist(), 'cost': cost})
        if stop_gap is not None and min(counts[0]).item() > 0 and (
                ordered[0]-ordered[1] >= stop_gap and ordered[0] >= stop_score):
            status, reason = 'CONFIRMED', 'uncalibrated_diagnostic_threshold'
            break
        if step == budget:
            break
        actions = legal_actions(episode, state)
        if not actions:
            reason = 'no_actions'
            break
        if policy == 'random':
            action = rng.choice(actions)
        elif policy == 'fixed':
            action = min(actions, key=lambda a: (a.id.partition('_')[2] or a.id, a.candidate))
        elif policy == 'coverage':
            action = coverage_action(actions, state)
        elif policy == 'learned':
            predicted = value(**policy_inputs(q, memories, scores, counts, actions))[0]
            utilities = predicted - torch.tensor([a.cost*cost_weight for a in actions],
                                                 device=predicted.device)
            if stop_on_nonpositive and utilities.max().item() <= 0:
                reason = 'no_positive_predicted_net_gain'
                break
            action = actions[int(utilities.argmax())]
        else:
            # Only this branch receives privileged after-observations and target identity.
            utilities = []
            for candidate_action in actions:
                after = apply_action(state, candidate_action, episode.outcomes[candidate_action.id])
                after_scores = runtime.encode_state(episode, after)[2][0]
                utilities.append(true_margin(after_scores, episode.target)-margin
                                 -cost_weight*candidate_action.cost)
            action = actions[int(np.argmax(utilities))]
        observation = episode.outcomes[action.id]
        state = apply_action(state, action, observation)
        failures += int(not observation.valid)
        cost += action.cost
        last_action = action.id
    return {'episode_id': episode.id, 'group': episode.group, 'policy': policy, 'seed': seed,
            'status': status, 'reason': reason, 'attempts': len(state.attempted),
            'failed_attempts': failures, 'cost': cost, 'steps': steps,
            'confirmed_correct': status == 'CONFIRMED' and steps[-1]['correct_rank'],
            'stopping_calibrated': False, 'confirmation_enabled': stop_gap is not None}


def summarize_rollouts(traces, budget):
    if not traces:
        raise ValueError('cannot summarize empty rollouts')
    curves = []
    for step in range(budget+1):
        curves.append(float(np.mean([t['steps'][min(step, len(t['steps'])-1)]['correct_rank']
                                     for t in traces])))
    confirmed_correct = sum(t['confirmed_correct'] for t in traces)
    confirmed = sum(t['status'] == 'CONFIRMED' for t in traces)
    return {'episodes': len({t['episode_id'] for t in traces}), 'rollouts': len(traces),
            'independent_groups': len({t['group'] for t in traces}),
            'rank_accuracy_by_budget': curves,
            'correct_confirmation_rate': confirmed_correct/len(traces),
            'false_confirmation_rate': (confirmed-confirmed_correct)/len(traces),
            'unresolved_rate': 1-confirmed/len(traces),
            'mean_attempts': float(np.mean([t['attempts'] for t in traces])),
            'mean_cost': float(np.mean([t['cost'] for t in traces])),
            'penalized_confirmation_attempts': float(np.mean([
                t['attempts'] if t['confirmed_correct'] else budget for t in traces])),
            'failed_attempts': sum(t['failed_attempts'] for t in traces),
            'scope': 'fixed-budget ranking; stopping uncalibrated or disabled'}
