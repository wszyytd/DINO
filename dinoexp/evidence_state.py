"""Immutable real-observation histories; no network or simulator dependencies."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    id: str
    candidate_id: str
    image_hash: str | None
    pose: tuple
    quality: tuple = (1., 0., 0.)
    valid: bool = True
    source: str = ''
    reason: str = ''


@dataclass(frozen=True)
class Action:
    id: str
    candidate: int
    pose: tuple
    view: str
    cost: float = 1.


@dataclass(frozen=True)
class ReplayState:
    candidate_ids: tuple
    histories: tuple
    attempted: tuple = ()

    def __post_init__(self):
        if len(self.candidate_ids) < 2 or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('need at least two distinct candidates')
        if len(self.histories) != len(self.candidate_ids):
            raise ValueError('candidate/history count mismatch')
        for candidate, history in zip(self.candidate_ids, self.histories, strict=True):
            if any(o.candidate_id != candidate or not o.valid for o in history):
                raise ValueError('invalid candidate history')


def apply_action(state, action, observation, capacity=11):
    """Consume an attempt; exact replays and failed captures add no evidence."""
    if action.id in state.attempted:
        raise ValueError('action already attempted')
    if not 0 <= action.candidate < len(state.histories):
        raise ValueError('invalid candidate index')
    if observation.candidate_id != state.candidate_ids[action.candidate]:
        raise ValueError('observation belongs to another candidate')
    if not math.isfinite(action.cost) or action.cost <= 0:
        raise ValueError('action cost must be positive')
    histories = list(state.histories)
    old = histories[action.candidate]
    if observation.valid:
        if not observation.image_hash or len(observation.pose) != 5:
            raise ValueError('valid observation needs image hash and pose')
        repeated = any(o.id == observation.id or (
            o.image_hash == observation.image_hash and o.pose == observation.pose
        ) for o in old)
        if not repeated:
            if len(old) >= capacity:
                raise ValueError('memory capacity exceeded')
            histories[action.candidate] = (*old, observation)
    return ReplayState(state.candidate_ids, tuple(histories), (*state.attempted, action.id))


def true_margin(scores, target):
    values = [float(s) for s in scores]
    if len(values) < 2 or not 0 <= target < len(values) or not all(map(math.isfinite, values)):
        raise ValueError('need finite scores and a valid target')
    return values[target] - max(v for i, v in enumerate(values) if i != target)


def legal_actions(episode, state):
    attempted = set(state.attempted)
    return tuple(a for a in episode.actions if a.id not in attempted)
