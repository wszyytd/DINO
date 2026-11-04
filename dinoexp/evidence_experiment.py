"""Explicit provenance checks for downstream-only experiments."""
from .evidence_training import load_memory


def load_representation(path, feature_config, episodes, alpha, device='cpu'):
    model, metadata = load_memory(path, feature_config, device)
    if metadata.get('status') != 'complete' or metadata.get('stage') != 'representation':
        raise ValueError('upstream representation checkpoint is not complete')
    if metadata.get('alpha') != alpha:
        raise ValueError('upstream auxiliary weight differs from requested alpha')
    if set(metadata.get('training_episodes', [])) != {e.id for e in episodes}:
        raise ValueError('upstream training episodes differ from representation role')
    if set(metadata.get('training_groups', [])) != {e.group for e in episodes}:
        raise ValueError('upstream training groups differ from representation role')
    return model, metadata
