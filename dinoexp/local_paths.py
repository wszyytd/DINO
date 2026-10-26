"""Find existing DINOv2 assets without downloading or creating output directories."""

import os
from pathlib import Path


def resolve_backbone_paths(repo, checkpoint, model, project_root, *, hub_dir=None, environ=None):
    env = os.environ if environ is None else environ
    project = Path(project_root).resolve()
    if model not in ('dinov2_vits14', 'dinov2_vitb14'):
        raise ValueError('unsupported DINOv2 backbone')
    if hub_dir is None:
        torch_home = env.get('TORCH_HOME')
        cache_home = Path(env.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))).expanduser()
        hub_dir = (Path(torch_home).expanduser() if torch_home else cache_home / 'torch') / 'hub'
    hub = Path(hub_dir).resolve()
    roots = [project, project.parent, project.parent / 'Cache', project.parent.parent / 'Cache']
    repo_candidates = [hub / 'facebookresearch_dinov2_main']
    if hub.is_dir():
        repo_candidates += sorted(hub.glob('facebookresearch_dinov2_*'))
    for root in roots:
        repo_candidates += [root / 'dinov2_repo', root / 'dinov2',
                            root / 'models/dinov2_repo', root / 'third_party/dinov2']
    name = model + '_pretrain.pth'
    weight_candidates = [hub / 'checkpoints' / name]
    for root in roots:
        weight_candidates += [root / name, root / 'models' / name,
                              root / 'dinov2_w' / name, root / 'checkpoints' / name]

    def select(explicit, key, candidates, is_valid, explanation):
        configured = explicit
        if configured is None:
            value = env.get(key)
            # Earlier instructions contained these examples. They are never usable paths.
            if value and '实际路径' not in value and '/path/to/' not in value.replace('\\', '/'):
                configured = value
        if configured is not None:
            target = Path(configured).expanduser().resolve()
            if not is_valid(target):
                raise ValueError(f'{key}: {explanation}: {target}')
            return target
        for candidate in dict.fromkeys(candidates):
            if is_valid(candidate):
                return candidate.resolve()
        searched = '\n  '.join(str(p) for p in dict.fromkeys(candidates))
        raise ValueError(f'Cannot locate {key} ({explanation}). Searched:\n  {searched}\n'
                         f'Specify its existing path using {key} or the corresponding CLI option.')

    selected_repo = select(repo, 'DINO_REPO', repo_candidates,
                           lambda p: p.is_dir() and (p / 'hubconf.py').is_file(),
                           'DINOv2 source directory must contain hubconf.py')
    selected_checkpoint = select(checkpoint, 'DINO_BACKBONE', weight_candidates,
                                 lambda p: p.is_file() and p.stat().st_size > 0,
                                 f'need existing nonempty official weights, normally {name}')
    return selected_repo, selected_checkpoint
