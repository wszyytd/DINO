"""All-state update comparison and privileged real-observation input diagnostic (CPU)."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import torch

from dinoexp.evidence_data import FeatureBank, episode_hashes, load_episodes
from dinoexp.evidence_labels import FrozenRuntime, fingerprint, generate_labels
from dinoexp.evidence_observation_data import attach_observations, paired_unique
from dinoexp.evidence_observation_diagnostics import CONDITIONS, fit_condition
from dinoexp.evidence_training import load_memory, save_json
from dinoexp.local_pairs import sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'source-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7])
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--include-development', action='store_true')
    parser.add_argument('--development-states-per-episode', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    if min(args.steps, args.threads, args.chunk_size, args.development_states_per_episode) < 1 or not (
            math.isfinite(args.lr) and args.lr > 0):
        parser.error('positive finite training settings required')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('seeds must be distinct and nonnegative')
    return args


def select_roles(source, episodes, include_development):
    if source['audit']['scope'] != 'pilot_location_split_identity_overlap_same_background_reference':
        raise ValueError('this diagnostic currently supports the recorded pilot location protocol only')
    roles = {}
    for role in ('repr', 'value', 'eval'):
        expected = set(source['audit']['roles'][role]['episodes'])
        roles[role] = [e for e in episodes if e.id in expected]
        if {e.id for e in roles[role]} != expected or not expected:
            raise ValueError('missing recorded role episodes: '+role)
    for i, left in enumerate(roles):
        for right in list(roles)[i+1:]:
            if ({e.id for e in roles[left]} & {e.id for e in roles[right]} or
                    {e.position for e in roles[left]} & {e.position for e in roles[right]} or
                    episode_hashes(roles[left]) & episode_hashes(roles[right])):
                raise ValueError('overlap between source roles')
    return roles['value'], roles['eval'] if include_development else [], roles['repr']


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    source_path = args.source_run/'run.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source['status'] != 'complete' or source['arguments']['score_mode'] != 'relative_softmax_v2':
        raise ValueError('need a completed relative_softmax_v2 source run')
    episodes, audit = load_episodes(args.data, args.cache)
    if audit['cache_manifest_sha256'] != source['audit']['cache_manifest_sha256']:
        raise ValueError('cache manifest changed')
    def source_files(hashes):
        return {'/'.join(Path(p.replace('\\', '/')).parts[-2:]): v for p, v in hashes.items()}
    if source_files(audit['source_hashes']) != source_files(source['audit']['source_hashes']):
        raise ValueError('capture metadata changed since source experiment')
    training, development, representation = select_roles(source, episodes, args.include_development)
    bank = FeatureBank(args.cache, episode_hashes(training+development),
                       grid=source['feature_config']['pooled_grid'])
    runtimes, branches, configs, immutable = {}, {}, {}, {str(source_path): sha256(source_path)}
    immutable[str(args.cache/'cache.json')] = sha256(args.cache/'cache.json')
    for name in ('alpha_0', 'alpha_0.1'):
        directory = args.source_run/name
        paths = [directory/'memory/last.pt', directory/'value/labels.pt', directory/'value/training.json']
        immutable.update({str(p): sha256(p) for p in paths})
        model, memory_meta = load_memory(paths[0], bank.config)
        if set(memory_meta['training_episodes']) != {e.id for e in representation}:
            raise ValueError('upstream representation used unexpected episodes')
        runtime = FrozenRuntime(model, bank, source['arguments']['score_temperature'],
                                source['arguments']['score_mode'])
        rows = torch.load(paths[1], map_location='cpu', weights_only=True)
        meta = json.loads(paths[2].read_text(encoding='utf-8'))
        expected = {e.id for e in training}
        if (set(meta['training_episodes']) != expected or {r['episode_id'] for r in rows} != expected or
                meta['upstream_fingerprint'] != runtime.fingerprint or any(
                    r['upstream_fingerprint'] != runtime.fingerprint for r in rows)):
            raise ValueError('training record provenance differs')
        runtimes[name], branches[name], configs[name] = runtime, rows, meta['model_config']
    branches, selection = paired_unique(branches)
    repository = Path(__file__).resolve().parent
    files = [Path(__file__), *(repository/'dinoexp').glob('evidence_*.py')]
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', arguments={k: str(v) if isinstance(v, Path) else v
                                             for k, v in vars(args).items()},
                  selection=selection, inputs_sha256=immutable, source_sha256={str(p): sha256(p) for p in files},
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                                   text=True).strip(),
                  scope='pilot development, identity overlap; future feature conditions privileged',
                  roles={'training': [e.id for e in training], 'development': [e.id for e in development]},
                  preparation={}, runs={})
    save_json(args.output/'run.json', result)
    upstream = {name: fingerprint(runtime.model) for name, runtime in runtimes.items()}
    try:
        dev_branches = {name: generate_labels(runtime, development,
                                             states_per_episode=args.development_states_per_episode, seed=7)
                        for name, runtime in runtimes.items()} if development else {}
        if development:
            dev_branches, result['development_selection'] = paired_unique(dev_branches)
        for name, runtime in runtimes.items():
            train_records, train_audit = attach_observations(runtime, training, branches[name])
            dev_records, dev_audit = attach_observations(runtime, development, dev_branches[name]) if development else ([], None)
            result['preparation'][name] = dict(training=train_audit, development=dev_audit)
            branch_dir = args.output/name
            branch_dir.mkdir()
            torch.save(dict(training=train_records, development=dev_records,
                            diagnostic_only=True, upstream_fingerprint=runtime.fingerprint), branch_dir/'records.pt')
            result['runs'][name] = {}
            for seed in args.seeds:
                modes = result['runs'][name][str(seed)] = {}
                for condition in CONDITIONS:
                    fitted = fit_condition(train_records, dev_records, configs[name],
                                           branch_dir/f'seed_{seed}'/condition, condition,
                                           args.steps, seed, args.lr, args.chunk_size)
                    # Detailed state predictions remain in each fit.json, not duplicated in run.json.
                    for role in ('training', 'development'):
                        if fitted[role]:
                            fitted[role] = {k: v for k, v in fitted[role].items() if k != 'rows'}
                    modes[condition] = fitted
                    save_json(args.output/'run.json', result)
                if (modes['direct_single']['initial_fingerprint'] != modes['direct_joint']['initial_fingerprint'] or
                        modes['observation_zero_joint']['initial_fingerprint'] != modes['observation_real_joint']['initial_fingerprint']):
                    raise ValueError('paired initialization differs')
            if fingerprint(runtime.model) != upstream[name] or any(p.grad is not None for p in runtime.model.parameters()):
                raise ValueError('upstream model changed')
        if any(sha256(Path(p)) != value for p, value in immutable.items()):
            raise ValueError('source artifact changed during experiment')
        from dinoexp.evidence_observation_report import render_report
        render_report(result, args.output)
        result['status'] = 'complete'
        save_json(args.output/'run.json', result)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_json(args.output/'run.json', result)
        raise
    print('Report: '+str(args.output/'report.md'), flush=True)


if __name__ == '__main__':
    main()
