"""Audit saved training labels against photos, then fit 1/4/16 fixed states locally."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import time

import torch

from dinoexp.core import checked_path
from dinoexp.evidence_data import FeatureBank, episode_hashes, load_episodes
from dinoexp.evidence_diagnostic_report import audit_summary, render_audit, render_curves, render_summary
from dinoexp.evidence_diagnostics import (
    fixed_fit, input_audit, paired_selection, reconstruct_state, transition_result,
)
from dinoexp.evidence_labels import CLSRuntime, FrozenRuntime, policy_inputs
from dinoexp.evidence_state import apply_action, legal_actions
from dinoexp.evidence_training import load_memory, save_json
from dinoexp.local_pairs import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1, 4, 16])
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    if args.steps < 1 or args.threads < 1 or args.lr <= 0 or min(args.sizes) < 1:
        parser.error('steps/threads/lr/sizes must be positive')
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    source = json.loads((args.source_run/'run.json').read_text(encoding='utf-8'))
    if source['status'] != 'complete' or source['arguments']['score_mode'] != 'relative_softmax_v2':
        raise ValueError('need a completed v2 source run')
    expected = set(source['audit']['roles']['value']['episodes'])
    all_episodes, data_audit = load_episodes(args.data, args.cache)
    if data_audit['cache_manifest_sha256'] != source['audit']['cache_manifest_sha256']:
        raise ValueError('source feature cache manifest differs')
    episodes = [e for e in all_episodes if e.id in expected]
    if {e.id for e in episodes} != expected or {e.position for e in episodes} != {'64'}:
        raise ValueError('diagnostic only supports the recorded value-training position 64')
    bank = FeatureBank(args.cache, episode_hashes(episodes), grid=source['feature_config']['pooled_grid'])
    runtimes = {'cls_mean': CLSRuntime(bank)}
    branches, configs, hashes = {}, {}, {}
    for name in ('alpha_0', 'alpha_0.1'):
        directory = args.source_run/name
        memory = directory/'memory/last.pt'
        labels = directory/'value/labels.pt'
        model, _ = load_memory(memory, bank.config)
        runtime = FrozenRuntime(model, bank, source['arguments']['score_temperature'],
                                source['arguments']['score_mode'])
        rows = torch.load(labels, map_location='cpu', weights_only=True)
        meta = json.loads((directory/'value/training.json').read_text(encoding='utf-8'))
        if any(r['episode_id'] not in expected or r['upstream_fingerprint'] != runtime.fingerprint
               for r in rows) or set(meta['training_episodes']) != expected:
            raise ValueError('labels are outside training role or have a wrong upstream fingerprint')
        runtimes[name], branches[name], configs[name] = runtime, rows, meta['model_config']
        hashes[name] = {'memory': sha256(memory), 'labels': sha256(labels)}
    selection = paired_selection(list(branches.values()), args.sizes, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parent
    code_files = [Path(__file__), *(repository/'dinoexp').glob('evidence_*.py')]
    result = dict(status='running', scope='position 64 training-only diagnostic',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  selection=selection, inputs_sha256=hashes,
                  feature_config=bank.config, data_audit=data_audit,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                                   text=True).strip(),
                  source_sha256={str(p): sha256(p) for p in code_files}, fits={})
    save_json(args.output/'run.json', result)
    by_episode = {e.id: e for e in episodes}
    states, audit_rows = {}, []
    max_label_error = {name: 0. for name in branches}
    max_input_error = {name: 0. for name in branches}
    first = next(iter(branches.values()))
    try:
        with torch.no_grad():
            for index in selection['unique_indices']:
                record = first[index]
                episode = by_episode[record['episode_id']]
                state = reconstruct_state(episode, record)
                actions = legal_actions(episode, state)
                with (args.data/episode.id/'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
                    ref = next(r for r in csv.DictReader(stream) if r['role'] == 'reference')
                reference = checked_path(args.data/episode.id, ref['crop'])
                if sha256(reference) != episode.reference_hash:
                    raise ValueError('reference source image differs')
                states[str(index)] = dict(episode_id=episode.id, target=episode.target,
                                         candidate_ids=episode.candidate_ids, reference=str(reference),
                                         history=[[{'id': o.id, 'source': o.source} for o in h]
                                                  for h in state.histories])
                before = {}
                for name, runtime in runtimes.items():
                    encoded = runtime.encode_state(episode, state)
                    before[name] = encoded[2][0].tolist()
                    if name in branches:
                        current = policy_inputs(*encoded, actions)
                        saved = branches[name][index]['inputs']
                        for key, value in current.items():
                            error = float((value.cpu()-saved[key]).abs().max())
                            max_input_error[name] = max(max_input_error[name], error)
                            if not torch.allclose(value.cpu(), saved[key], atol=5e-4, rtol=5e-4):
                                raise ValueError(f'reconstructed inputs differ: {name}/{index}/{key}')
                for j, action in enumerate(actions):
                    observation = episode.outcomes[action.id]
                    after = apply_action(state, action, observation)
                    row = dict(record_index=index, episode_id=episode.id, action_id=action.id,
                               view=action.view, candidate_id=episode.candidate_ids[action.candidate],
                               source=observation.source, valid=observation.valid, models={})
                    for name, runtime in runtimes.items():
                        scores = runtime.encode_state(episode, after)[2][0].tolist()
                        value = transition_result(before[name], scores, episode.target)
                        row['models'][name] = value
                        if name in branches:
                            target = float(branches[name][index]['targets'][0, j])
                            error = abs(target-value['gain'])
                            max_label_error[name] = max(max_label_error[name], error)
                            if error > 5e-4:
                                raise ValueError(f'recomputed label differs: {name}/{index}/{j}')
                    audit_rows.append(row)
                if len(states) % 10 == 0:
                    print(f'Audit: {len(states)}/{len(selection["unique_indices"])} states', flush=True)
        audit = dict(states=states, rows=audit_rows, summary=audit_summary(audit_rows),
                     max_label_error=max_label_error, max_input_error=max_input_error,
                     input_diagnostics={name: input_audit(rows) for name, rows in branches.items()})
        save_json(args.output/'audit.json', audit)
        previews = render_audit(audit_rows, states, args.output)
        result.update(audit_summary=audit['summary'], max_label_error=max_label_error,
                      max_input_error=max_input_error, preview_images=previews,
                      selected_episode_counts={size: dict(Counter(first[i]['episode_id'] for i in indices))
                                               for size, indices in selection['subsets'].items()})
        save_json(args.output/'run.json', result)
        print(f'Audit complete: {len(audit_rows)} transitions, {previews} previews', flush=True)
        for name, records in branches.items():
            result['fits'][name] = {}
            for size, indices in selection['subsets'].items():
                fit = fixed_fit([records[i] for i in indices], configs[name], args.output/name/f'n_{size}',
                                steps=args.steps, seed=args.seed, lr=args.lr)
                # The per-fit file retains every update/prediction; main manifest stays compact.
                result['fits'][name][size] = {k: v for k, v in fit.items()
                                             if k not in ('updates', 'predictions', 'targets', 'state_ids')}
                save_json(args.output/'run.json', result)
        render_curves(result['fits'], args.output)
        for name in branches:
            directory = args.source_run/name
            if hashes[name] != {'memory': sha256(directory/'memory/last.pt'),
                                'labels': sha256(directory/'value/labels.pt')}:
                raise RuntimeError('source files changed during diagnostic')
        result.update(status='complete', seconds=time.monotonic()-start)
        save_json(args.output/'run.json', result)
        render_summary(result, args.output)
        print(f'Complete: {args.output}', flush=True)
    except Exception as error:
        result.update(status='failed', error=str(error), seconds=time.monotonic()-start)
        save_json(args.output/'run.json', result)
        raise


if __name__ == '__main__':
    main()
