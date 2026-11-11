"""Read-only fixed-checkpoint causal diagnostics. Execute on the training server."""
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

from .evidence_data import load_evidence_cache
from .evidence_diagnostic_data import enrich, load_source_metadata
from .evidence_diagnostic_reports import (
    metric_counts, score_description, write_csv, write_json, write_reports,
)
from .evidence_memory import EvidenceMatcher
from .evidence_training import batches, make_routes, pack, partition_scenes
from .local_features import resolve_device
from .local_pairs import sha256


def protect_output(output, inputs):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) for p in inputs):
        raise ValueError('diagnosis output must be new and outside run/cache/data')


def clone_state(state):
    return None if state is None else {key: value.clone() for key, value in state.items()}


def swap_slots(state):
    """Rotate contents of seen candidates only, keeping flags/counts and source immutable."""
    result = clone_state(state)
    eligible = state['seen'].sum(-1) >= 2
    for b in torch.where(eligible)[0].tolist():
        ids = torch.where(state['seen'][b])[0]
        result['slots'][b, ids] = state['slots'][b, ids.roll(1)]
    return result, eligible


class UpdateObserver:
    """External, non-mutating forward hooks. No changes to checkpoint/model interfaces."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.gates = self.proposal_delta = None

    def __enter__(self):
        if self.model.mode == 'memory':
            def gate_hook(module, inputs, output):
                gate = output.detach().sigmoid().flatten(1)
                self.gates = list(zip(gate.mean(1).cpu().tolist(), gate.max(1).values.cpu().tolist(), strict=True))

            def updater_hook(module, inputs, output):
                self.proposal_delta = (output[0].detach()-inputs[0].detach()).flatten(1).norm(dim=1).cpu().tolist()
            self.handles = [self.model.gate.register_forward_hook(gate_hook),
                            self.model.updater.register_forward_hook(updater_hook)]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()

    def update(self, observation, valid, state):
        self.gates = self.proposal_delta = None
        b, k = observation.shape[:2]
        before = state if state is not None and self.model.mode != 'current' else self.model.initial_state(
            b, k, observation.device, observation.dtype)
        after = self.model.update(observation, valid, state)
        change = (after['slots']-before['slots']).flatten(2).norm(dim=2)
        relative = change / before['slots'].flatten(2).norm(dim=2).clamp_min(1e-8)
        details = [[dict(written=False, gate_applied=False) for _ in range(k)] for _ in range(b)]
        for n, (bi, ki) in enumerate(torch.nonzero(valid.any(-1)).tolist()):
            first = not bool(before['seen'][bi, ki])
            record = dict(written=True, first_observation=first,
                          gate_applied=self.model.mode == 'memory' and not first,
                          state_change_norm=float(change[bi, ki]),
                          relative_state_change=float(relative[bi, ki]))
            if self.gates is not None:
                record.update(gate_mean=self.gates[n][0], gate_max=self.gates[n][1],
                              proposal_delta_norm=self.proposal_delta[n])
            details[bi][ki] = record
        return after, details


def _context(route, scene, step, visible, ever, target=None):
    target = route['target'] if target is None else target
    ids = scene['objects']; prefix = route['frames'][:step+1]
    path = json.dumps([scene['scene_id'], route['frames']], separators=(',', ':'))
    prefix_key = json.dumps([scene['scene_id'], prefix], separators=(',', ':'))
    return dict(scene_index=route['scene'], scene_id=scene['scene_id'],
                task_id=route['diagnostic_task_index'],
                path_id=hashlib.sha256(path.encode()).hexdigest()[:20],
                prefix_id=hashlib.sha256(prefix_key.encode()).hexdigest()[:20],
                step=step, frame_index=route['frames'][step], frame_id=scene['state_ids'][route['frames'][step]],
                prefix_frames=prefix, candidate_ids=ids, target_index=target, target_id=ids[target],
                first_seen_step=[next((j for j, frame in enumerate(prefix) if bool(scene['valid'][frame, k].any())), None)
                                 for k in range(len(ids))],
                visible=visible, ever_seen=ever, target_seen=bool(ever[target]),
                target_visible=bool(visible[target]), both_visible=bool(visible[target] and sum(visible) >= 2),
                is_final=step+1 == len(route['frames']))


def _prediction(logits, target, ever, ids):
    result = score_description(logits, target, ever)
    result['predicted_id'] = ids[result['predicted_index']] if result['predicted_index'] < len(ids) else 'Null'
    return result


@torch.inference_mode()
def replay_batches(model, routes, scenes, device, batch_size, controls=True):
    """Original batch order/shape and legacy labels; interventions fork, never rewrite truth."""
    model.eval()
    annotated = [dict(route, diagnostic_task_index=i) for i, route in enumerate(routes)]
    with torch.inference_mode(), UpdateObserver(model) as observer:
        for rows in batches(annotated, scenes, batch_size):
            x, valid, ref, rv, target = pack(rows, scenes, device)
            state = None
            ever = torch.zeros(x.shape[0], x.shape[2], dtype=torch.bool, device=device)
            any_single_correct = [False] * len(rows)
            result = dict(normal=[], controls=[], switches=[])
            for t in range(x.shape[1]):
                previous = state
                ever_before = ever.clone()
                before_scores = model.score(ref, rv, previous).cpu().tolist() if previous is not None and controls else None
                state, update_details = observer.update(x[:, t], valid[:, t], previous)
                normal_scores = model.score(ref, rv, state).cpu().tolist()
                visible = valid[:, t].any(-1); ever = ever | visible
                visible_list, ever_list = visible.cpu().tolist(), ever.cpu().tolist()
                reset_state = model.update(x[:, t], valid[:, t], None)
                reset_scores = model.score(ref, rv, reset_state).cpu().tolist()
                post_contexts = []
                for i, route in enumerate(rows):
                    scene = scenes[route['scene']]; ti = route['target']
                    context = _context(route, scene, t, visible_list[i], ever_list[i])
                    post_contexts.append(context)
                    predicted = _prediction(normal_scores[i], ti, ever_list[i], scene['objects'])
                    reset = _prediction(reset_scores[i], ti, ever_list[i], scene['objects'])
                    any_single_correct[i] |= visible_list[i][ti] and reset['predicted_index'] == ti
                    result['normal'].append(dict(**context, **predicted, update=update_details[i],
                        before_update=_prediction(before_scores[i], ti, ever_before[i].cpu().tolist(), scene['objects'])
                            if before_scores is not None else None,
                        state_seen_before=previous['seen'][i].cpu().tolist() if previous is not None else [False]*len(ever_list[i]),
                        state_seen_after=state['seen'][i].cpu().tolist(),
                        state_count_after=state['count'][i].cpu().tolist(),
                        any_single_frame_correct_so_far=bool(any_single_correct[i]),
                        correct_without_any_correct_single=bool(predicted['correct'] and ever_list[i][ti] and not any_single_correct[i])))
                    if controls and model.mode == 'memory':
                        result['controls'].append(dict(**context, condition='reset', stage='after_current',
                                                       baseline=predicted, intervention=reset))
                if controls and model.mode == 'memory' and previous is not None:
                    swapped, eligible = swap_slots(previous)
                    swapped_before = model.score(ref, rv, swapped).cpu().tolist()
                    swapped_after = model.update(x[:, t], valid[:, t], clone_state(swapped))
                    swapped_after_scores = model.score(ref, rv, swapped_after).cpu().tolist()
                    for i, route in enumerate(rows):
                        ti = route['target']; ids = scenes[route['scene']]['objects']
                        normal = _prediction(normal_scores[i], ti, ever_list[i], ids)
                        if bool(previous['seen'][i, ti]):
                            result['controls'].append(dict(**post_contexts[i], condition='pause', stage='after_current',
                                baseline=normal, intervention=_prediction(before_scores[i], ti, ever_list[i], ids)))
                        if bool(eligible[i]) and bool(ever_before[i, ti]):
                            old_ever = ever_before[i].cpu().tolist()
                            old_visible = valid[i, t-1].any(-1).cpu().tolist()
                            pre_context = _context(route, scenes[route['scene']], t-1, old_visible, old_ever)
                            result['controls'].append(dict(**pre_context, condition='swap_slots', stage='before_current',
                                next_frame_id=scenes[route['scene']]['state_ids'][route['frames'][t]],
                                baseline=_prediction(before_scores[i], ti, old_ever, ids),
                                intervention=_prediction(swapped_before[i], ti, old_ever, ids)))
                            result['controls'].append(dict(**post_contexts[i], condition='swap_slots', stage='after_current',
                                baseline=normal, intervention=_prediction(swapped_after_scores[i], ti, ever_list[i], ids)))
                if controls:
                    # Same state, no write between references. One row per physical path/query;
                    # paired target=1 task already repeats this state, so export only target=0 paths.
                    for query in range(x.shape[2]):
                        refs = torch.stack([scenes[r['scene']]['references'][query] for r in rows]).to(device)
                        rvs = torch.stack([scenes[r['scene']]['reference_valid'][query] for r in rows]).to(device)
                        scores = model.score(refs, rvs, state).cpu().tolist()
                        for i, route in enumerate(rows):
                            if route['target'] == 0 and ever_list[i][query]:
                                scene = scenes[route['scene']]
                                context = _context(route, scene, t, visible_list[i], ever_list[i], query)
                                result['switches'].append(dict(**context, **_prediction(scores[i], query, ever_list[i], scene['objects'])))
            yield result


@torch.inference_mode()
def repeat_trial(model, base_state, x, valid, reference, reference_valid, target, ever, count, next_observation=None):
    before_scores = model.score(reference, reference_valid, base_state)[0].cpu().tolist()
    before = score_description(before_scores, target, ever)
    repeated = clone_state(base_state)
    for _ in range(count):
        repeated = model.update(x, valid, repeated)
    after = score_description(model.score(reference, reference_valid, repeated)[0].cpu().tolist(), target, ever)
    dedup = score_description(model.score(reference, reference_valid, clone_state(base_state))[0].cpu().tolist(), target, ever)
    change = max(abs(a-b) for a, b in zip(before['probabilities'], after['probabilities'], strict=True))
    dedup_change = max(abs(a-b) for a, b in zip(before['probabilities'], dedup['probabilities'], strict=True))
    if dedup_change > 1e-6:
        raise ValueError('unchanged-state repeat read is not numerically stable')
    result = dict(extra_reads=count, before=before, after=after, max_probability_change=change,
                  target_probability_change=after['target_probability']-before['target_probability'],
                  choice_flipped=before['predicted_index'] != after['predicted_index'],
                  dedup_max_probability_change=dedup_change)
    if next_observation is not None:
        nx, nv = next_observation
        next_ever = [a or b for a, b in zip(ever, nv[0].any(-1).cpu().tolist(), strict=True)]
        clean_next = model.update(nx, nv, clone_state(base_state))
        repeat_next = model.update(nx, nv, clone_state(repeated))
        result['after_next'] = dict(
            without_repeats=score_description(model.score(reference, reference_valid, clean_next)[0].cpu().tolist(), target, next_ever),
            with_repeats=score_description(model.score(reference, reference_valid, repeat_next)[0].cpu().tolist(), target, next_ever))
    return result


def choose_repeat_anchors(routes, scenes, contexts, per_group):
    """Bounded deterministic sampling using labels only as diagnostic selectors."""
    anchors, counts, used = [], defaultdict(int), set()
    for route_index, route in enumerate(routes):
        scene = scenes[route['scene']]; context = contexts[route['scene']]; target = route['target']
        prefix_quality = []
        for t, frame in enumerate(route['frames']):
            visible = bool(scene['valid'][frame, target].any())
            quality = context['frames'][frame]['candidates'][target]['quality']
            if t == 0:
                kind = 'empty'
            elif prefix_quality == ['usable']:
                kind = 'one_usable'
            elif 'usable' in prefix_quality and any(q != 'usable' for q in prefix_quality):
                kind = 'mixed'
            else:
                kind = None
            key = (scene['scene_id'], scene['objects'][target], kind)
            prefix_key = (scene['scene_id'], target, tuple(route['frames'][:t+1]))
            if visible and kind and counts[key] < per_group and prefix_key not in used:
                digest = scene['image_hashes'][frame]
                later = next((s for s in range(t+1, len(route['frames']))
                              if scene['image_hashes'][route['frames'][s]] != digest
                              and bool(scene['valid'][route['frames'][s], target].any())
                              and context['frames'][route['frames'][s]]['candidates'][target]['quality'] == 'usable'), None)
                anchors.append(dict(route_index=route_index, step=t, history_type=kind, next_step=later))
                counts[key] += 1; used.add(prefix_key)
            if visible:
                prefix_quality.append(quality)
    return anchors


@torch.inference_mode()
def repeat_diagnostics(model, routes, scenes, contexts, device, per_group, counts):
    model.eval()
    anchors = choose_repeat_anchors(routes, scenes, contexts, per_group)
    results = []
    for anchor in anchors:
        route = dict(routes[anchor['route_index']], diagnostic_task_index=anchor['route_index'])
        scene = scenes[route['scene']]
        x, valid, ref, rv, target = pack([route], scenes, device)
        state = None
        for t in range(anchor['step']+1):
            state = model.update(x[:, t], valid[:, t], state)
        t = anchor['step']; ever = valid[:, :t+1].any(1).any(-1)[0].cpu().tolist()
        context = _context(route, scene, t, valid[0, t].any(-1).cpu().tolist(), ever)
        later = anchor['next_step']
        next_observation = (x[:, later], valid[:, later]) if later is not None else None
        for count in counts:
            result = repeat_trial(model, state, x[:, t], valid[:, t], ref, rv,
                                  int(target[0]), ever, count, next_observation)
            results.append(dict(**context, **result, history_type=anchor['history_type'],
                                next_step=later, next_frame_id=scene['state_ids'][route['frames'][later]] if later is not None else None,
                                next_view_selection='first_later_distinct_rgb_target_usable',
                                next_view_is_known_complementary=False,
                                repeat_batch_size=1))
    return results, anchors


def _validate_routes(routes, scenes, allowed):
    allowed_map = {i: set(ids) for i, ids in allowed}
    for route in routes:
        i = route['scene']; frames = route['frames']
        if i not in allowed_map or not frames or not set(frames) <= allowed_map[i]:
            raise ValueError('diagnostic route crosses split or references invalid frames')
        if not 0 <= route['target'] < len(scenes[i]['objects']):
            raise ValueError('diagnostic route has unknown reference')


def _git_provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*parts):
        result = subprocess.run(['git', *parts], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(commit=git('rev-parse', 'HEAD'), tracked_changes=git('status', '--porcelain', '--untracked-files=no'))


def run_diagnostics(args):
    run_dir, cache_dir, root, output = (Path(p).resolve() for p in
                                      (args.run, args.cache, args.data_root, args.output))
    protect_output(output, [run_dir, cache_dir, root])
    if args.threads < 1 or args.max_routes < 0 or args.repeat_samples < 0 or args.examples < 0:
        raise ValueError('invalid diagnostic limits')
    if not args.repeat_counts or min(args.repeat_counts) < 1 or len(set(args.repeat_counts)) != len(args.repeat_counts):
        raise ValueError('repeat counts must be distinct positive integers')
    for name in ('modes', 'seeds', 'partitions'):
        values = getattr(args, name)
        if values and len(values) != len(set(values)):
            raise ValueError('duplicate diagnostic selections: ' + name)
    run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    if run['status'] != 'complete':
        raise ValueError('source training run is incomplete')
    if sha256(cache_dir / 'cache.json') != run['cache_manifest_sha256']:
        raise ValueError('cache manifest differs from training run')
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device, torch)
    meta, scenes = load_evidence_cache(cache_dir)
    if meta['contract'] != run['feature_contract']:
        raise ValueError('run/cache feature contracts differ')
    partitions, scope = partition_scenes(scenes, meta)
    probe = json.loads((run_dir / 'probe_routes.json').read_text(encoding='utf-8'))
    fit = make_routes(scenes, partitions['train'], run['args']['routes'], run['args']['length'], 902)
    all_routes = dict(train=fit, probe=probe)
    recheck_path = run_dir / 'checkpoint_recheck.json'
    recheck = json.loads(recheck_path.read_text(encoding='utf-8')) if recheck_path.is_file() else dict(results=[])
    fit_checks = {(r['mode'], r['seed']): r for r in recheck['results']}
    selected = [r for r in run['results'] if r['mode'] in args.modes and
                (not args.seeds or r['seed'] in args.seeds)]
    if not selected:
        raise ValueError('no checkpoints selected')
    requested = {(mode, seed) for mode in args.modes for seed in (args.seeds or run['args']['seeds'])}
    if requested != {(r['mode'], r['seed']) for r in selected}:
        raise ValueError('requested mode/seed not present in source run')
    for split in args.partitions:
        _validate_routes(all_routes[split], scenes, partitions[split])
    print('Checking original RGB, masks, references and cache identities...', flush=True)
    contexts, quality = load_source_metadata(root, scenes, meta)
    output.mkdir(parents=True)
    source_code = [Path(__file__), Path(__file__).with_name('evidence_diagnostic_data.py'),
                   Path(__file__).with_name('evidence_diagnostic_reports.py'),
                   Path(__file__).with_name('evidence_memory.py'),
                   Path(__file__).with_name('evidence_training.py'),
                   Path(__file__).resolve().parents[1] / 'diagnose_evidence_memory.py']
    manifest = dict(status='running', kind='evidence_diagnostics_v1', scope=scope,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_run_sha256=sha256(run_dir / 'run.json'),
                    source_routes_sha256=sha256(run_dir / 'probe_routes.json'),
                    cache_manifest_sha256=sha256(cache_dir / 'cache.json'), feature_contract=meta['contract'],
                    code=_git_provenance(), code_sha256={p.name: sha256(p) for p in source_code},
                    torch_version=str(torch.__version__), device=str(device), cuda_version=torch.version.cuda,
                    source_training_device=run['args']['device'],
                    partial=bool(args.max_routes), model_forward_validation='pending',
                    no_training=True, checkpoint_sha256={}, selected_runs=[(r['mode'], r['seed']) for r in selected])
    if device.type == 'cuda':
        manifest['device_name'] = torch.cuda.get_device_name(device)
    write_json(output / 'diagnosis_manifest.json', manifest)
    write_csv(output / 'candidate_quality.csv', quality)
    mismatch = sum(r['visible_gt'] != r['legacy_visible'] for r in quality)
    (output / 'data_audit.md').write_text(
        '# 数据来源检查\n\n原始 RGB、参考、掩码哈希已核对；这不替代目视身份核查。\n\n'
        f'- 场景数：{len(scenes)}\n- 帧候选记录：{len(quality)}\n'
        f'- 原始掩码可见与缓存有效性不一致：{mismatch}\n'
        '- 物理可见定义：原始实例掩码存在正像素；旧口径：缓存 valid.any()。\n'
        '- HTML 中提供有限样例的原图、参考、独立掩码，仍需人工核对身份。\n', encoding='utf-8')
    normal, controls, switches, repeats, checks = [], [], [], [], []
    try:
        for source in selected:
            mode, seed = source['mode'], source['seed']
            path = run_dir / f'{mode}_seed{seed}' / 'last.pt'
            digest = sha256(path)
            historical_check = fit_checks.get((mode, seed))
            if historical_check and historical_check['checkpoint_sha256'] != digest:
                raise ValueError('checkpoint differs from saved recheck')
            cp = torch.load(path, map_location='cpu', weights_only=True)
            if (cp['kind'] != 'oracle_evidence_memory_v1' or cp['feature_contract'] != meta['contract'] or
                    cp['config']['mode'] != mode or cp['seed'] != seed or cp['epoch'] != run['args']['epochs']):
                raise ValueError('checkpoint configuration/provenance mismatch')
            manifest['checkpoint_sha256'][f'{mode}_seed{seed}'] = digest
            model = EvidenceMatcher(**cp['config']).to(device).eval()
            model.load_state_dict(cp['model_state'], strict=True)
            for split in args.partitions:
                routes = all_routes[split]
                if args.max_routes:
                    routes = routes[:args.max_routes]
                write_json(output / f'{split}_routes.json', routes)
                tags = dict(mode=mode, seed=seed, split=split)
                print(f'Diagnosing {mode} seed={seed} split={split} tasks={len(routes)}', flush=True)
                part = []
                for batch in replay_batches(model, routes, scenes, device, run['args']['batch_size']):
                    for destination, key in ((part, 'normal'), (controls, 'controls'), (switches, 'switches')):
                        for row in batch[key]:
                            destination.append(enrich(dict(**row, **tags), contexts))
                observed = metric_counts(part)
                expected = source['metrics'] if split == 'probe' else historical_check['fit'] if historical_check else None
                checked = not args.max_routes and expected is not None
                match = observed == expected if checked else None
                checks.append(dict(**tags, verified=bool(checked and match), metrics_match=match,
                                   partial=bool(args.max_routes), expected=expected, observed=observed))
                if checked and not match:
                    write_json(output / 'replay_checks.json', checks)
                    from .evidence_diagnostic_reports import write_jsonl
                    write_jsonl(output / 'mismatched_replay_steps.jsonl', part)
                    raise ValueError(f'normal replay differs: {mode}/{seed}/{split}; source reports left unchanged')
                normal.extend(part)
                repeated, anchors = repeat_diagnostics(model, routes, scenes, contexts, device,
                                                       args.repeat_samples, args.repeat_counts)
                write_json(output / f'{mode}_seed{seed}_{split}_repeat_selection.json', dict(
                    per_history_type_per_reference_per_scene=args.repeat_samples,
                    history_types=['empty', 'one_usable', 'mixed'], anchors=anchors))
                repeats.extend(dict(**r, **tags) for r in repeated)
            del model
        write_reports(output, root, normal, controls, switches, repeats, checks, args.examples)
        manifest.update(status='complete', model_forward_validation='server_run_completed',
                        all_selected_replays_verified=bool(checks) and all(r['verified'] for r in checks),
                        counts=dict(steps=len(normal), controls=len(controls), reference_switches=len(switches), repeats=len(repeats)))
        write_json(output / 'diagnosis_manifest.json', manifest)
        print('Diagnosis complete:', output, flush=True)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        write_json(output / 'diagnosis_manifest.json', manifest)
        raise
