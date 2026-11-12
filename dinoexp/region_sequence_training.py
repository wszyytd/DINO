"""CUDA-only causal sequence training and paired history diagnostics."""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evidence_memory import EvidenceMatcher
from .region_ab_data import digest_json, read_json, sha, write_json
from .region_ab_prepare import cuda_device
from .region_sequence_data import load_prepared, sequence_arrays, source_record, training_weights
from .region_sequence_metrics import control_summary, summarize

METHODS = ('last', 'mean', 'memory')
NETWORK_FIELDS = ('observation', 'valid', 'current_valid', 'reference', 'reference_valid')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parameter_hash(model, common=False):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if common and name.startswith(('updater.', 'gate.')):
            continue
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model(cfg, seed, device, method):
    if method not in METHODS or torch.device(device).type != 'cuda':
        raise ValueError('known sequence method and authorized CUDA device required')
    seed_all(seed)
    master = EvidenceMatcher(dim=256, heads=cfg['heads'], slots=cfg['slots'], mode='memory')
    if method == 'memory':
        return master.to(device)
    model = EvidenceMatcher(dim=256, heads=cfg['heads'], slots=cfg['slots'],
                            mode='current' if method == 'last' else 'mean')
    source = master.state_dict()
    model.load_state_dict({name: source[name] for name in model.state_dict()}, strict=True)
    return model.to(device)


def network_inputs(arrays, device):
    # IDs, teacher scores, boxes, masks and truth remain outside the network.
    return {key: torch.from_numpy(arrays[key]).to(device) for key in NETWORK_FIELDS}


def advance(model, method, observation, valid, state=None):
    if method != 'last':
        return model.update(observation, valid, state)
    # Last valid view PER CAR; a temporarily absent car keeps its previous view.
    fresh = model.update(observation, valid, state=None)
    if state is None:
        return fresh
    visible = valid.any(-1)
    return dict(slots=torch.where(visible[..., None, None], fresh['slots'], state['slots']),
                seen=state['seen'] | visible, count=state['count'] + visible.to(observation.dtype))


def score(model, inputs, state):
    logits = model.score(inputs['reference'], inputs['reference_valid'], state)
    available = torch.cat((state['seen'], torch.ones(len(logits), 1, dtype=torch.bool, device=logits.device)), 1)
    return logits.masked_fill(~available, torch.finfo(logits.dtype).min)


def rollout(model, method, inputs):
    state, values = None, []
    for t in range(inputs['observation'].shape[1]):
        state = advance(model, method, inputs['observation'][:, t], inputs['valid'][:, t], state)
        values.append(score(model, inputs, state))
    return torch.stack(values, 1), state


def swap_content(state):
    swapped = {name: value.clone() for name, value in state.items()}
    for i in range(len(state['seen'])):
        ids = state['seen'][i].nonzero().flatten()
        if len(ids) >= 2:
            swapped['slots'][i, ids] = state['slots'][i, ids.roll(1)]
    return swapped


def _describe(logits, state, arrays, routes, t, previous_slots):
    values = logits.double().cpu().numpy()
    probabilities = np.exp(values-values.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    seen = state['seen'].cpu().numpy()
    counts = state['count'].cpu().numpy()
    slots = state['slots'].float().cpu().numpy()
    result = []
    capacity = values.shape[1]-1
    for i, route in enumerate(routes):
        identities = arrays['identities'][i]
        uid = lambda j, ids=identities: ids[int(j)] if int(j) < len(ids) else None
        choice = int(values[i].argmax())
        ranked = np.where(seen[i], values[i, :-1], -np.inf)
        rank = int(ranked.argmax()) if seen[i].any() else capacity
        truth = uid(arrays['truth'][i, t])
        physical = uid(arrays['physical_truth'][i, t])
        raw, wrote = arrays['current_valid'][i, t].any(-1), arrays['valid'][i, t].any(-1)
        traces = {}
        for j, identity in enumerate(identities):
            delta = None if previous_slots is None else float(np.linalg.norm(slots[i, j]-previous_slots[i, j]))
            traces[identity] = dict(seen=bool(seen[i, j]), writes=int(counts[i, j]),
                                    slot_sha256=hashlib.sha256(slots[i, j].tobytes()).hexdigest(),
                                    change_l2=delta)
        result.append(dict(route_id=route['id'], step=t, record_id=route['records'][t],
                           target_identity=route['target_identity'], identities=identities,
                           choice_identity=uid(choice), rank_identity=uid(rank),
                           scores=[*values[i, :len(identities)].tolist(), float(values[i, -1])],
                           probabilities=[*probabilities[i, :len(identities)].tolist(), float(probabilities[i, -1])],
                           accepted_truth=truth, physical_truth=physical,
                           current_target=uid(arrays['current_truth'][i, t]),
                           correct=uid(choice) == truth,
                           seen_identities=[u for j, u in enumerate(identities) if seen[i, j]],
                           current_identities=[u for j, u in enumerate(identities) if raw[j]],
                           written_identities=[u for j, u in enumerate(identities) if wrote[j]],
                           observation_keys=arrays['observation_keys'][i][t], memory=traces))
    return result, slots, probabilities


def _control_rows(name, alternative, normal, normal_probabilities,
                  arrays, routes, t, eligible):
    values = alternative.cpu().numpy()
    probability = torch.softmax(alternative.double(), -1).cpu().numpy()
    rows = []
    for i, route in enumerate(routes):
        if not bool(eligible[i]):
            continue
        ids = arrays['identities'][i]
        index = int(values[i].argmax())
        choice = ids[index] if index < len(ids) else None
        target = int(arrays['target_index'][i])
        rows.append(dict(control=name, route_id=route['id'], step=t, eligible=True,
                         choice_identity=choice, normal_choice_identity=normal[i]['choice_identity'],
                         control_correct=choice == normal[i]['accepted_truth'], normal_correct=normal[i]['correct'],
                         target_probability=float(probability[i, target]),
                         normal_target_probability=float(normal_probabilities[i, target]),
                         probability_l1=float(np.abs(probability[i]-normal_probabilities[i]).sum())))
    return rows


@torch.inference_mode()
def evaluate(model, method, selected, records, refs, features, device, batch_size, *, dedup, diagnostics=False):
    model.eval()
    predictions, controls = [], []
    start_total = time.perf_counter()
    forward_seconds = 0.
    for offset in range(0, len(selected), batch_size):
        routes = selected[offset:offset+batch_size]
        arrays = sequence_arrays(routes, records, refs, features, dedup=dedup)
        inputs = network_inputs(arrays, device)
        state, previous_slots, repeated = None, None, np.zeros(len(routes), bool)
        for t in range(inputs['observation'].shape[1]):
            previous = state
            torch.cuda.synchronize(device); before = time.perf_counter()
            state = advance(model, method, inputs['observation'][:, t], inputs['valid'][:, t], state)
            logits = score(model, inputs, state)
            torch.cuda.synchronize(device); forward_seconds += time.perf_counter()-before
            normal, slots, probability = _describe(logits, state, arrays, routes, t, previous_slots)
            predictions.extend(normal); previous_slots = slots
            if not diagnostics:
                continue
            # Each intervention starts from the NORMAL prefix; never pollute the next step.
            reset = advance(model, method, inputs['observation'][:, t], inputs['current_valid'][:, t], None)
            controls.extend(_control_rows('reset', score(model, inputs, reset), normal, probability,
                                           arrays, routes, t, np.ones(len(routes), bool)))
            if previous is not None:
                swapped = advance(model, method, inputs['observation'][:, t], inputs['valid'][:, t], swap_content(previous))
                eligible = previous['seen'].sum(-1).cpu().numpy() >= 2
                controls.extend(_control_rows('swap_content_before_current', score(model, inputs, swapped), normal,
                                               probability, arrays, routes, t, eligible))
                target_ids = torch.from_numpy(arrays['target_index']).to(device)
                eligible = previous['seen'].gather(1, target_ids[:, None]).squeeze(1).cpu().numpy()
                eligible &= arrays['valid'][:, t].any(axis=(1, 2))
                controls.extend(_control_rows('pause_current_write', score(model, inputs, previous), normal,
                                               probability, arrays, routes, t, eligible))
            # Once per route, intentionally bypass dedup to measure updater drift.
            eligible = arrays['current_valid'][:, t].any(axis=(1, 2)) & ~repeated
            if eligible.any():
                extra = {k: v.clone() for k, v in state.items()}
                valid = inputs['current_valid'][:, t] & torch.from_numpy(eligible).to(device)[:, None, None]
                for n in range(1, 9):
                    extra = advance(model, method, inputs['observation'][:, t], valid, extra)
                    if n in (1, 8):
                        controls.extend(_control_rows(f'repeat_bypass_dedup_extra{n}', score(model, inputs, extra),
                                                       normal, probability, arrays, routes, t, eligible))
                repeated |= eligible
    return dict(predictions=predictions, metrics=summarize(predictions), controls=controls,
                control_summary=control_summary(controls),
                timing=dict(cached_normal_forward_seconds=forward_seconds,
                            evaluation_with_logging_controls_wall_seconds=time.perf_counter()-start_total,
                            prefixes=len(predictions), scope='precomputed crops only; not online end-to-end latency'))


def _loss(logits, arrays, routes, weights, device):
    truth = torch.from_numpy(arrays['truth']).to(device)
    capacity = logits.shape[-1]-1
    sample_weights = np.asarray([[weights['Null' if label == capacity else route['target_identity']]
                                  for label in arrays['truth'][i]] for i, route in enumerate(routes)], np.float32)
    losses = F.cross_entropy(logits.flatten(0, 1), truth.flatten(), reduction='none').reshape_as(truth)
    return (losses * torch.from_numpy(sample_weights).to(device)).mean()


def server_checks(device, cfg):
    """Mandatory before training; CUDA synthetic checks, never called by local tests."""
    seed_all(123)
    model = make_model(cfg, 123, device, 'memory')
    x = torch.randn(2, 4, 3, 16, 256, device=device)
    valid = torch.zeros(2, 4, 3, 16, dtype=torch.bool, device=device)
    valid[:, 0, 0] = True; valid[:, 1, 0] = True; valid[:, 2, 1] = True
    inputs = dict(observation=x, valid=valid, current_valid=valid.clone(),
                  reference=torch.randn(2, 64, 256, device=device),
                  reference_valid=torch.ones(2, 64, dtype=torch.bool, device=device))
    logits, final = rollout(model, 'memory', inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), torch.zeros(8, dtype=torch.long, device=device))
    loss.backward()
    if (not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
            or not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.updater.parameters())):
        raise AssertionError('sequence gradients invalid or updater disconnected')
    with torch.no_grad():
        moved = {k: v.clone() for k, v in inputs.items()}
        for key in ('observation', 'valid', 'current_valid'):
            moved[key] = inputs[key][:, :, [2, 0, 1]]
        after, _ = rollout(model, 'memory', moved)
        torch.testing.assert_close(after, logits[:, :, [2, 0, 1, 3]], atol=2e-5, rtol=1e-5)
        changed = dict(inputs, observation=x.clone()); changed['observation'][:, 2:] += 100
        future, _ = rollout(model, 'memory', changed)
        torch.testing.assert_close(future[:, :2], logits[:, :2], atol=0, rtol=0)
        stopped = advance(model, 'memory', x[:, -1], torch.zeros_like(valid[:, -1]), final)
        for key in final:
            torch.testing.assert_close(stopped[key], final[key], atol=0, rtol=0)
        swapped = swap_content(final)
        for key in ('seen', 'count'):
            torch.testing.assert_close(swapped[key], final[key], atol=0, rtol=0)
        if torch.equal(swapped['slots'], final['slots']):
            raise AssertionError('content swap did not exchange seen cars')
        last = make_model(cfg, 123, device, 'last')
        mean = make_model(cfg, 123, device, 'mean')
        if len({parameter_hash(m, common=True) for m in (last, mean, model)}) != 1:
            raise AssertionError('common initialization differs')
        _, last_state = rollout(last, 'last', inputs)
        if not last_state['seen'][:, :2].all():
            raise AssertionError('last-view baseline discarded absent cars')
    return dict(status='passed', loss=float(loss.detach()), checks=[
        'temporal backward reaches updater', 'fixed candidate permutation', 'no future influence',
        'no-write preserves state exactly', 'swap content preserves seen/count',
        'common initialization', 'last valid per car preserves disappeared vehicles'])


def train(args):
    output = Path(args.output).resolve(); directory = output/'training'
    if directory.exists():
        raise ValueError('training exists; preserve prior checkpoints and choose a new output')
    device = cuda_device(args.device); torch.set_num_threads(args.threads)
    meta, records, routes, refs, features = load_prepared(output, args.data)
    cfg = meta['config']['training']; dedup = meta['config']['exact_dedup']
    training = [r for r in routes if r['split'] == 'train']
    partitions = {s: [r for r in routes if r['split'] == s] for s in sorted({r['split'] for r in routes})}
    weighting = training_weights(training, records, cfg['balance_classes'])
    directory.mkdir()
    run = dict(kind='region_sequence_training_v1', status='running', config=cfg, methods=list(METHODS),
               plan_sha256=sha(output/'plan.json'), prepared_sha256=sha(output/'prepared.json'),
               code=meta['code'], assistance=meta['assistance'], sampling=meta['sampling'], weighting=weighting,
               planned_batches_per_epoch=math.ceil(len(training)/cfg['batch_size']),
               selection='fixed final epoch; no validation/probe/test selection', gradient_clip_norm=1.,
               comparison='same routes, features, label weights, common initial parameters and batch order; '
                          'last keeps the most recent valid observation per identity; mean and memory accumulate',
               environment=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                                gpu=torch.cuda.get_device_name(device), numpy=str(np.__version__)), results=[])
    write_json(directory/'run.json', run)
    checks = server_checks(device, cfg)
    # Real-data smoke: two updates, discarded before all seeded comparisons.
    smoke = make_model(cfg, 123, device, 'memory')
    optimizer = torch.optim.AdamW(smoke.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    real = training[:cfg['batch_size']]
    arrays = sequence_arrays(real, records, refs, features, dedup=dedup)
    inputs = network_inputs(arrays, device); checks['real_smoke_losses'] = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(rollout(smoke, 'memory', inputs)[0], arrays, real, weighting['weights'], device)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite real sequence smoke loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(smoke.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); checks['real_smoke_losses'].append(float(loss.detach()))
    del smoke, optimizer, inputs, loss
    write_json(directory/'checks.json', checks)
    for seed in cfg['seeds']:
        common_hash, fingerprints = None, []
        for method in METHODS:
            folder = directory/f'{method}_seed{seed}'; folder.mkdir()
            model = make_model(cfg, seed, device, method)
            initial = parameter_hash(model, common=True)
            if common_hash is not None and common_hash != initial:
                raise AssertionError('shared initial parameters differ between methods')
            common_hash = initial
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            schedule, updates = hashlib.sha256(), 0
            torch.cuda.reset_peak_memory_stats(device); start_training = time.perf_counter()
            for epoch in range(cfg['epochs']):
                model.train(); rng = np.random.default_rng(seed*100000+epoch)
                order = rng.permutation(len(training)).tolist(); total_loss = 0.
                for offset in range(0, len(order), cfg['batch_size']):
                    batch = [training[j] for j in order[offset:offset+cfg['batch_size']]]
                    arrays = sequence_arrays(batch, records, refs, features, dedup=dedup, rng=rng)
                    schedule.update(digest_json([epoch, [r['id'] for r in batch], arrays['identities']]).encode())
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = rollout(model, method, network_inputs(arrays, device))
                    loss = _loss(logits, arrays, batch, weighting['weights'], device)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite sequence training loss; preserve failed run')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); updates += 1; total_loss += float(loss.detach())*len(batch)
                entry = dict(epoch=epoch+1, mean_loss=total_loss/len(training), routes=len(training), optimizer_updates=updates)
                with (folder/'train.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry)+'\n')
                if epoch == 0 or (epoch+1) % 10 == 0:
                    print(f'{method} seed={seed} epoch={epoch+1}/{cfg["epochs"]} loss={entry["mean_loss"]:.6f}', flush=True)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-start_training
            fingerprints.append((schedule.hexdigest(), updates))
            if len(set(fingerprints)) != 1:
                raise AssertionError('training order or optimizer budget differs between methods')
            checkpoint = dict(kind='region_sequence_matcher_v1', method=method, seed=seed, config=model.config,
                              epoch=cfg['epochs'], model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                              plan_sha256=run['plan_sha256'], prepared_sha256=run['prepared_sha256'], code=run['code'],
                              common_initial_sha256=initial, schedule_sha256=schedule.hexdigest(), optimizer_updates=updates)
            torch.save(checkpoint, folder/'last.pt')
            result = dict(method=method, seed=seed, checkpoint=f'training/{method}_seed{seed}/last.pt',
                          checkpoint_sha256=sha(folder/'last.pt'), common_initial_sha256=initial,
                          schedule_sha256=schedule.hexdigest(), optimizer_updates=updates, train_seconds=seconds,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device), splits={})
            for split, selected in partitions.items():
                value = evaluate(model, method, selected, records, refs, features, device, cfg['batch_size'],
                                 dedup=dedup, diagnostics=method == 'memory' and split != 'train')
                write_json(folder/(split+'.json'), value)
                result['splits'][split] = dict(metrics=value['metrics'], control_summary=value['control_summary'],
                                               timing=value['timing'], evaluation_sha256=sha(folder/(split+'.json')))
            saved = torch.load(folder/'last.pt', map_location=device, weights_only=True)
            restored = EvidenceMatcher(**saved['config']).to(device)
            restored.load_state_dict(saved['model_state'], strict=True)
            error = 0.
            for split, selected in partitions.items():
                if split == 'train':
                    continue
                expected = read_json(folder/(split+'.json'))['predictions']
                actual = evaluate(restored, method, selected, records, refs, features, device, cfg['batch_size'],
                                  dedup=dedup)['predictions']
                for a, b in zip(expected, actual, strict=True):
                    delta = float(np.max(np.abs(np.asarray(a['scores'])-np.asarray(b['scores']))))
                    error = max(error, delta)
                    if ((a['route_id'], a['step'], a['choice_identity']) != (b['route_id'], b['step'], b['choice_identity'])
                            or delta > 2e-5):
                        raise AssertionError('checkpoint reload did not reproduce sequence scores')
            result['reload_max_abs_score_error'] = error
            run['results'].append(result); write_json(directory/'run.json', run)
            print(method, seed, {s: v['metrics']['accepted_target_accuracy'] for s, v in result['splits'].items()}, flush=True)
            del model, optimizer, restored, checkpoint, saved, logits, loss
    if (source_record() != meta['code'] or sha(output/'plan.json') != run['plan_sha256']
            or sha(output/'prepared.json') != run['prepared_sha256']):
        raise ValueError('implementation or sequence artifacts changed during training')
    run['status'] = 'complete'; write_json(directory/'run.json', run)
    return run
