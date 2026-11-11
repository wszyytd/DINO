"""Matched-budget baselines and explicitly scoped, causal grid-sequence diagnostics."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dense_cache import save_json
from .evidence_memory import EvidenceMatcher
from .local_pairs import sha256


def partition_scenes(scenes, meta):
    """Prefer supplied independent groups; otherwise expose a spatial holdout diagnostic."""
    supplied = {s['split'] for s in scenes}
    partitions = {'train': [], 'probe': []}
    teacher_images = set(meta['teacher_training_images'])
    if 'val' in supplied or 'test' in supplied:
        groups = {}
        for i, scene in enumerate(scenes):
            part = 'train' if scene['split'] == 'train' else 'probe'
            if scene['group'] in groups and groups[scene['group']] != part:
                raise ValueError('scene group crosses split')
            groups[scene['group']] = part
            if part == 'probe' and (scene['group'] in meta['teacher_training_groups'] or
                                    teacher_images.intersection(scene['image_hashes'])):
                raise ValueError('held-out scene overlaps teacher pretraining')
            partitions[part].append((i, list(range(len(scene['state_ids'])))))
        scope = 'supplied independent-layout holdout; fixed final epoch, no probe model selection'
    else:
        if len(scenes) != 1:
            raise ValueError('supply val/test scene groups for multiple layouts')
        scene = scenes[0]
        cols = [p['col'] for p in scene['graph']['points'].values()]
        cut = (max(cols)+1)*3//5
        fit, probe = [], []
        for j, sid in enumerate(scene['state_ids']):
            point = scene['graph']['points'][scene['graph']['states'][sid]['point_id']]
            (fit if point['col'] < cut or scene['image_hashes'][j] in teacher_images else probe).append(j)
        partitions = dict(train=[(0, fit)], probe=[(0, probe)])
        scope = ('SINGLE-LAYOUT DIAGNOSTIC: columns split, same assets/background/references; '
                 'probe RGB not used by teacher fitting or memory fitting; NOT independent-scene validation')
    hashes = {name: {scenes[i]['image_hashes'][j] for i, ids in parts for j in ids}
              for name, parts in partitions.items()}
    if not all(hashes.values()) or hashes['train'] & hashes['probe']:
        raise ValueError('empty split or shared source images')
    return partitions, scope


def make_routes(scenes, parts, count=64, length=8, seed=7):
    if count < 1 or length < 2:
        raise ValueError('need positive routes and sequence length >=2')
    rng = random.Random(seed)
    routes = []
    for n in range(count):
        scene_index, allowed_ids = parts[n % len(parts)]
        scene = scenes[scene_index]
        index = {sid: j for j, sid in enumerate(scene['state_ids'])}
        allowed = set(allowed_ids)
        current = rng.choice(allowed_ids)
        path, actions = [current], []
        for _ in range(length-1):
            edges = scene['graph']['states'][scene['state_ids'][current]]['edges']
            choices = [(name, index[e['to']]) for name, e in edges.items() if index[e['to']] in allowed]
            if not choices:
                raise ValueError('isolated observation state')
            weights = [0.1 if name == 'hold_capture' else 1. for name, _ in choices]
            name, current = rng.choices(choices, weights=weights, k=1)[0]
            path.append(current); actions.append(name)
        # Each path is evaluated against every reference. No target-conditioned route choice.
        for target in range(len(scene['objects'])):
            routes.append(dict(scene=scene_index, frames=path, target=target, actions=actions))
    return routes


def batches(routes, scenes, batch_size, rng=None):
    # Candidate counts vary between scenes: group only for tensor batching, not model capacity.
    groups = {}
    for route in routes:
        groups.setdefault(len(scenes[route['scene']]['objects']), []).append(route)
    result = []
    for rows in groups.values():
        rows = list(rows)
        if rng:
            rng.shuffle(rows)
        result.extend(rows[i:i+batch_size] for i in range(0, len(rows), batch_size))
    if rng:
        rng.shuffle(result)
    return result


def pack(rows, scenes, device, rng=None):
    xs, vs, refs, rvs, targets = [], [], [], [], []
    for row in rows:
        scene = scenes[row['scene']]
        k = len(scene['objects'])
        permutation = list(range(k))
        if rng:
            rng.shuffle(permutation)
        xs.append(scene['observations'][row['frames']][:, permutation])
        vs.append(scene['valid'][row['frames']][:, permutation])
        refs.append(scene['references'][row['target']]); rvs.append(scene['reference_valid'][row['target']])
        targets.append(permutation.index(row['target']))
    return (torch.stack(xs).to(device), torch.stack(vs).to(device), torch.stack(refs).to(device),
            torch.stack(rvs).to(device), torch.tensor(targets, device=device))


def score_batch(model, batch, reset=False, swap=False):
    x, valid, ref, rv, target = batch
    state, ever = None, torch.zeros(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
    losses, records = [], []
    for t in range(x.shape[1]):
        if reset:
            state = None
        elif swap and state is not None:
            state = {key: value.roll(1, 1) for key, value in state.items()}
        state = model.update(x[:, t], valid[:, t], state)
        logits = model.score(ref, rv, state)
        visible = valid[:, t].any(-1)
        ever = ever | visible
        available = state['seen'].gather(1, target[:, None]).squeeze(1)
        label = torch.where(available, target, x.shape[2])
        losses.append(F.cross_entropy(logits, label))
        # Common historical labels for reporting; current-visible strata isolate fusion.
        target_seen = ever.gather(1, target[:, None]).squeeze(1)
        truth = torch.where(target_seen, target, x.shape[2])
        prob = logits.softmax(-1)
        confidence, prediction = prob.max(-1)
        records.append(dict(correct=(prediction == truth), confidence=confidence,
                            visible=visible.gather(1, target[:, None]).squeeze(1),
                            both=visible.sum(-1) >= 2, historical=target_seen, step=t,
                            predicted=prediction, truth=truth))
    return torch.stack(losses).mean(), records


@torch.no_grad()
def evaluate(model, routes, scenes, device, batch_size=16, reset=False, swap=False):
    model.eval()
    counts = {name: [0, 0] for name in ('all_prefix', 'current_visible', 'both_visible',
                                       'target_hidden_after_seen', 'target_not_seen', 'final')}
    conf_wrong = confident = 0
    probabilities, examples = [], []
    loss = 0.
    for rows in batches(routes, scenes, batch_size):
        value, records = score_batch(model, pack(rows, scenes, device), reset, swap)
        loss += float(value)*len(rows)
        for j, row in enumerate(records):
            conditions = dict(all_prefix=torch.ones_like(row['correct']), current_visible=row['visible'],
                              both_visible=row['both'] & row['visible'],
                              target_hidden_after_seen=row['historical'] & ~row['visible'],
                              target_not_seen=~row['historical'],
                              final=torch.ones_like(row['correct']) if j+1 == len(records) else torch.zeros_like(row['correct']))
            for name, keep in conditions.items():
                counts[name][0] += int((row['correct'] & keep).sum())
                counts[name][1] += int(keep.sum())
            high = row['confidence'] >= .9
            confident += int(high.sum()); conf_wrong += int((high & ~row['correct']).sum())
            probabilities.extend(row['confidence'].cpu().tolist())
        for n, route in enumerate(rows):
            examples.append(dict(scene=route['scene'], target=route['target'], frames=route['frames'],
                                 final_correct=bool(records[-1]['correct'][n]),
                                 final_prediction=int(records[-1]['predicted'][n])))
    return dict(loss=loss/len(routes), metrics={name: dict(correct=a, count=b, accuracy=a/b if b else None)
                                              for name, (a, b) in counts.items()},
                confident_at_0_9=confident, wrong_confident=conf_wrong,
                confidence_threshold_calibrated=False, examples=examples)


@torch.no_grad()
def repeated_drift(model, scenes, parts, device):
    changes = []
    for i, ids in parts:
        scene = scenes[i]
        selected = [j for j in ids if scene['valid'][j].any()][::8]
        for j in selected:
            row = dict(scene=i, frames=[j], target=0)
            x, v, r, rv, _ = pack([row], scenes, device)
            state = model.update(x[:, 0], v[:, 0])
            before = model.score(r, rv, state).softmax(-1)
            for _ in range(3):
                state = model.update(x[:, 0], v[:, 0], state)
            changes.append(float((model.score(r, rv, state).softmax(-1)-before).abs().max()))
    return dict(samples=len(changes), mean_max_probability_change=float(np.mean(changes)) if changes else None,
                max_probability_change=max(changes) if changes else None)


def train_comparison(scenes, meta, args):
    from .local_features import resolve_device
    device = resolve_device(args.device, torch)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    if output.exists():
        raise ValueError('output exists; choose a new directory')
    if min(args.epochs, args.batch_size, args.threads, args.routes) < 1 or args.lr <= 0:
        raise ValueError('positive training settings required')
    partitions, scope = partition_scenes(scenes, meta)
    probe_routes = make_routes(scenes, partitions['probe'], args.routes, args.length, 901)
    output.mkdir(parents=True)
    split = {name: [dict(scene=scenes[i]['scene_id'], states=[scenes[i]['state_ids'][j] for j in ids])
                   for i, ids in parts] for name, parts in partitions.items()}
    save_json(output/'split.json', split)
    save_json(output/'probe_routes.json', probe_routes)
    run = dict(status='running', scope=scope, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               feature_contract=meta['contract'], cache_manifest_sha256=sha256(Path(args.cache)/'cache.json'),
               task='oracle-masked, oracle-associated candidate retrieval; not localization or navigation',
               selection='fixed final epoch; no tuning or checkpoint selection on probe', results=[])
    save_json(output/'run.json', run)
    dim = scenes[0]['observations'].shape[-1]
    for seed in args.seeds:
        torch.manual_seed(seed)
        common = EvidenceMatcher(dim=dim, mode='current').state_dict()
        for mode in ('current', 'mean', 'memory'):
            torch.manual_seed(seed)
            model = EvidenceMatcher(dim=dim, mode=mode).to(device)
            model.load_state_dict(common, strict=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            history = []
            for epoch in range(args.epochs):
                model.train()
                routes = make_routes(scenes, partitions['train'], args.routes, args.length, seed*1000+epoch)
                rng = random.Random(seed*1000+epoch)
                total = 0.
                for rows in batches(routes, scenes, args.batch_size, rng):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = score_batch(model, pack(rows, scenes, device, rng))
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step(); total += float(loss.detach())*len(rows)
                history.append(dict(epoch=epoch+1, loss=total/len(routes)))
                if epoch == 0 or (epoch+1)%5 == 0 or epoch+1 == args.epochs:
                    print(f'{mode} seed={seed} epoch={epoch+1}/{args.epochs} loss={history[-1]["loss"]:.4f}', flush=True)
            folder = output/f'{mode}_seed{seed}';folder.mkdir()
            torch.save(dict(kind='oracle_evidence_memory_v1', config=model.config, model_state=model.state_dict(),
                            feature_contract=meta['contract'], seed=seed, epoch=args.epochs, scope=scope), folder/'last.pt')
            save_json(folder/'history.json', history)
            probe = evaluate(model, probe_routes, scenes, device, args.batch_size)
            save_json(folder/'probe.json', probe)
            result = dict(mode=mode, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                          metrics=probe['metrics'], wrong_confident=probe['wrong_confident'],
                          repeat=repeated_drift(model, scenes, partitions['probe'], device))
            if mode == 'memory':
                for name, reset, swap in [('reset_history', True, False), ('swapped_history', False, True)]:
                    diagnostic = evaluate(model, probe_routes, scenes, device, args.batch_size, reset, swap)
                    result[name] = diagnostic['metrics']
                    save_json(folder/(name+'.json'), diagnostic)
            run['results'].append(result);save_json(output/'run.json', run)
            print('PROBE', mode, seed, json.dumps(probe['metrics']), flush=True)
    run['status'] = 'complete';save_json(output/'run.json', run)
    lines = ['# Stage-one memory diagnostic', '', scope, '',
             'Frozen full-image branch only; GT masks and identity routing. Not the teacher spatial scorer.', '',
             '| Mode | Seed | Current-visible | Both-visible | Hidden after seen | All prefixes |',
             '|---|---:|---:|---:|---:|---:|']
    for r in run['results']:
        def fmt(key, r=r):
            m=r['metrics'][key]
            return f"{m['correct']}/{m['count']} ({m['accuracy']:.1%})" if m['count'] else 'N/A'
        lines.append(f"| {r['mode']} | {r['seed']} | {fmt('current_visible')} | {fmt('both_visible')} | {fmt('target_hidden_after_seen')} | {fmt('all_prefix')} |")
    lines += ['', 'Counts are correlated prefixes/routes from the same source images, not independent trials.',
              'All modes use the same frozen features, masks, per-frame slot writer, matching reader, routes and training budget.',
              'Memory adds a gated recurrent update. Mean averages written slots. Current discards history.',
              'Current trains current-frame availability; history modes train observed-so-far availability. Compare both-visible to isolate fusion.',
              'A probability threshold of 0.9 is diagnostic only, not a calibrated stopping rule.',
              'Full reset and deliberately swapped-history ablations and repeat drift are in run.json.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return run
