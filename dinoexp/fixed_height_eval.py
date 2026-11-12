"""Authorized-CUDA-only feature extraction and fixed-checkpoint evaluation.

No optimizer, fitting, validation selection or changes to the original model.
"""
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .core import checked_path
from .evidence_memory import EvidenceMatcher
from .fixed_height_data import batch_arrays, load_features, load_plan, summarize
from .region_ab_data import sha, write_json
from .region_ab_prepare import FrozenRuntime, cuda_device, pool
from .region_sequence_training import advance, network_inputs, parameter_hash, score


def check_encoder(runtime, source):
    if (runtime.teacher_sha256 != source['teacher_sha256']
            or sha(runtime.backbone_path) != source['backbone_sha256']
            or runtime.cfg != source['teacher_feature_config']):
        raise ValueError('frozen encoder differs from the source training experiment')
    for name, expected in source['backbone_source'].items():
        if sha(checked_path(runtime.repo, name)) != expected:
            raise ValueError('external DINO source differs: '+name)


@torch.inference_mode()
def prepare(args):
    output, root = Path(args.output).resolve(), Path(args.data).resolve()
    plan = load_plan(args)
    if any((output/name).exists() for name in ('prepared.json', 'features', 'references')):
        raise ValueError('features already started; choose a new output')
    device = cuda_device(args.device); torch.set_num_threads(args.threads)
    runtime = FrozenRuntime(args); check_encoder(runtime, plan['source'])
    (output/'features').mkdir(); (output/'references').mkdir()
    meta = dict(kind='fixed_height_test_features_v1', status='running',
                plan_sha256=sha(output/'plan.json'), protocol=plan['protocol'],
                code=plan['code'], observations={}, references={},
                environment=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                                 gpu=torch.cuda.get_device_name(device), numpy=str(np.__version__)))
    write_json(output/'prepared.json', meta)
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for case in plan['cases']:
        key = case['reference_sha256']
        if key not in meta['references']:
            with Image.open(checked_path(root, case['reference'])) as im:
                _, _, tokens, valid = runtime.reference(im.convert('RGB'), key)
            if tokens.shape != (64, 256) or not valid.any() or not np.isfinite(tokens).all():
                raise ValueError('invalid reference features')
            file = 'references/'+key+'.npz'
            np.savez_compressed(output/file, tokens=tokens, valid=valid)
            meta['references'][key] = dict(file=file, sha256=sha(output/file))
    for i, record in enumerate(plan['records']):
        wanted = [o for o in record['observations'] if o['key'] not in meta['observations']]
        if wanted:
            with Image.open(checked_path(root, record['image'])) as im:
                image = im.convert('RGB')
            for obj in wanted:
                crop = image.crop(obj['box']); q = runtime.independent(crop)
                tokens, valid = pool(q, crop.size, [0, 0, crop.width, crop.height])
                if tokens.shape != (16, 256) or not valid.any() or not np.isfinite(tokens).all():
                    raise ValueError('failed crop extraction; never drop or relabel: '+record['id'])
                file = 'features/'+obj['key']+'.npz'
                np.savez_compressed(output/file, observation=tokens, valid=valid)
                meta['observations'][obj['key']] = dict(file=file, sha256=sha(output/file))
        if (i+1) % 24 == 0:
            write_json(output/'prepared.json', meta)
            print(f'Fixed-height features {i+1}/{len(plan["records"])} frames', flush=True)
    torch.cuda.synchronize(device)
    load_plan(args); check_encoder(runtime, plan['source'])
    meta.update(status='complete', seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    write_json(output/'prepared.json', meta)
    return meta


def load_matcher(args, entry, plan, device):
    file = checked_path(Path(args.checkpoint_run).resolve(), entry['checkpoint'])
    if sha(file) != entry['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    checkpoint = torch.load(file, map_location='cpu', weights_only=True)
    expected_mode = {'last': 'current', 'mean': 'mean', 'memory': 'memory'}[entry['method']]
    if (checkpoint.get('kind') != 'region_sequence_matcher_v1'
            or (checkpoint['method'], checkpoint['seed']) != (entry['method'], entry['seed'])
            or checkpoint['config']['mode'] != expected_mode
            or checkpoint['plan_sha256'] != plan['source']['plan_sha256']
            or checkpoint['prepared_sha256'] != plan['source']['prepared_sha256']
            or checkpoint['epoch'] != plan['source']['config']['epochs']):
        raise ValueError('checkpoint provenance/config mismatch')
    model = EvidenceMatcher(**checkpoint['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    if sha(file) != entry['checkpoint_sha256']:
        raise ValueError('checkpoint changed while loading')
    return model


def decoded(logits, order, target):
    values = logits.detach().float().cpu().numpy()
    probabilities = logits.softmax(-1).detach().float().cpu().numpy()
    choice = int(np.argmax(values)); truth = order.index(target)
    return dict(scores=values.tolist(), probabilities=probabilities.tolist(),
                choice_identity=order[choice] if choice < len(order) else None,
                correct=choice == truth,
                target_ranks_first=bool(values[truth] > max(v for j, v in enumerate(values[:-1]) if j != truth)))


@torch.inference_mode()
def predict(model, method, cases, plan, refs, features, device, batch_size):
    records = {r['id']: r for r in plan['records']}; rows = []
    for start in range(0, len(cases), batch_size):
        selected = cases[start:start+batch_size]
        arrays = batch_arrays(selected, records, refs, features)
        inputs = network_inputs(arrays, device); state = None
        for t in range(inputs['observation'].shape[1]):
            previous = None if state is None else state['slots'].clone()
            observation = inputs['observation'][:, t]
            state = advance(model, method, observation, inputs['valid'][:, t], state)
            logits = score(model, inputs, state)
            fresh = advance(model, method, observation, inputs['current_valid'][:, t], None)
            single = score(model, inputs, fresh)
            for i, case in enumerate(selected):
                order = arrays['identities'][i]; target = case['target_identity']
                current = decoded(logits[i], order, target); single_value = decoded(single[i], order, target)
                record = records[case['records'][t]]
                slots = state['slots'][i].detach().float().cpu().numpy()
                counts = state['count'][i].detach().cpu().numpy()
                rows.append(dict(case_id=case['id'], group=case['group'], pair_kind=case['pair_kind'],
                                 target_identity=target, step=t, angle=record['angle'],
                                 record_id=record['id'], identities=order, **current,
                                 last_step=t == len(case['records'])-1,
                                 single_frame_correct=single_value['correct'], single_frame=single_value,
                                 state=[dict(identity=uid, count=float(counts[j]),
                                             sha256=hashlib.sha256(slots[j].tobytes()).hexdigest(),
                                             change_l2=None if previous is None else float(torch.linalg.vector_norm(
                                                 state['slots'][i, j]-previous[i, j])))
                                        for j, uid in enumerate(order)]))
    return rows


@torch.inference_mode()
def check_order(model, method, case, plan, refs, features, device):
    records = {r['id']: r for r in plan['records']}
    normal = network_inputs(batch_arrays([case], records, refs, features), device)
    swapped = network_inputs(batch_arrays([case], records, refs, features, reverse_candidates=True), device)
    left, right, error = None, None, 0.
    for t in range(normal['observation'].shape[1]):
        left = advance(model, method, normal['observation'][:, t], normal['valid'][:, t], left)
        right = advance(model, method, swapped['observation'][:, t], swapped['valid'][:, t], right)
        a, b = score(model, normal, left), score(model, swapped, right)[:, [1, 0, 2]]
        error = max(error, float((a-b).abs().max()))
    if error > 2e-5:
        raise AssertionError('candidate order changes identity-aligned scores')
    return error


@torch.inference_mode()
def evaluate(args):
    output = Path(args.output).resolve(); directory = output/'evaluation'
    plan, _, refs, features = load_features(args)
    if directory.exists():
        raise ValueError('evaluation exists; preserve it and choose a new output')
    device = cuda_device(args.device); torch.set_num_threads(args.threads)
    directory.mkdir()
    run = dict(kind='fixed_height_test_evaluation_v1', status='running',
               plan_sha256=sha(output/'plan.json'), prepared_sha256=sha(output/'prepared.json'),
               source=plan['source'], protocol=plan['protocol'], results=[],
               negative_examples=0, threshold_selection='none; unchanged logits including Null',
               reference_order='each capture case independently; no refA/refB RGB pairing assumed',
               timing_scope='cached features, batched matching and diagnostics; excludes RGB encoding',
               environment=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                                gpu=torch.cuda.get_device_name(device)))
    write_json(directory/'run.json', run)
    for entry in plan['source']['checkpoints']:
        model = load_matcher(args, entry, plan, device); before = parameter_hash(model)
        torch.cuda.synchronize(device); start = time.perf_counter()
        rows = predict(model, entry['method'], plan['cases'], plan, refs, features, device, args.batch_size)
        torch.cuda.synchronize(device); seconds = time.perf_counter()-start
        order_error = check_order(model, entry['method'], plan['cases'][0], plan, refs, features, device)
        restored = load_matcher(args, entry, plan, device)
        repeated = predict(restored, entry['method'], plan['cases'], plan, refs, features, device, args.batch_size)
        error = 0.
        for first, second in zip(rows, repeated, strict=True):
            error = max(error, float(np.max(np.abs(np.array(first['scores'])-second['scores']))),
                        float(np.max(np.abs(np.array(first['single_frame']['scores'])-second['single_frame']['scores']))))
            if ((first['case_id'], first['step'], first['choice_identity'], first['single_frame']['choice_identity'])
                    != (second['case_id'], second['step'], second['choice_identity'], second['single_frame']['choice_identity'])):
                raise AssertionError('checkpoint reload changed predictions')
        if error > 2e-5 or parameter_hash(model) != before:
            raise AssertionError('fixed-checkpoint scores or weights changed during evaluation')
        file = f'{entry["method"]}_seed{entry["seed"]}.json'
        write_json(directory/file, dict(predictions=rows, metrics=summarize(rows)))
        result = dict(**entry, file='evaluation/'+file, sha256=sha(directory/file),
                      metrics=summarize(rows), reload_max_abs_score_error=error,
                      candidate_order_max_abs_score_error=order_error, cached_matching_seconds=seconds,
                      parameter_sha256_before=before, parameter_sha256_after=parameter_hash(model))
        run['results'].append(result); write_json(directory/'run.json', run)
        print(entry['method'], entry['seed'], result['metrics']['all_prefixes'], flush=True)
        del model, restored
    load_plan(args)
    if (sha(output/'plan.json') != run['plan_sha256']
            or sha(output/'prepared.json') != run['prepared_sha256']):
        raise ValueError('test plan/features changed during evaluation')
    run['status'] = 'complete'; write_json(directory/'run.json', run)
    return run
