"""Head training and evaluation with explicit split and feature provenance checks."""

import csv
import html
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from .core import checked_path, prepare_image
from .local_features import FeatureCache, cls_cosine, resolve_device
from .local_pairs import check_split, ranking_metrics, read_manifest, sha256, verify_images


def validate_training_setup(train, validation, allow_train_only):
    if {r['label'] for r in train} != {0, 1}:
        raise ValueError('training needs both positive and negative confirmed pairs')
    if validation is None:
        if not allow_train_only:
            raise ValueError('no validation set: explicitly use --allow-train-only for a pilot')
    else:
        if {r['label'] for r in validation} != {0, 1}:
            raise ValueError('validation needs both classes')
        check_split(train, validation)


def check_evaluation_scope(rows, checkpoint, allow_training):
    overlap = bool({r['group'] for r in rows} & set(checkpoint['training_groups']))
    images = {r[k] for r in rows for k in ('reference_sha256', 'query_sha256')}
    overlap = overlap or bool(images & set(checkpoint['training_images']))
    if overlap and not allow_training:
        raise ValueError('evaluation overlaps training data; use --allow-training-data for diagnostics')
    return 'training_overlap' if overlap else 'held_out'


def load_feature_bank(rows, cache):
    hashes = {r[k] for r in rows for k in ('reference_sha256', 'query_sha256')}
    return {h: cache.get(h) for h in sorted(hashes)}


def tensors_for(rows, bank, device, torch):
    def stack(role, key):
        return torch.from_numpy(np.stack([bank[r[role + '_sha256']][key] for r in rows])).to(device)
    return (stack('reference', 'tokens').float(), stack('reference', 'valid').bool(),
            stack('query', 'tokens').float(), stack('query', 'valid').bool())


def predict(model, rows, bank, device, batch_size, torch, attention=False):
    model.eval()
    logits, maps = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output = model(*tensors_for(batch, bank, device, torch))
            logits.extend(output['logit'].cpu().tolist())
            if attention:
                maps.extend(output['query_attention'].cpu().numpy())
    if not np.isfinite(logits).all():
        raise RuntimeError('nonfinite model prediction')
    return logits, maps


def train_head(args):
    import torch
    from .local_attention import LocalMatcher
    train_manifest = read_manifest(args.train_manifest)
    val_manifest = read_manifest(args.val_manifest) if args.val_manifest else None
    rows = train_manifest['pairs']
    val_rows = val_manifest['pairs'] if val_manifest else None
    validate_training_setup(rows, val_rows, args.allow_train_only)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        raise ValueError('positive epochs, batch size, learning rate and thread count required')
    if args.output.exists():
        raise ValueError('training output exists; choose a new run directory')
    cache = FeatureCache(args.cache)
    bank = load_feature_bank(rows + (val_rows or []), cache)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device, torch)
    model = LocalMatcher(input_dim=cache.config['input_dim'], layers=len(cache.config['layers']),
                         dim=args.dim, heads=args.heads, probes=args.probes,
                         reference_grid=args.reference_grid, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positive = sum(r['label'] for r in rows)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(rows) - positive) / positive, device=device))
    provenance = {'format_version': 1, 'model_config': model.config,
                  'feature_config': cache.config, 'seed': args.seed,
                  'training_groups': sorted({r['group'] for r in rows}),
                  'training_images': sorted(bank_hash for bank_hash in {
                      r[k] for r in rows for k in ('reference_sha256', 'query_sha256')}),
                  'train_manifest_sha256': sha256(args.train_manifest),
                  'validation_manifest_sha256': sha256(args.val_manifest) if val_rows else None,
                  'warning': 'Match scores are uncalibrated; no candidate exclusion implemented.'}
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', mode='held_out_validation' if val_rows else 'train_only_pilot',
                  feature_config=cache.config, model_config=model.config,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  torch_version=str(torch.__version__), device=str(device),
                  train_pairs=len(rows), val_pairs=len(val_rows or []))
    marker = args.output / 'training_run.json'
    marker.write_text(json.dumps(status, indent=2), encoding='utf-8')
    history, best_val = [], float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(len(rows)).tolist()
        loss_sum = 0.0
        for start in range(0, len(rows), args.batch_size):
            batch = [rows[i] for i in indices[start:start + args.batch_size]]
            labels = torch.tensor([r['label'] for r in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*tensors_for(batch, bank, device, torch))['logit'], labels)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss; incomplete output retained for diagnostics')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        record = {'epoch': epoch, 'train_loss': loss_sum / len(rows)}
        if val_rows:
            val_logits, _ = predict(model, val_rows, bank, device, args.batch_size, torch)
            val_loss = float(criterion(torch.tensor(val_logits, device=device),
                                      torch.tensor([r['label'] for r in val_rows],
                                                   dtype=torch.float32, device=device)))
            record['val_loss'] = val_loss
            record['val_ranking'] = ranking_metrics(val_rows, val_logits)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(dict(provenance, epoch=epoch, model_state=model.state_dict()),
                           args.output / 'best.pt')
        history.append(record)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'Epoch {epoch}/{args.epochs}: train_loss={record["train_loss"]:.5f}'
              + (f' val_loss={record["val_loss"]:.5f}' if val_rows else ' [train-only pilot]'),
              flush=True)
    torch.save(dict(provenance, epoch=args.epochs, model_state=model.state_dict()),
               args.output / 'last.pt')
    status.update(status='complete', epochs=args.epochs,
                  checkpoint_sha256=sha256(args.output / 'last.pt'))
    marker.write_text(json.dumps(status, indent=2), encoding='utf-8')
    return status


def evaluate_head(args):
    import torch
    from .local_attention import LocalMatcher
    manifest = read_manifest(args.manifest)
    verify_images(manifest, args.data_root)
    rows = manifest['pairs']
    cache = FeatureCache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint['feature_config'] != cache.config:
        raise ValueError('checkpoint and cache backbone/preprocessing contracts differ')
    scope = check_evaluation_scope(rows, checkpoint, args.allow_training_data)
    if args.output.exists() or args.batch_size < 1:
        raise ValueError('output must be new and batch size must be positive')
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device, torch)
    model = LocalMatcher(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    bank = load_feature_bank(rows, cache)
    logits, maps = predict(model, rows, bank, device, args.batch_size, torch, True)
    baseline = [cls_cosine(bank[r['reference_sha256']]['cls'], bank[r['query_sha256']]['cls'])
                for r in rows]
    summary = {'status': 'running', 'scope': scope, 'pairs': len(rows),
               'manifest_sha256': sha256(args.manifest),
               'checkpoint_sha256': sha256(args.checkpoint), 'feature_config': cache.config,
               'learned': ranking_metrics(rows, logits), 'cls_cosine': ranking_metrics(rows, baseline),
               'warning': 'Training overlap is fitting diagnostics only. Scores are uncalibrated. '
                          'Attention is not verified semantic evidence. No automatic exclusion.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fields = ['scene', 'sample_id', 'candidate', 'view', 'label', 'comparison_group',
              'match_logit', 'match_score_uncalibrated', 'cls_cosine']
    with (args.output / 'scores.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, logit, cosine in zip(rows, logits, baseline, strict=True):
            writer.writerow(dict({k: row.get(k, '') for k in fields[:6]}, match_logit=logit,
                                 match_score_uncalibrated=float(torch.sigmoid(torch.tensor(logit))),
                                 cls_cosine=cosine))
    np.savez_compressed(args.output / 'attention.npz', weights=np.stack(maps),
                        query_sha256=np.array([r['query_sha256'] for r in rows]))
    build_report(args.output, rows, logits, baseline, maps, args.data_root,
                 cache.config['image_size'], scope)
    summary['status'] = 'complete'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def build_report(output, rows, logits, baseline, maps, data_root, size, scope):
    assets = output / 'assets'
    assets.mkdir()
    cards = []
    for i, (row, logit, cosine, attention) in enumerate(zip(rows, logits, baseline, maps, strict=True)):
        for role in ('reference', 'query'):
            with Image.open(checked_path(data_root, row[role])) as image:
                prepared = prepare_image(image, size)
            prepared.save(assets / f'{i}_{role}.png')
            if role == 'query':
                side = int(np.sqrt(attention.shape[-1]))
                heat = attention.mean(0).reshape(side, side)
                # Per-image normalization is display only, never an evidence strength measure.
                heat = np.clip(heat / max(float(heat.max()), 1e-12), 0, 1)
                alpha = Image.fromarray((heat * 150).astype(np.uint8)).resize(
                    prepared.size, Image.Resampling.BILINEAR)
                red = Image.new('RGB', prepared.size, (255, 0, 0))
                Image.composite(red, prepared, alpha).save(assets / f'{i}_attention.png')
        cards.append(f'<article><h2>{html.escape(row.get("scene", ""))} / '
                     f'{html.escape(row["sample_id"])}</h2><p>真值 {row["label"]}，'
                     f'匹配logit {logit:.4f}，CLS余弦 {cosine:.4f}</p>'
                     f'<figure><img src="assets/{i}_reference.png"><figcaption>参考</figcaption></figure>'
                     f'<figure><img src="assets/{i}_query.png"><figcaption>候选</figcaption></figure>'
                     f'<figure><img src="assets/{i}_attention.png"><figcaption>'
                     '学习查询的平均注意力，每图归一化</figcaption></figure></article>')
    warning = ('训练数据重合，仅检查拟合和数据流' if scope == 'training_overlap'
               else '与已记录训练组和图片无重合，仍需审查场景独立性')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>局部注意力匹配实验</title><style>body{font:17px sans-serif;max-width:1100px;'
            'margin:30px auto;color:#222}article{border-top:1px solid #aaa;padding:18px 0}'
            'figure{display:inline-block;margin:8px}img{width:240px}figcaption{font-size:14px}'
            '</style><h1>参考图与候选图的局部注意力匹配</h1><p>' + warning
            + '</p><p>分数尚未校准。注意力没有车门、车窗等已验证语义，也不证明匹配正确。'
              '当前不执行候选排除。</p>' + ''.join(cards) + '</html>')
    (output / 'index.html').write_text(page, encoding='utf-8')
