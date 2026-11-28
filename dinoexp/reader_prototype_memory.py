"""Separate max-score reading, bidirectional Reader deduplication, and full-bank replacement."""

from itertools import combinations

import torch

from .memory_retention import angular_gap, score_pair
from .prototype_memory import (
    CAPACITY, SEQUENCES, angle_label, choose_threshold, cosines, encode_observation, read_banks,
    retention_tables, update_bank,
)


METHODS = ("grid_keep_4x4", "grid_keep_max", "reader_keep_max", "reader_replace_max", "unlimited_4x4", "unlimited_max")


def score_slots(model, references, valid, prototypes):
    slots = torch.stack([p.slots for p in prototypes])
    state = model.initial_state(len(references), len(prototypes), references.device, references.dtype)
    state["seen"].fill_(True)
    state["slots"] = slots[None].expand(len(references), -1, -1, -1)
    logits = model.score(references, valid, state)
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite Reader comparison")
    return logits


def max_reader_scores(logits, counts):
    """Maximize per candidate over its own prototypes and keep the Reader's Null score."""
    grouped = logits[:, :-1].split(counts, dim=1)
    values, picks = zip(*(scores.max(dim=1) for scores in grouped), strict=True)
    return torch.cat((torch.stack(values, dim=1), logits[:, -1:]), dim=1), torch.stack(picks, dim=1)


def read_max(model, banks, reference, valid):
    raw = score_slots(model, reference, valid, [p for bank in banks for p in bank])
    logits, selections = max_reader_scores(raw, tuple(map(len, banks)))
    chosen = []
    for indices, values in zip(selections.tolist(), logits.tolist(), strict=True):
        chosen.append([dict(camera=bank[i].camera, reader_score=score)
                       for bank, i, score in zip(banks, indices, values[:-1], strict=True)])
    return logits, chosen


def update_reader_bank(bank, incoming, directed, threshold, *, capacity=CAPACITY, evict=False):
    def compare(a, b):
        forward, reverse = directed[a.camera, b.camera], directed[b.camera, a.camera]
        return dict(new_reads_old=forward, old_reads_new=reverse, bidirectional=min(forward, reverse))

    event = dict(camera=incoming.camera)
    if bank:
        comparisons = [dict(compare(incoming, p), old_camera=p.camera) for p in bank]
        best = max(comparisons, key=lambda score: score["bidirectional"])
        event.update(best, comparisons=comparisons)
        if best["bidirectional"] >= threshold:
            return dict(event, event="kept")
    if len(bank) < capacity:
        bank.append(incoming)
        return dict(event, event="opened", slot=len(bank)-1)
    if not evict:
        return dict(event, event="rejected_full")

    proposals = [*bank, incoming]
    pairs = [(i, j, compare(a, b)) for (i, a), (j, b) in combinations(enumerate(proposals), 2)]
    i, j, score = max(pairs, key=lambda pair: pair[2]["bidirectional"])
    event["redundant_pair"] = dict(cameras=[proposals[i].camera, proposals[j].camera], score=score["bidirectional"])
    if j == len(bank):
        return dict(event, event="rejected_full")
    # Preserve the older element of the most redundant pair. Keep complete observations.
    removed = bank.pop(j)
    bank.append(incoming)
    return dict(event, event="replaced_redundant", evicted_camera=removed.camera)


def compare_observations(model, route, observations, write_references, max_gap):
    samples = {(s["identity"], s["camera_index"]): s for s in route["samples"].values()}
    scores, rows = {}, []
    for uid in route["identities"]:
        prototypes = [p for (identity, _), (_, _, p) in observations.items() if identity == uid]
        cams = [p.camera for p in prototypes]
        references, valid = (torch.stack(x) for x in zip(*(write_references[uid, c] for c in cams), strict=True))
        logits = score_slots(model, references, valid, prototypes)[:, :-1].cpu()
        keys = [p.keys for p in prototypes]
        grid = cosines(keys, keys).cpu()
        scores[uid] = {(a, b): logits[i, j].item() for i, a in enumerate(cams) for j, b in enumerate(cams) if i != j}
        for (i, a), (j, b) in combinations(enumerate(cams), 2):
            gap = angular_gap(samples[uid, a]["azimuth"], samples[uid, b]["azimuth"])
            u, v = scores[uid][a, b], scores[uid][b, a]
            rows.append(dict(route=route["id"], identity=uid, cameras=[a, b], gap=gap, label=angle_label(gap, max_gap),
                             grid=grid[i, j].item(), reader_ab=u, reader_ba=v, reader=min(u, v)))
    return scores, rows


@torch.inference_mode()
def calibrate_reader(runtime, root, model, routes, max_gap):
    rows = []
    for route in routes:
        if route["group"] != "spawn-42" or route["split"] != "train":
            raise ValueError("Reader threshold calibration requires training location 42")
        observations, references = {}, {}
        for s in route["samples"].values():
            if s["role"] == "write":
                key = s["identity"], s["camera_index"]
                observations[key], references[key] = encode_observation(runtime, root, model, s, with_reference=True)
        _, pairs = compare_observations(model, route, observations, references, max_gap)
        rows.extend(pairs)
    thresholds = {}
    for kind in ("grid", "reader"):
        dup = [r[kind] for r in rows if r["label"] == "duplicate"]
        novel = [r[kind] for r in rows if r["label"] == "novel"]
        value, stats = choose_threshold(dup, novel)
        thresholds[kind] = dict(threshold=value, **stats)
    return dict(thresholds=thresholds, duplicate_max_degrees=max_gap, novel_min_degrees=45, pairs=rows)


def classification(rows, calibration):
    result = {}
    for kind, fit in calibration["thresholds"].items():
        threshold = fit["threshold"]
        duplicates = [r for r in rows if r["label"] == "duplicate"]
        novel = [r for r in rows if r["label"] == "novel"]
        false_repeats = [r for r in novel if r[kind] >= threshold]
        result[kind] = dict(duplicate_n=len(duplicates), duplicate_correct=sum(r[kind] >= threshold for r in duplicates),
                            novel_n=len(novel), novel_correct=len(novel)-len(false_repeats),
                            opposing_false_repeats=sum(r["gap"] >= 135 for r in false_repeats))
    return result


@torch.inference_mode()
def replay_reader(model, route, probes, encoded, directed, thresholds):
    evidence, references, valid, query_keys = encoded
    ids = route["identities"]
    rows, events = [], []
    for name, orders in SEQUENCES.items():
        banks = {m: [[], []] for m in ("grid_keep", "reader_keep", "reader_replace", "unlimited")}
        arrivals = [(idx, cam) for frames in zip(*orders, strict=True) for idx, cam in enumerate(frames)]
        for step, (idx, camera) in enumerate(arrivals, 1):
            uid = ids[idx]
            _, _, incoming = evidence[uid, camera]
            decisions = {
                "grid_keep": update_bank(banks["grid_keep"][idx], incoming, "keep", thresholds["grid"]["threshold"]),
                "reader_keep": update_reader_bank(banks["reader_keep"][idx], incoming, directed[uid], thresholds["reader"]["threshold"]),
                "reader_replace": update_reader_bank(banks["reader_replace"][idx], incoming, directed[uid],
                                                     thresholds["reader"]["threshold"], evict=True),
                "unlimited": update_bank(banks["unlimited"][idx], incoming, "unlimited", thresholds["grid"]["threshold"]),
            }
            changes = {}
            reads = {}
            for method in METHODS:
                bank, mode = method.rsplit("_", 1)
                pair = banks[bank]
                changes[method] = dict(decisions[bank], banks=[[p.camera for p in b] for b in pair])
                if step > 1:
                    if mode == "max":
                        logits, chosen = read_max(model, pair, references, valid)
                    else:
                        logits, chosen = read_banks(model, pair, references, valid, query_keys)
                    reads[method] = logits.cpu().tolist(), chosen
            events.append(dict(route=route["id"], sequence=name, step=step, identity=uid, camera=camera, writes=changes))
            if step == 1:
                continue
            for i, p in enumerate(probes):
                label = ids.index(p["identity"])
                scores = {m: dict(score_pair(logits[i], [True, True], label), logits=logits[i], selected=chosen[i])
                          for m, (logits, chosen) in reads.items()}
                rows.append(dict(route=route["id"], sequence=name, step=step, identity=p["identity"], view=p["view"],
                                 query=p["query"], candidates=ids, scores=scores))
    return rows, events


def report_reader(summary, calibration, held_out):
    lines = ["# 实验八 · Reader 逐原型最高分与双向去重", "",
             "对同一候选车逐原型读取，取最高的原始身份分，再与另一车和 Null 一起比较。",
             "grid_keep_4x4 与 grid_keep_max 共用同一个原型库；unlimited 两组也共用同一个库。",
             "grid_keep_max 与 reader_keep_max 只改变写入判重。Reader 判重取两个方向分数的较小值。",
             "reader_replace_max 在满仓且新照片通过判重时，从五份中找最可替代的一对，保留较早的完整原型。",
             "地点 42 决定阈值，地点 64 的位姿只用于检查判重。", "",
             "| 描述符 | 阈值 | 地点 42 近角度 | 地点 42 新视角 | 地点 64 近角度 | 地点 64 新视角 | 地点 64 反面误合 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, trained in calibration["thresholds"].items():
        val = held_out[method]
        lines.append(f"| {method} | {trained['threshold']:.5f} | "
                     f"{trained['duplicate_correct']}/{trained['duplicate_n']} | {trained['novel_correct']}/{trained['novel_n']} | "
                     f"{val['duplicate_correct']}/{val['duplicate_n']} | {val['novel_correct']}/{val['novel_n']} | "
                     f"{val['opposing_false_repeats']} |")
    lines.extend(retention_tables(summary))
    lines.extend(["", "丢失按四方向已写时答对的题计数。两种读取的逐题得分，以及每次写入的双向分数与腾位事件，见 JSON。", ""])
    return "\n".join(lines)
