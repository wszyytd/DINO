"""Fixed-capacity gallery updates and asynchronous four-direction retention tests."""

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, pairwise

import torch
from torch.nn.functional import normalize

from .dense_views import prepare_reference
from .evidence_query_data import pixel_hash
from .evidence_query_prepare import encode_crop
from .four_view_memory import VIEWS, select_views
from .memory_retention import angular_gap, open_crop, score_pair
from .routing_foreground import keys_from_sample, spatial_cosines


CAPACITY = 4
METHODS = ("recurrent", "fifo", "keep", "mean", "latest", "unlimited")
SEQUENCES = {
    "repeat_after_four": ((0, 20, 16, 12, 23, 2, 22, 17, 13, 14), (12, 16, 20, 0, 2, 22, 23, 14, 13, 17)),
    "rear_arrives_late": ((0, 23, 2, 22, 20, 16, 12, 17, 13, 14), (22, 2, 0, 23, 16, 20, 12, 14, 17, 13)),
}
AFTER_FOUR_DIRECTIONS = {"repeat_after_four": 8, "rear_arrives_late": 14}


@dataclass(frozen=True)
class Prototype:
    camera: int
    first_camera: int
    keys: dict
    slots: torch.Tensor
    count: int = 1


def cosines(probes, bank):
    def stack(records, name):
        return normalize(torch.stack([record[name] for record in records]), dim=-1)

    mean_cosines = stack(probes, "foreground_mean") @ stack(bank, "foreground_mean").T
    return spatial_cosines(stack(probes, "foreground_grid"), stack(bank, "foreground_grid"), mean_cosines)


def update_bank(bank, incoming, policy, threshold, capacity=CAPACITY):
    event = dict(camera=incoming.camera)
    if policy == "fifo" and len(bank) == capacity:
        event.update(event="evicted_fifo", previous_camera=bank.pop(0).camera)
        bank.append(incoming)
        return event
    if bank and policy in ("keep", "mean", "latest"):
        similarity, index = cosines([incoming.keys], [p.keys for p in bank])[0].max(dim=0)
        slot, value = index.item(), similarity.item()
        previous = bank[slot]
        event.update(similarity=value, previous_camera=previous.camera, slot=slot)
        if value >= threshold:
            if policy == "mean":
                keys = {k: previous.keys[k] + incoming.keys[k] for k in incoming.keys}
                slots = previous.slots + (incoming.slots - previous.slots) / (previous.count + 1)
                bank[slot] = Prototype(incoming.camera, previous.first_camera, keys, slots, previous.count + 1)
            elif policy == "latest":
                bank[slot] = incoming
            event["event"] = dict(keep="kept", mean="averaged", latest="replaced_duplicate")[policy]
            return event
    if len(bank) == capacity and policy != "unlimited":
        return dict(event, event="rejected_full")
    bank.append(incoming)
    return dict(event, event="opened", slot=len(bank) - 1)


def choose_threshold(duplicates, novel):
    if not duplicates or not novel or not all(math.isfinite(x) for x in (*duplicates, *novel)):
        raise ValueError("threshold calibration needs finite scores for both labels")
    scores = sorted(set(duplicates) | set(novel))
    thresholds = [scores[0], *((a+b)/2 for a, b in pairwise(scores)), math.nextafter(scores[-1], math.inf)]
    positive, negative = len(duplicates), len(novel)
    best = None
    for threshold in thresholds:
        tp = sum(x >= threshold for x in duplicates)
        tn = sum(x < threshold for x in novel)
        # Compare balanced accuracy by integer counts. Prefer the stricter match on ties.
        score = (tp * negative + tn * positive, threshold)
        if best is None or score > best[0]:
            best = (score, dict(duplicate_correct=tp, duplicate_n=positive, novel_correct=tn, novel_n=negative))
    return best[0][1], best[1]


def describe_write(runtime, root, sample, crop):
    prepared, _ = prepare_reference(crop, runtime.cfg["image_size"])
    raw = runtime.tensor(runtime.raw([prepared])[0])
    return keys_from_sample(runtime, root, sample, raw)


@torch.inference_mode()
def calibrate(runtime, root, routes, max_duplicate_angle):
    rows = []
    for route in routes:
        if route["split"] != "train" or route["group"] != "spawn-42":
            raise ValueError("threshold calibration requires training location 42")
        for uid in route["identities"]:
            samples = [s for s in route["samples"].values() if s["identity"] == uid and s["role"] == "write"]
            keys = [describe_write(runtime, root, s, open_crop(root, s, foreground=True)) for s in samples]
            similarity = cosines(keys, keys).cpu()
            for i, j in combinations(range(len(samples)), 2):
                a, b = samples[i], samples[j]
                gap = angular_gap(a["azimuth"], b["azimuth"])
                if 0 < gap <= max_duplicate_angle:
                    kind = "duplicate"
                elif gap >= 45:
                    kind = "novel"
                else:
                    continue
                rows.append(dict(route=route["id"], identity=uid, cameras=[a["camera_index"], b["camera_index"]],
                                 gap=gap, label=kind, similarity=similarity[i, j].item()))
    duplicate, novel = ([r["similarity"] for r in rows if r["label"] == kind] for kind in ("duplicate", "novel"))
    threshold, stats = choose_threshold(duplicate, novel)
    return dict(threshold=threshold, duplicate_max_degrees=max_duplicate_angle, novel_min_degrees=45, **stats, pairs=rows)


def select_stream(route, max_gap):
    if route["group"] != "spawn-64":
        raise ValueError("prototype recall requires held-out location 64")
    probes = select_views(route, max_gap)
    cameras = set(SEQUENCES["repeat_after_four"][0])
    lookup = {(s["identity"], s["camera_index"]): k for k, s in route["samples"].items()}
    writes = {}
    for uid in route["identities"]:
        keys = {cam: lookup[uid, cam] for cam in sorted(cameras)}
        if not all(route["samples"][k].get("mask") for k in keys.values()):
            raise ValueError(f"prototype writes require vehicle masks: {uid}")
        writes[uid] = keys
    for field in ("frame", "rgb_sha256"):
        samples = route["samples"]
        w = {samples[k][field] for items in writes.values() for k in items.values()}
        q = {samples[p["query"]][field] for p in probes}
        if w & q:
            raise ValueError(f"prototype stream shares {field} with a held-out query")
    return writes, probes


@torch.inference_mode()
def encode_stream(runtime, root, model, route, writes, probes):
    samples = route["samples"]
    evidence = {}
    for uid, views in writes.items():
        for cam, key in views.items():
            sample = samples[key]
            crop = open_crop(root, sample, foreground=True)
            keys = describe_write(runtime, root, sample, crop)
            feature, valid = encode_crop(runtime, crop, "observations")
            feature, valid = runtime.tensor(feature), runtime.tensor(valid)
            slots = model.update(feature[None, None], valid[None, None])["slots"][0, 0]
            evidence[uid, cam] = (feature, valid, Prototype(cam, cam, keys, slots))
    references, query_keys = [], []
    for p in probes:
        sample = samples[p["query"]]
        crop = open_crop(root, sample, foreground=True)
        raw, _, feature, valid = runtime.reference(crop, pixel_hash(crop))
        query_keys.append(keys_from_sample(runtime, root, sample, raw[0]))
        references.append((runtime.tensor(feature), runtime.tensor(valid)))
    features, valid = (torch.stack(x) for x in zip(*references, strict=True))
    return evidence, features, valid, query_keys


def read_banks(model, banks, reference, valid, query_keys):
    slots, histories = [], []
    for bank in banks:
        values, picks = cosines(query_keys, [p.keys for p in bank]).max(dim=1)
        selected = [bank[i] for i in picks.tolist()]
        slots.append(torch.stack([view.slots for view in selected]))
        histories.append([dict(camera=view.camera, first_camera=view.first_camera, images=view.count, similarity=sim)
                          for view, sim in zip(selected, values.tolist(), strict=True)])
    state = model.initial_state(len(query_keys), len(banks), reference.device, reference.dtype)
    state["seen"].fill_(True)
    state["slots"] = torch.stack(slots, dim=1)
    return model.score(reference, valid, state), list(zip(*histories, strict=True))


@torch.inference_mode()
def replay(model, route, probes, encoded, threshold):
    evidence, references, reference_valid, query_keys = encoded
    ids = route["identities"]
    n = len(probes)
    rows, events = [], []
    for name, orders in SEQUENCES.items():
        state = None
        banks = {method: [[], []] for method in METHODS if method != "recurrent"}
        # Only one candidate writes per event. The two candidates can see different directions at every step.
        arrivals = [(i, cam) for cameras in zip(*orders, strict=True) for i, cam in enumerate(cameras)]
        for step, (idx, camera) in enumerate(arrivals, 1):
            uid = ids[idx]
            features, valid, incoming = evidence[uid, camera]
            observation = features.new_zeros((1, 2, *features.shape))
            observed = valid.new_zeros((1, 2, *valid.shape))
            observation[0, idx], observed[0, idx] = features, valid
            state = model.update(observation, observed, state)
            changes = {}
            for method, pair in banks.items():
                changes[method] = update_bank(pair[idx], incoming, method, threshold)
                changes[method]["banks"] = [[p.camera for p in bank] for bank in pair]
            events.append(dict(route=route["id"], sequence=name, step=step, identity=uid, camera=camera, writes=changes))
            if step == 1:
                continue
            expanded = {k: v.expand(n, *v.shape[1:]) for k, v in state.items()}
            scores = {"recurrent": (model.score(references, reference_valid, expanded).cpu().tolist(), [None]*n)}
            for method, pair in banks.items():
                logits, choices = read_banks(model, pair, references, reference_valid, query_keys)
                scores[method] = logits.cpu().tolist(), choices
            for i, p in enumerate(probes):
                label = ids.index(p["identity"])
                row = {m: dict(score_pair(logits[i], [True, True], label), logits=logits[i], selected=choices[i])
                       for m, (logits, choices) in scores.items()}
                rows.append(dict(route=route["id"], sequence=name, step=step, identity=p["identity"],
                                 view=p["view"], query=p["query"], candidates=ids, scores=row))
    return rows, events


def summarize(rows, events):
    """Compare fixed queries across steps and retain irreversible and recoverable errors."""
    summaries = []
    for route in sorted({r["route"] for r in rows}):
        for name in SEQUENCES:
            selected = [r for r in rows if (r["route"], r["sequence"]) == (route, name)]
            writes = [r["writes"] for r in events if (r["route"], r["sequence"]) == (route, name)]
            milestone = AFTER_FOUR_DIRECTIONS[name]
            final_step = max(r["step"] for r in selected)
            for method in METHODS:
                def right(r, mode=method):
                    return r["scores"][mode]["decision"] == "target"

                after_four = {r["query"] for r in selected if r["step"] == milestone and right(r)}
                wrong_later = {r["query"] for r in selected if r["step"] > milestone and not right(r)}
                last = [r for r in selected if r["step"] == final_step]
                by_view = {v: sum(right(r) for r in last if r["view"] == v) for v, _, _ in VIEWS}
                tail = None
                if name == "rear_arrives_late":
                    tail = [sum(right(r) for r in selected if r["step"] == step and r["view"] == "正车尾")
                            for step in (milestone - 2, milestone, final_step)]
                summaries.append(dict(route=route, sequence=name, method=method, after_four=len(after_four),
                    lost_ever=len(after_four & wrong_later), lost_final=sum(r["query"] in after_four and not right(r) for r in last),
                    final=sum(by_view.values()), by_view=by_view, questions=len(last), late_tail_hits=tail,
                    decisions=dict(Counter(w[method]["event"] for w in writes)) if method != "recurrent" else {}))
    return summaries


def report(summary, calibration):
    lines = ["# 实验八 · 固定容量长序列记忆", "",
             f"地点 42 的写入图对确定阈值 {calibration['threshold']:.5f}，在地点 64 评估。",
             f"地点 42 的阈值校准分类成绩为重复 {calibration['duplicate_correct']}/{calibration['duplicate_n']}，"
             f"新视角 {calibration['novel_correct']}/{calibration['novel_n']}。",
             "A、B 交替各写一张，并分别选择自己的读取原型。后续每一步复问相同的四方向照片。",
             "repeat_after_four 在写过四个方向后追加同方向照片。rear_arrives_late 先连续看车头，再看两侧和车尾。",
             "丢失按写过四个方向时答对的题计数，中途丢失包括最终恢复的题。", "",
             "| 车辆组 | 序列 | 方法 | 四方向已写时 | 最终 | 正车头 | 左前侧 | 左后侧 | 正车尾 | 中途丢失 | 最终丢失 | 满仓拒收 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary:
        n, old = r["questions"], r["after_four"]
        cells = [r["route"], r["sequence"], r["method"], f"{old}/{n}", f"{r['final']}/{n}",
                 *(f"{v}/2" for v in r["by_view"].values()), f"{r['lost_ever']}/{old}" if old else "—",
                 f"{r['lost_final']}/{old}" if old else "—", str(r["decisions"].get("rejected_full", 0))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## 晚到的车尾", "",
                  "| 车辆组 | 方法 | 第 12 步，车尾到来前 | 第 14 步，车尾到来后 | 最终 |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for r in summary:
        if r["late_tail_hits"] is not None:
            cells = [r["route"], r["method"], *(f"{c}/2" for c in r["late_tail_hits"])]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "FIFO、去重和不限容量共享前景 4×4 硬路由。",
                  "去重遇到低相似度且已满仓时拒收。完整的写入决策和每步逐题分数见 JSON。", ""])
    return "\n".join(lines)
