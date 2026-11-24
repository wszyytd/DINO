"""Held-out diagnostics for experiment 8. All probe histories are selected without scores."""

from collections import defaultdict

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .competitive_writer import CompetitiveWriter
from .memory_retention import angular_gap, score_pair
from .region_ab_data import write_json


def select_view_pair(route, uid, near, far):
    """One reproducible old/new angle pair per vehicle. Prefer matched controls and long streaks."""
    samples = route["samples"]
    writes = [s["writes"][uid] for s in route["steps"] if uid in s["writes"]]
    keys = [w["sample"] for w in writes]
    anchors = [w for w in writes if w["independent"]]

    def gap(left, right):
        return angular_gap(samples[left]["azimuth"], samples[right]["azimuth"])

    def neighbors(anchor, own, other, other_probe):
        return [k for k in keys if k not in (anchor, other)
                and max(gap(k, anchor), gap(k, own)) <= near
                and min(gap(k, other), gap(k, other_probe)) >= far]

    best, priority = None, None
    for first in anchors:
        a, qa = first["sample"], first["independent"]
        for second in anchors:
            b, qb = second["sample"], second["independent"]
            if samples[qa]["rgb_sha256"] == samples[qb]["rgb_sha256"]:
                continue
            if min(gap(a, b), gap(qa, qb), gap(a, qb), gap(b, qa)) < far:
                continue
            aa = neighbors(a, qa, b, qb)
            bb = neighbors(b, qb, a, qa)
            rank = (bool(aa and bb), min(3, len(bb)), gap(a, b))
            if priority is None or rank > priority:
                priority = rank
                best = dict(a=a, b=b, qa=qa, qb=qb, extra_a=aa[:1], extra_b=bb[:3],
                            azimuth_a=samples[a]["azimuth"], azimuth_b=samples[b]["azimuth"])
    return best


def _write_one(model, state, writes, key, label):
    tokens, mask = writes[key]
    observations = tokens.new_zeros(1, 2, *tokens.shape)
    valid = mask.new_zeros(1, 2, *mask.shape)
    observations[0, label], valid[0, label] = tokens, mask
    return model.update(observations, valid, state)


def probe_sequence(model, route, features, label, pair):
    writes, queries = features
    rival = route["identities"][1 - label]
    rival_key = next(s["writes"][rival]["sample"] for s in route["steps"] if rival in s["writes"])
    start = _write_one(model, None, writes, rival_key, 1 - label)
    a, b = pair["a"], pair["b"]
    cases = {"A": [a], "B": [b], "AB": [a, b], "BA": [b, a]}
    if pair["extra_a"]:
        cases["AA"] = [a, *pair["extra_a"]]
    if pair["extra_b"]:
        cases["BB"] = [b, pair["extra_b"][0]]
    history = [a, b]
    for key in pair["extra_b"]:
        history = [*history, key]
        cases["A" + "B" * (len(history) - 1)] = history
    probes = [queries[pair[k]] for k in ("qa", "qb")]
    tokens, masks = (torch.stack(col) for col in zip(*probes, strict=True))
    results = {}
    for name, keys in cases.items():
        state = start
        for key in keys:
            state = _write_one(model, state, writes, key, label)
        batch = {k: v.expand(2, *v.shape[1:]) for k, v in state.items()}
        logits = model.score(tokens, masks, batch).tolist()
        results[name] = {k: score_pair(row, [True, True], label)
                         for k, row in zip(("qa", "qb"), logits, strict=True)}
    return dict(rival=rival_key, **pair, scores=results)


def _off_diagonal_cosine(x):
    x = F.normalize(x, dim=-1)
    slots = x.shape[-2]
    mask = ~torch.eye(slots, dtype=torch.bool, device=x.device)
    return (x @ x.transpose(-1, -2))[..., mask].mean(-1)


def writer_observations(model, route, features):
    """Read per-head maps on real writes; the normal model's update computes temporal changes."""
    writer = model.writer
    writes, _ = features
    rows = []
    for label, uid in enumerate(route["identities"]):
        state = None
        for step in route["steps"]:
            if uid not in step["writes"]:
                continue
            key = step["writes"][uid]["sample"]
            x, valid = writes[key]
            x = x.masked_fill(~valid[:, None], 0)[None]
            query = model.initial
            if isinstance(writer, CompetitiveWriter):
                assignment, weights, _ = writer.distribute(query, x, valid[None])
            else:
                assignment = None
                _, weights = writer.attention(writer.norm(query), x, x, key_padding_mask=~valid[None],
                                              need_weights=True, average_attn_weights=False)
            written, _ = writer(query, x, valid[None])
            previous = state
            state = _write_one(model, state, writes, key, label)
            change = None if previous is None else (
                state["slots"][0, label] - previous["slots"][0, label]).norm(dim=-1).tolist()
            rows.append(dict(route=route["id"], step=step["step"], identity=uid, sample=key,
                             valid=valid.tolist(), weights=weights[0].tolist(),
                             assignment=None if assignment is None else assignment[0].tolist(),
                             attention_cosine_by_head=_off_diagonal_cosine(weights[0]).tolist(),
                             writer_slot_cosine=_off_diagonal_cosine(written).item(),
                             update_norm_by_slot=change))
    return rows


def _heatmap(rows, path):
    # Same first valid write of each route for all models; one panel per attention head.
    examples = {}
    for row in rows:
        examples.setdefault(row["route"], row)
    first = next(iter(examples.values()))
    heads, slots, count = len(first["weights"]), len(first["weights"][0]), len(first["valid"])
    cell, gap = 7, 12
    width, height = heads * (count * cell + gap), 32 + slots * cell
    img = Image.new("RGB", (max(width, 560), height * len(examples)), "white")
    draw = ImageDraw.Draw(img)
    for index, row in enumerate(examples.values()):
        top = index * height
        draw.text((0, top), f"{row['route']}  {row['sample']}  |  heads left to right", fill="black")
        for h, matrix in enumerate(row["weights"]):
            for s, line in enumerate(matrix):
                for n, value in enumerate(line):
                    x, y = h * (count * cell + gap) + n * cell, top + 18 + s * cell
                    level = round(255 * (1 - min(1, max(0, value))))
                    color = (level, level, 255) if row["valid"][n] else (120, 120, 120)
                    draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    img.save(path)


@torch.inference_mode()
def evaluate_writer(model, routes, features, plan, folder):
    model.eval()
    maps, pairs = [], []
    for route in routes:
        cache = features[route["id"]]
        maps.extend(writer_observations(model, route, cache))
        for label, uid in enumerate(route["identities"]):
            pair = select_view_pair(route, uid, plan["max_query_angle"], plan["novelty_angle"])
            result = probe_sequence(model, route, cache, label, pair) if pair else None
            pairs.append(dict(route=route["id"], identity=uid, result=result))
    _heatmap(maps, folder / "writer_heatmaps.png")
    write_json(folder / "writer_maps.json", maps)
    write_json(folder / "view_pairs.json", pairs)
    return dict(writes=len(maps), attention_cosine=sum(
        sum(r["attention_cosine_by_head"]) / len(r["attention_cosine_by_head"]) for r in maps) / len(maps),
        writer_cosine=sum(r["writer_slot_cosine"] for r in maps) / len(maps),
        eligible=sum(p["result"] is not None for p in pairs), candidates=len(pairs),
        probes=summarize_pairs(pairs))


def summarize_pairs(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        if pair["result"] is None:
            continue
        scores = pair["result"]["scores"]
        for case, reads in scores.items():
            groups[case].append((reads, scores["A"]["qa"]["decision"] == "target"))
        if "AA" in scores and "BB" in scores:
            for case in ("AA", "BB", "AB", "BA"):
                groups["matched-" + case].append((scores[case], scores["A"]["qa"]["decision"] == "target"))
    result = {}
    for case, group in groups.items():
        reads = [read for read, _ in group]
        result[case] = dict(total=len(reads), old=sum(r["qa"]["decision"] == "target" for r in reads),
                            new=sum(r["qb"]["decision"] == "target" for r in reads),
                            both=sum(all(x["decision"] == "target" for x in r.values()) for r in reads),
                            initially_correct=sum(right for _, right in group),
                            lost=sum(right and r["qa"]["decision"] != "target" for r, right in group),
                            old_margin=sum(r["qa"]["margin"] for r in reads) / len(reads),
                            new_margin=sum(r["qb"]["margin"] for r in reads) / len(reads))
    return result


def writer_report(diagnostics, validations):
    lines = ["# Experiment 8 – Writer slot competition", "",
             "Validation on held-out routes of the same two vehicle assets. Cosines measure redundancy, "
             "with attention compared within each head. The heatmap rows are slots, columns are input tokens.",
             "", "| Model | Writes | Mean attention cosine | Mean Writer slot cosine | Eligible angle pairs |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for model, d in diagnostics.items():
        lines.append(f"| {model} | {d['writes']} | {d['attention_cosine']:.4f} | "
                     f"{d['writer_cosine']:.4f} | {d['eligible']}/{d['candidates']} |")
    lines += ["", "## Original routes", "",
              "| Model | Route | Paired new pre → post | New help / harm | Old reads correct | Distinct old probes | "
              "Initially correct old reads later wrong |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, routes in validations.items():
        for route, metrics in routes.items():
            new = metrics["paired_independent"]
            old = metrics["groups"].get("old_view", {})
            lines.append(f"| {model} | {route} | {new['pre_correct']} → {new['post_correct']} / {new['total']} | "
                         f"{new['help']} / {new['harm']} | {old.get('correct_after', 0)}/{old.get('total', 0)} | "
                         f"{old.get('unique_probes', 0)} | {old.get('later_wrong', 0)}/{old.get('initially_correct', 0)} |")
    lines += ["", "## Controlled histories", "",
              "A and B denote two camera angles of the target vehicle. The two held-out query photos and the "
              "rival memory stay fixed. Matched rows use only cases with both two-photo controls. "
              "Longer B streaks require distinct photos; their denominators can be smaller. "
              "The lost column compares the same A query against its decision immediately after A.",
              "", "| Model | History | Cases | A correct | B correct | Both correct | A lost / originally correct | "
              "Mean A / B margin |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, d in diagnostics.items():
        if not d["probes"]:
            lines.append(f"| {model} | No eligible pairs | 0 | — | — | — | — | — |")
        for case, m in d["probes"].items():
            lines.append(f"| {model} | {case} | {m['total']} | {m['old']} | {m['new']} | {m['both']} | "
                         f"{m['lost']}/{m['initially_correct']} | {m['old_margin']:.3f} / {m['new_margin']:.3f} |")
    return "\n".join(lines) + "\n"
