"""Train the gallery readout on held-out query crops from each observed direction."""

import random
from itertools import combinations
from statistics import mean

import torch
from torch.nn import functional as F

from .four_view_memory import VIEWS
from .memory_retention import score_pair
from .symmetric_memory_eval import _write, gallery_state


TRAIN_VIEWS = tuple((f"cam{w:02}", w, q) for w, q in ((21, 22), (0, 1), (3, 4), (5, 6), (7, 8)))


def freeze_for_gallery(model):
    model.eval().requires_grad_(False)
    model.reader.requires_grad_(True)
    model.match.requires_grad_(True)


@torch.no_grad()
def cache_route(model, route, pairs, observations, queries):
    # Caches are ordinary detached tensors. Autograd needs them to differentiate the Reader.
    ids = route["identities"]
    frames = []
    for position in sorted({p["position"] for p in pairs}):
        current = [p for p in pairs if p["position"] == position]
        if [p["identity"] for p in current] != ids:
            raise ValueError(f"candidate order differs between views: {route['id']}")
        frames.append(_write(model, [p["write"] for p in current], observations))
    probes = [p for p in pairs if p["query"] is not None]
    tokens, valid = (torch.stack(items) for items in zip(*(queries[p["query"]] for p in probes), strict=True))
    labels = torch.tensor([ids.index(p["identity"]) for p in probes], device=tokens.device)
    return dict(frames=frames, pairs=probes, ids=ids, tokens=tokens, valid=valid, labels=labels, split=route["split"])


def subset_schedule(n_views, epochs, seed, *, rear_views=()):
    """Cycle through all subsets at each length before repeating one."""
    rng = random.Random(seed)
    rear = set(rear_views)
    by_size = []
    for size in (2, 3, 4):
        subsets = list(combinations(range(n_views), size))
        if rear:
            subsets = [subset for subset in subsets if rear.intersection(subset) and set(subset) - rear]
        if not subsets:
            raise ValueError(f"no valid gallery subsets of size {size}")
        order = []
        while len(order) < epochs:
            rng.shuffle(subsets)
            order.extend(subsets)
        by_size.append(order[:epochs])
    return list(zip(*by_size, strict=True))


def read_gallery(model, cache, subset, indices):
    state = gallery_state([cache["frames"][i] for i in subset])
    batch = {k: v.expand(len(indices), *v.shape[1:]) for k, v in state.items()}
    return model.score(cache["tokens"][indices], cache["valid"][indices], batch)


def train_epoch(model, optimizer, caches, subsets, seed):
    model.reader.train()
    model.match.train()
    routes = list(caches)
    random.Random(seed).shuffle(routes)
    losses = {}
    updates = 0
    for start in range(0, len(routes), 2):
        batch = routes[start:start+2]
        optimizer.zero_grad(set_to_none=True)
        for route in batch:
            cache = caches[route]
            if cache["split"] != "train":
                raise ValueError("only training routes may contribute gradients")
            terms = []
            for subset in subsets[route]:
                indices = [i for i,p in enumerate(cache["pairs"]) if p["position"]-1 in subset]
                logits = read_gallery(model, cache, subset, indices)
                terms.append(F.cross_entropy(logits, cache["labels"][indices]))
            loss = torch.stack(terms).mean()
            (loss / len(batch)).backward()
            losses[route] = loss.item()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1., error_if_nonfinite=True)
        optimizer.step()
        updates += 1
    return dict(updates=updates, by_route=losses, mean_ce=mean(losses.values()))


@torch.inference_mode()
def evaluate(model, caches):
    model.eval()
    full = tuple(range(len(VIEWS)))
    histories = {f"prefix_{n}": full[:n] for n in range(1, 5)}
    for i, (_, cam, _) in enumerate(VIEWS):
        histories[f"single_{cam:02}"] = (i,)
        histories[f"drop_{cam:02}"] = tuple(j for j in full if j != i)
    rows = []
    for route, cache in caches.items():
        if cache["split"] != "val":
            raise ValueError("four-view evaluation requires held-out routes")
        indices = list(range(len(cache["pairs"])))
        logits = {}
        for subset in dict.fromkeys(histories.values()):
            values = read_gallery(model, cache, subset, indices)
            if not torch.isfinite(values).all():
                raise ValueError(f"non-finite gallery scores: {route}, {subset}")
            logits[subset] = values.cpu().tolist()
        for i, p in enumerate(cache["pairs"]):
            label = cache["ids"].index(p["identity"])
            conditions = {}
            for key, subset in histories.items():
                values = logits[subset][i]
                conditions[key] = dict(score_pair(values, [True, True], label), logits=values,
                                       cameras=[VIEWS[v][1] for v in subset])
            all_views = conditions["prefix_4"]
            for key, scores in conditions.items():
                if key.startswith("drop_"):
                    scores["margin_change"] = scores["margin"] - all_views["margin"]
            rows.append(dict(route=route, identity=p["identity"], probe=p["query"], view=p["view"],
                             query_camera=p["query_camera"], candidates=cache["ids"], conditions=conditions))
    return rows


def report(stages):
    lines = ["# 实验八 Gallery Reader", "",
             "prefix 表示前 1～4 张，single 表示单存该机位。drop 表示两辆车同时去掉指定机位，每辆均剩 24 槽。",
             "每个阶段和条件都使用同一批查询。分差为正确车得分减对手车得分。",
             "抽走后的改对、改错和分差变化均相对同阶段四图完整 gallery。"]
    for stage, rows in stages.items():
        lines.extend(["", f"## {stage}", "", "| 车辆组 | 条件 | 正车头 | 左前侧 | 左后侧 | 正车尾 | 总答对 |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for route in sorted({r["route"] for r in rows}):
            group = [r for r in rows if r["route"] == route]
            for key in group[0]["conditions"]:
                counts = []
                for name, _, _ in VIEWS:
                    values = [r["conditions"][key]["decision"] for r in group if r["view"] == name]
                    counts.append(f"{values.count('target')}/{len(values)}")
                total = sum(r["conditions"][key]["decision"] == "target" for r in group)
                lines.append("| " + " | ".join([route, key, *counts, f"{total}/{len(group)}"]) + " |")
        lines.extend(["", "### 同一查询抽去不同照片", "",
                      "各列标出判定相对四图的变化及分差变化量。", "",
                      "| 查询图 | 四图判定 | 四图分差 | 去车头 | 去左前 | 去左后 | 去车尾 |",
                      "| --- | --- | ---: | --- | --- | --- | --- |"])
        for r in rows:
            cond = r["conditions"]
            full = cond["prefix_4"]
            cells = [r["probe"], full["decision"], f"{full['margin']:+.3f}"]
            for _, cam, _ in VIEWS:
                drop = cond[f"drop_{cam:02}"]
                old, new = full["decision"], drop["decision"]
                change = "保持" if new == old else "改对" if new == "target" else "改错" if old == "target" else "仍错"
                cells.append(f"{change} {drop['margin_change']:+.3f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
