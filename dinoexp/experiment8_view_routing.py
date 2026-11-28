"""Compare query-based view selection with the frozen four-view gallery."""

import math
from statistics import mean

import torch

from .dense_views import prepare_reference
from .evidence_query_data import pixel_hash
from .evidence_query_prepare import encode_crop
from .four_view_memory import VIEWS
from .gallery_reader import cache_route, evaluate
from .memory_retention import open_crop, score_pair
from .routing_foreground import keys_from_sample, spatial_cosines


CAMERAS = tuple(w for _, w, _ in VIEWS)
SINGLES = tuple(f"single_{cam:02}" for cam in CAMERAS)
DROPS = tuple(f"drop_{cam:02}" for cam in CAMERAS)
DEFAULT_TEMPERATURE = .05
POOLING = ("global_mean", "foreground_mean", "foreground_grid")


def pool_view_key(raw, valid):
    """Use the deepest selected DINOv2 layer and exclude padding-only patches."""
    key = raw[-1, valid.bool()].float().mean(0)
    norm = key.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("empty or non-finite DINOv2 view descriptor")
    return key / norm


@torch.inference_mode()
def encode_route(runtime, root, model, route, pairs, *, compare_pooling=False):
    observations, queries = {}, {}
    write_keys, query_keys = {}, {}

    def describe(sample, raw, valid):
        baseline = pool_view_key(raw, valid)
        if not compare_pooling:
            return baseline
        return dict(global_mean=baseline, **keys_from_sample(runtime, root, sample, raw))

    for p in pairs:
        w, q = p["write"], p["query"]
        write, query = route["samples"][w], route["samples"][q]
        obs = open_crop(root, write, foreground=True)
        features, valid = encode_crop(runtime, obs, "observations")
        observations[w] = runtime.tensor(features), runtime.tensor(valid)
        prepared, keep = prepare_reference(obs, runtime.cfg["image_size"])
        write_keys[w] = describe(write, runtime.tensor(runtime.raw([prepared])[0]), runtime.tensor(keep))

        probe = open_crop(root, query, foreground=True)
        raw, keep, features, valid = runtime.reference(probe, pixel_hash(probe))
        queries[q] = runtime.tensor(features), runtime.tensor(valid)
        query_keys[q] = describe(query, raw[0], keep[0])
    cache = cache_route(model, route, pairs, observations, queries)
    return cache, write_keys, query_keys


def route_logits(similarities, single_logits, *, temperature):
    """Use one cosine average and one set of weights for both candidates."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("routing temperature must be finite and positive")
    if similarities.shape[1:] != (len(CAMERAS), 2) or single_logits.shape != (*similarities.shape[:2], 3):
        raise ValueError("expected query × camera × candidate similarities and A/B/Null logits")
    avg = similarities.mean(dim=-1).to(dtype=single_logits.dtype)
    if not torch.isfinite(avg).all():
        raise ValueError("non-finite view similarities")
    rows = torch.arange(len(avg), device=avg.device)
    states = {}
    for condition, removed in (("full", None), *zip(DROPS, range(len(CAMERAS)), strict=True)):
        masked = avg.clone()
        if removed is not None:
            masked[:, removed] = -torch.inf
        picked = masked.argmax(dim=-1)
        # Shift before dividing so even a very small temperature retains a finite maximum.
        weights = ((masked - masked.max(dim=-1, keepdim=True).values) / temperature).softmax(dim=-1)
        states[condition] = dict(picked=picked, weights=weights, hard=single_logits[rows, picked],
                                 soft=(weights[..., None] * single_logits).sum(dim=1))
    return states


@torch.inference_mode()
def evaluate_routes(model, encoded, temperature=DEFAULT_TEMPERATURE, *, compare_pooling=False):
    caches = {route: record[0] for route, record in encoded.items()}
    rows = evaluate(model, caches)
    if compare_pooling:
        return {method: attach_routing(rows, encoded, temperature, method) for method in POOLING}
    return attach_routing(rows, encoded, temperature, None)


def view_similarities(writes, probes, method):
    def stack(values, kind):
        return torch.stack([v[kind] for v in values] if kind else values)

    source = "foreground_mean" if method == "foreground_grid" else method
    cosines = stack(probes, source) @ stack(writes, source).T
    if method == "foreground_grid":
        cosines = spatial_cosines(stack(probes, method), stack(writes, method), cosines)
    return cosines.reshape(-1, len(CAMERAS), 2).cpu()


def attach_routing(rows, encoded, temperature, method):
    rows = [dict(r) for r in rows]
    by_route = {route: [r for r in rows if r["route"] == route] for route in encoded}
    for route, group in by_route.items():
        cache, writes, probes = encoded[route]
        # Orientation labels are held out of the selection. Each column corresponds to one stored frame.
        pairs = cache["pairs"]
        keys = [writes[p["write"]] for i in range(1, len(CAMERAS) + 1) for p in pairs if p["position"] == i]
        similarities = view_similarities(keys, [probes[r["probe"]] for r in group], method)
        singles = torch.tensor([[r["conditions"][c]["logits"] for c in SINGLES] for r in group], dtype=torch.float64)
        routed = route_logits(similarities, singles, temperature=temperature)
        for i, (r, pair) in enumerate(zip(group, pairs, strict=True)):
            label = r["candidates"].index(r["identity"])
            expected = CAMERAS[pair["position"] - 1]
            r["expected_write_camera"] = expected
            r["cosines"] = {f"{cam:02}": dict(by_candidate=similarities[i, j].tolist(),
                                               mean=similarities[i, j].mean().item()) for j, cam in enumerate(CAMERAS)}
            r["oracle"] = r["conditions"][SINGLES[pair["position"] - 1]]
            r["routing"] = {}
            for name, result in routed.items():
                selected = CAMERAS[result["picked"][i]]
                read = dict(picked_camera=selected, weights=dict(zip((f"{cam:02}" for cam in CAMERAS),
                                                                    result["weights"][i].tolist(), strict=True)))
                for mode in ("hard", "soft"):
                    logits = result[mode][i].tolist()
                    read[mode] = dict(score_pair(logits, [True, True], label), logits=logits)
                r["routing"][name] = read
    return rows


def summarize(rows):
    def count_correct(values):
        return sum(s["decision"] == "target" for s in values)

    result = {}
    for route in sorted({r["route"] for r in rows}):
        group = [r for r in rows if r["route"] == route]
        full = dict(gallery=[r["conditions"]["prefix_4"] for r in group], oracle=[r["oracle"] for r in group])
        full.update({name: [r["conditions"][name] for r in group] for name in SINGLES})
        full.update({method: [r["routing"]["full"][method] for r in group] for method in ("hard", "soft")})
        direction = {f"{cam:02}": [i for i, r in enumerate(group) if r["expected_write_camera"] == cam]
                     for cam in CAMERAS}
        hits = {cam: sum(f"{group[i]['routing']['full']['picked_camera']:02}" == cam for i in indices)
                for cam, indices in direction.items()}
        drops = {}
        for name in DROPS:
            methods = {}
            for method in ("hard", "soft"):
                delta = [r["routing"][name][method] for r in group]
                old = [s["decision"] == "target" for s in full[method]]
                new = [s["decision"] == "target" for s in delta]
                methods[method] = dict(correct=sum(new), helped=sum(b and not a for a, b in zip(old, new, strict=True)),
                    harmed=sum(a and not b for a, b in zip(old, new, strict=True)),
                    by_direction={cam: sum(new[i] for i in indices) for cam, indices in direction.items()})
            drops[name] = methods
        result[route] = dict(n=len(group), counts={cam: len(indices) for cam, indices in direction.items()},
            view_hits=hits, correct={name: count_correct(values) for name, values in full.items()},
            by_direction={name: {cam: count_correct(values[i] for i in indices) for cam, indices in direction.items()}
                          for name, values in full.items()}, drops=drops)
    return result


def report(summary, temperature):
    lines = ["# 实验八 · 冻结 DINOv2 的四视角路由", "",
             f"零训练。使用原第 100 轮 Writer 和 Reader。软路由温度取 {temperature:g}。",
             "两辆候选车采用同一机位选择。方向命中按同方向指定机位统计，已知方向仅用于该指标及 oracle 对照。",
             "hard 只读选中的 8 槽，soft 对四次单图判别分数逐项加权，gallery 一次读完整 32 槽。", "",
             "| 车辆组 | 方向命中 | 同方向单图 oracle | 最好固定单图 | gallery | hard | soft |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for route, stats in summary.items():
        c, n = stats["correct"], stats["n"]
        nums = (sum(stats["view_hits"].values()), c["oracle"], max(c[s] for s in SINGLES),
                c["gallery"], c["hard"], c["soft"])
        lines.append("| " + " | ".join([route, *(f"{v}/{n}" for v in nums)]) + " |")
    lines.extend(["", "## 分方向结果与抽图对照", "",
                  "| 车辆组 | 条件 | 正车头 | 左前侧 | 左后侧 | 正车尾 | 改对 / 改错 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for route, stats in summary.items():
        rows = [("方向命中", stats["view_hits"], "—")]
        rows.extend((m, stats["by_direction"][m], "—") for m in ("oracle", "gallery", *SINGLES, "hard", "soft"))
        rows.extend((f"{condition} {method}", data["by_direction"], f"{data['helped']} / {data['harmed']}")
                    for condition, variants in stats["drops"].items() for method, data in variants.items())
        for name, views, changed in rows:
            cols = [f"{v}/{stats['counts'][cam]}" for cam, v in views.items()]
            lines.append("| " + " | ".join([route, name, *cols, changed]) + " |")
    lines.extend(["", "抽图条件同时去掉两辆车的指定机位，再用同一规则重新路由。",
                  "改对和改错按各自的完整四图 hard 或 soft 逐题比较。", ""])
    return "\n".join(lines)


def pooling_report(summaries, questions):
    lines = ["# 实验八 · 前景池化与空间路由对照", "",
             "三种方法共享第 100 轮模型的逐图 Reader 分数。主结果使用 hard 选图。",
             "方向栏统计同方向机位命中。身份变化比较相同查询在该组和 global_mean 下的判定。", "",
             "| 车辆组 | 描述符 | 正车头 | 左前侧 | 左后侧 | 正车尾 | 总方向命中 | 身份 hard | 身份 soft | 身份改对 / 改错 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    baseline = {r["probe"]: r["routing"]["full"]["hard"]["decision"] == "target" for r in questions["global_mean"]}
    cosine_rows = []
    for route in summaries["global_mean"]:
        for method, summary in summaries.items():
            s = summary[route]
            group = [r for r in questions[method] if r["route"] == route]
            changed = [(baseline[r["probe"]], r["routing"]["full"]["hard"]["decision"] == "target") for r in group]
            helped = sum(b and not a for a, b in changed)
            harmed = sum(a and not b for a, b in changed)
            hit, n = s["view_hits"], s["n"]
            nums = [f"{hit[c]}/{s['counts'][c]}" for c in hit]
            nums.extend([f"{sum(hit.values())}/{n}", f"{s['correct']['hard']}/{n}", f"{s['correct']['soft']}/{n}"])
            lines.append("| " + " | ".join([route, f"[{method}]({method}/report.md)", *nums, f"{helped} / {harmed}"]) + " |")
            means = [mean(r["cosines"][f"{cam:02}"]["mean"] for r in group) for cam in CAMERAS]
            cosine_rows.append("| " + " | ".join([route, method, *(f"{v:.4f}" for v in means)]) + " |")
    lines.extend(["", "## 各写入机位对同组全部查询的平均相似度", "",
                  "| 车辆组 | 描述符 | 00 | 20 | 16 | 12 |", "| --- | --- | ---: | ---: | ---: | ---: |",
                  *cosine_rows, "", "各方法的抽图结果、逐题分数和候选相似度见对应的子目录。", ""])
    return "\n".join(lines)
