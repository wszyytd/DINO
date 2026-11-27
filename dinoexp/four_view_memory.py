"""Four prefixes of a fixed vehicle sequence, all scored against the same four unseen cameras."""

from collections import defaultdict
from statistics import mean

import torch

from .evidence_query_prepare import encode_crop
from .experiment8_long_eval import traced_write
from .memory_retention import angular_gap, open_crop, score_pair
from .symmetric_memory_eval import _camera_keys, _write, gallery_state


VIEWS = (("正车头", 0, 1), ("左前侧", 20, 21), ("左后侧", 16, 15), ("正车尾", 12, 11))
CONTROLS = ("updated", "written", "score_mean", "gallery")


def select_views(route, max_gap):
    keys, samples = _camera_keys(route), route["samples"]
    if route["split"] != "val" or len(route["identities"]) != 2 or len(keys) != len(samples):
        raise ValueError(f"expected two held-out candidates and unique camera IDs: {route['id']}")
    pairs = []
    for uid in route["identities"]:
        for position, (name, w, q) in enumerate(VIEWS, 1):
            if (uid, w) not in keys or (uid, q) not in keys:
                raise ValueError(f"missing four-view camera: {uid}, {w:02}/{q:02}")
            write, query = keys[uid, w], keys[uid, q]
            a, b = samples[write], samples[query]
            gap = angular_gap(a["azimuth"], b["azimuth"])
            if not (0 < gap <= max_gap):
                raise ValueError(f"invalid write/query angle: {write}, {query}, {gap}")
            if not a.get("mask") or not b.get("mask"):
                raise ValueError(f"missing vehicle mask: {write}, {query}")
            pairs.append(dict(route=route["id"], identity=uid, view=name, position=position, write_camera=w,
                              query_camera=q, write=write, query=query, write_azimuth=a["azimuth"],
                              query_azimuth=b["azimuth"], angle_gap=gap))
    for field in ("frame", "rgb_sha256"):
        written = {samples[p["write"]][field] for p in pairs}
        queried = {samples[p["query"]][field] for p in pairs}
        if written & queried:
            raise ValueError(f"four-view write/query overlap in {field}: {route['id']}")
    return pairs


def encode_sequence(runtime, root, route, pairs):
    def encode(key, role):
        crop = open_crop(root, route["samples"][key], foreground=True)
        features, mask = encode_crop(runtime, crop, role)
        return runtime.tensor(features), runtime.tensor(mask)
    return ({p["write"]: encode(p["write"], "observations") for p in pairs},
            {p["query"]: encode(p["query"], "queries") for p in pairs})


@torch.inference_mode()
def inspect_sequence(model, route, pairs, observations, queries):
    ids = route["identities"]
    x, valid = (torch.stack(items) for items in zip(*(queries[p["query"]] for p in pairs), strict=True))

    def read(state):
        batch = {k: v.expand(len(pairs), *v.shape[1:]) for k, v in state.items()}
        return model.score(x, valid, batch)

    rows, gates, singles, single_logits = [], [], [], []
    memory = None
    for step, (_, cam, _) in enumerate(VIEWS, 1):
        keys = [p["write"] for p in pairs if p["position"] == step]
        states = {}
        if memory is None:
            fresh = memory = _write(model, keys, observations)
        else:
            previous = memory
            memory, trace = traced_write(model, keys, observations, previous)
            # Each current candidate has one visible observation. The Writer uses the
            # learned initial queries every time, including when a previous state exists.
            fresh = dict(singles[0], slots=trace["written"][None])
            states.update(previous=previous, proposal=dict(memory, slots=trace["proposal"][None]))
            for i, uid in enumerate(ids):
                gates.append(dict(route=route["id"], identity=uid, step=step, camera=cam,
                                  gate=trace["gate"][i].tolist()))
        singles.append(fresh)
        single_logits.append(read(fresh))
        states.update(updated=memory, gallery=gallery_state(singles))
        logits = {name: read(state) for name, state in states.items()}
        logits.update(written=single_logits[-1], score_mean=torch.stack(single_logits).mean(0))
        stacked = torch.stack(list(logits.values())).cpu()
        if not torch.isfinite(stacked).all():
            raise ValueError(f"non-finite scores: {route['id']}, step {step}")
        for j, p in enumerate(pairs):
            scores = {}
            for i, method in enumerate(logits):
                values = stacked[i, j].tolist()
                scores[method] = dict(score_pair(values, [True, True], ids.index(p["identity"])), logits=values)
            rows.append(dict(route=route["id"], identity=p["identity"], candidates=ids,
                             step=step, history=[w for _, w, _ in VIEWS[:step]], probe=p["query"],
                             probe_position=p["position"], query_camera=p["query_camera"], scores=scores))
    return rows, gates


def summarize(rows):
    by_route = defaultdict(list)
    for r in rows:
        by_route[r["route"]].append(r)
    groups = {**by_route, "all": rows,
              "pair1+pair2": [r for r in rows if r["route"].startswith(("pair1_", "pair2_"))]}
    results = []
    for group, items in groups.items():
        if not items:
            continue
        index = {(r["step"], r["probe"]): r for r in items}
        for step in range(1, len(VIEWS)+1):
            subset = [r for r in items if r["step"] == step]
            for method in CONTROLS:
                views, right = {}, defaultdict(list)
                changes = {name: dict(n=0, helped=0, harmed=0) for name in ("old", "new")}
                for pos, (name, _, _) in enumerate(VIEWS, 1):
                    margins, correct = [], 0
                    for r in (q for q in subset if q["probe_position"] == pos):
                        score = r["scores"][method]
                        ok = score["decision"] == "target"
                        correct += ok
                        margins.append(score["margin"])
                        right[r["identity"]].append(ok)
                        if step > 1 and pos <= step:
                            prev = index[step-1, r["probe"]]["scores"][method]["decision"] == "target"
                            delta = changes["new" if pos == step else "old"]
                            delta["n"] += 1
                            delta["helped"] += ok and not prev
                            delta["harmed"] += prev and not ok
                    views[name] = dict(correct=correct, n=len(margins), mean_margin=mean(margins))
                results.append(dict(group=group, method=method, step=step, views=views,
                                    correct=sum(v["correct"] for v in views.values()), n=len(subset),
                                    all_right=sum(all(v) for v in right.values()), all_right_n=len(right),
                                    old_changes=changes["old"], new_changes=changes["new"]))
    return results
