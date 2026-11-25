"""Fixed, held-out views for the experiment 8 inference-only comparison."""

from itertools import groupby

import torch
from PIL import ImageOps

from .evidence_query_prepare import encode_crop
from .memory_retention import angular_gap, open_crop, score_pair


HISTORIES = {"F": (0,), "R": (12,), "FR": (0, 12), "RF": (12, 0),
             "FF'": (0, 23), "RR'": (12, 13)}
# For each target the predeclared front/rear question pairs are (01,11) and (22,14).
PROBES = (1, 11, 22, 14)
WRITE_CAMERAS = {cam for history in HISTORIES.values() for cam in history}


def _camera_keys(route):
    return {(s["identity"], s["camera_index"]): key for key, s in route["samples"].items()}


def check_angles(route, near, far):
    keys, samples = _camera_keys(route), route["samples"]

    def gap(uid, a, b):
        return angular_gap(samples[keys[uid, a]]["azimuth"], samples[keys[uid, b]]["azimuth"])

    for uid in route["identities"]:
        write_hashes = {samples[keys[uid, i]]["rgb_sha256"] for i in WRITE_CAMERAS}
        probe_hashes = {samples[keys[uid, i]]["rgb_sha256"] for i in PROBES}
        if write_hashes & probe_hashes:
            raise ValueError(f"overlapping write and query photos: {uid}")
        for a, b, probes in ((0, 23, (1, 22)), (12, 13, (11, 14))):
            if gap(uid, a, b) > near or any(min(gap(uid, a, q), gap(uid, b, q)) > near for q in probes):
                raise ValueError(f"same-direction camera pair outside {near} degrees: {uid}")
        if min(gap(uid, f, r) for f in (0, 23, 1, 22) for r in (12, 13, 11, 14)) < far:
            raise ValueError(f"front/rear views closer than {far} degrees: {uid}")


def encode_inputs(runtime, root, route):
    keys = _camera_keys(route)
    writes = {keys[uid, c] for uid in route["identities"] for c in WRITE_CAMERAS}
    probes = {keys[uid, c] for uid in route["identities"] for c in PROBES}
    for step in route["steps"]:
        for w in step["writes"].values():
            writes.add(w["sample"])
            if step["step"] == 0:
                probes.add(w["sample"])
            if w["independent"]:
                probes.add(w["independent"])
    obs, original, flip = {}, {}, {}
    def encode(image, role):
        x, mask = encode_crop(runtime, image, role)
        return runtime.tensor(x), runtime.tensor(mask)
    for key in sorted(writes | probes):
        image = open_crop(root, route["samples"][key])
        if key in writes:
            obs[key] = encode(image, "observations")
        if key in probes:
            original[key] = encode(image, "queries")
            flip[key] = encode(ImageOps.mirror(image), "queries")
    return obs, original, flip


def gallery_state(frames):
    # Each frame is written independently with empty memory, preserving candidate order.
    return dict(slots=torch.cat([s["slots"] for s in frames], dim=2), seen=frames[-1]["seen"],
                count=sum(s["count"] for s in frames))


def _write(model, keys, observations, state=None):
    x, mask = (torch.stack(t)[None] for t in zip(*(observations[k] for k in keys), strict=True))
    return model.update(x, mask, state)


def _read(model, state, key, target, original, flip):
    inputs = [original[key], flip[key]]
    x, mask = (torch.stack(t) for t in zip(*inputs, strict=True))
    batch = {k: v.expand(2, *v.shape[1:]) for k, v in state.items()}
    logits = model.score(x, mask, batch)
    if not torch.isfinite(logits).all():
        raise ValueError(f"non-finite scores for {key}")
    return {name: score_pair(row.tolist(), [True, True], target) for name, row in
            zip(("original", "flip", "mean"), (logits[0], logits[1], logits.mean(0)), strict=True)}


@torch.inference_mode()
def evaluate_route(model, route, observations, original, flip):
    ids, samples = route["identities"], route["samples"]
    natural, symmetric = [], []
    def question(key, label, state):
        return dict(route=route["id"], identity=ids[label], probe=key, camera=samples[key]["camera_index"],
                    scores=_read(model, state, key, label, original, flip))
    state = None
    for step in route["steps"]:
        entries = [step["writes"][uid] for uid in ids]
        state = _write(model, [e["sample"] for e in entries], observations, state)
        for label, entry in enumerate(entries):
            if step["step"] == 0:
                natural.append(dict(**question(entry["sample"], label, state), step=0, kind="self"))
            if entry["independent"]:
                natural.append(dict(**question(entry["independent"], label, state),
                                    step=step["step"], kind="new_view"))
    keys = _camera_keys(route)
    frames = {cam: _write(model, [keys[uid, cam] for uid in ids], observations) for cam in WRITE_CAMERAS}
    for history, cams in HISTORIES.items():
        memory = frames[cams[0]]
        for cam in cams[1:]:
            memory = _write(model, [keys[uid, cam] for uid in ids], observations, memory)
        states = {"memory": memory, "last": frames[cams[-1]], "gallery": gallery_state([frames[c] for c in cams])}
        for label, uid in enumerate(ids):
            for cam in PROBES:
                key = keys[uid, cam]
                distances = [angular_gap(samples[key]["azimuth"], samples[keys[uid, c]]["azimuth"]) for c in cams]
                for method, state in states.items():
                    symmetric.append(dict(**question(key, label, state), history=history, method=method,
                                          write_cameras=cams, angle_gaps=distances,
                                          nearest_stored_angle=distances[-1] if method == "last" else min(distances)))
    return natural, symmetric


def _stats(rows):
    original = [r["scores"]["original"] for r in rows]
    result = dict(n=len(rows))
    for name in ("original", "flip", "mean"):
        other = [r["scores"][name] for r in rows]
        comparisons = list(zip(original, other, strict=True))
        result[name] = dict(
            correct=sum(s["decision"] == "target" for s in other),
            agree=sum(a["decision"] == b["decision"] for a, b in comparisons),
            helped=sum(a["decision"] != "target" and b["decision"] == "target" for a, b in comparisons),
            harmed=sum(a["decision"] == "target" and b["decision"] != "target" for a, b in comparisons),
            mean_margin=sum(s["margin"] for s in other) / len(rows),
            mean_abs_margin_change=sum(abs(a["margin"]-b["margin"]) for a, b in comparisons) / len(rows))
    return result


def natural_summary(rows):
    independent = [r for r in rows if r["kind"] == "new_view"]
    return dict(first_exact=_stats([r for r in rows if r["kind"] == "self"]),
                first_independent=_stats([r for r in independent if r["step"] == 0]),
                all_independent=_stats(independent))


def paired_contrasts(rows):
    """Original queries only. Every transition compares one target and the same query."""
    indexed = {(r["history"], r["method"], r["identity"], r["camera"]):
               r["scores"]["original"]["decision"] == "target" for r in rows}
    ids = sorted({r["identity"] for r in rows})
    contrasts = []
    def compare(history1, method1, history2, method2, questions, label):
        transitions = []
        for uid in ids:
            for question in questions:
                before = all(indexed[history1, method1, uid, c] for c in question)
                after = all(indexed[history2, method2, uid, c] for c in question)
                transitions.append((before, after))
        contrasts.append(dict(baseline=f"{history1}/{method1}", comparison=f"{history2}/{method2}",
            questions=label, n=len(transitions), helped=transitions.count((False, True)),
            harmed=transitions.count((True, False)), before=sum(a for a, _ in transitions),
            after=sum(b for _, b in transitions)))
    for method in ("memory", "last", "gallery"):
        compare("R", method, "FR", method, [(1,), (22,)], "front/retention")
        compare("F", method, "FR", method, [(11,), (14,)], "rear/new evidence")
        for baseline in ("FF'", "RR'", "RF"):
            compare(baseline, method, "FR", method, [(1, 11), (22, 14)], "both views")
    for history in ("FR", "RF"):
        for left, right in (("last", "memory"), ("last", "gallery"), ("memory", "gallery")):
            compare(history, left, history, right, [(c,) for c in PROBES], "all")
    return contrasts


def _view_stats(rows):
    result = _stats(rows)
    targets = {}
    for row in rows:
        targets.setdefault((row["route"], row["identity"]), {})[row["camera"]] = row["scores"]
    result["paired_views_n"] = 2 * len(targets)
    for name in ("original", "flip", "mean"):
        result[name]["both_views_correct"] = sum(
            views[f][name]["decision"] == views[r][name]["decision"] == "target"
            for views in targets.values() for f, r in ((1, 11), (22, 14)))
    for side, cameras in (("front", (1, 22)), ("rear", (11, 14))):
        result[side] = _stats([r for r in rows if r["camera"] in cameras])
    return result


def symmetric_summary(rows):
    result = {}
    for h in HISTORIES:
        result[h] = {}
        for method in ("memory", "last", "gallery"):
            group = [r for r in rows if r["history"] == h and r["method"] == method]
            by_route = {route: _view_stats(list(items)) for route, items in
                        groupby(sorted(group, key=lambda r: r["route"]), key=lambda r: r["route"])}
            result[h][method] = dict(overall=_view_stats(group), by_route=by_route)
    return result
