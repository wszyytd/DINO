"""Read the actual second update and the single-view controls on matched, unflipped queries."""

from contextlib import ExitStack
from statistics import mean

import torch

from .memory_retention import score_pair
from .symmetric_memory_eval import HISTORIES, PROBES, _camera_keys, _write, gallery_state


METHODS = ("previous", "written", "proposal", "updated", "score_mean", "gallery")


def _second_write(model, keys, observations, previous):
    captured = {}

    def capture_updater(_module, inputs, output):
        captured["written"] = inputs[1]
        captured["proposal"] = output[0]

    def capture_gate(_module, _inputs, output):
        captured["gate"] = output.sigmoid().squeeze(-1)

    # Observe one production write, without changing the pinned model or its training path.
    with ExitStack() as hooks:
        hooks.enter_context(model.updater.register_forward_hook(capture_updater))
        hooks.enter_context(model.gate.register_forward_hook(capture_gate))
        updated = _write(model, keys, observations, previous)
    return updated, captured


@torch.inference_mode()
def inspect_route(model, route, features):
    ids = route["identities"]
    keys = _camera_keys(route)
    observations, queries = features
    probes = [(label, cam, keys[uid, cam]) for label, uid in enumerate(ids) for cam in PROBES]
    x, valid = (torch.stack(items) for items in zip(*(queries[k] for _, _, k in probes), strict=True))
    rows, gates = [], []
    for history in ("FR", "RF"):
        first, second = HISTORIES[history]
        previous = _write(model, [keys[uid, first] for uid in ids], observations)
        updated, captured = _second_write(model, [keys[uid, second] for uid in ids], observations, previous)
        p = previous["slots"][0]
        w, u, g = (captured[k] for k in ("written", "proposal", "gate"))
        if w.shape != p.shape or u.shape != p.shape or g.shape != p.shape[:2]:
            raise ValueError("expected one visible second write per candidate and a gate per slot")
        for i, uid in enumerate(ids):
            gates.append(dict(route=route["id"], history=history, identity=uid, gate=g[i].tolist(),
                              proposal_delta=(u[i]-p[i]).norm(dim=-1).tolist(),
                              updated_delta=(updated["slots"][0, i]-p[i]).norm(dim=-1).tolist()))

        written = dict(previous, slots=w[None])
        states = dict(previous=previous, written=written, proposal=dict(updated, slots=u[None]),
                      updated=updated, gallery=gallery_state([previous, written]))
        logits = {}
        for name, state in states.items():
            batch = {k: v.expand(len(probes), *v.shape[1:]) for k, v in state.items()}
            logits[name] = model.score(x, valid, batch)
        logits["score_mean"] = (logits["previous"] + logits["written"]) / 2
        stacked = torch.stack([logits[name] for name in METHODS]).cpu()
        if not torch.isfinite(stacked).all():
            raise ValueError(f"non-finite diagnostic logits: {route['id']}/{history}")
        probabilities = stacked.softmax(dim=-1)
        for j, (label, cam, key) in enumerate(probes):
            scores = {}
            for i, name in enumerate(METHODS):
                values = stacked[i, j].tolist()  # stable route identity order, then Null
                scores[name] = dict(score_pair(values, [True, True], label), logits=values,
                                    target_probability=probabilities[i, j, label].item())
            rows.append(dict(route=route["id"], history=history, write_cameras=[first, second],
                             candidates=ids, identity=ids[label], camera=cam, probe=key, scores=scores))
    return rows, gates


def compare_replay(reference, current):
    def index(rows):
        selected = [r for r in rows if r["kind"] == "new_view"]
        result = {(r["route"], r["probe"]): r["after"] for r in selected}
        if len(result) != len(selected) or not result:
            raise ValueError("missing or duplicate independent queries")
        return result

    old, new = index(reference), index(current)
    if old.keys() != new.keys():
        raise ValueError("independent query IDs differ from the original run")
    changed = [k for k in old if old[k]["decision"] != new[k]["decision"]]
    error = max(abs(old[k][field] - new[k][field]) for k in old for field in ("target", "other", "null"))
    return dict(n=len(old), reference_correct=sum(v["decision"] == "target" for v in old.values()),
                current_correct=sum(v["decision"] == "target" for v in new.values()),
                agree=len(old)-len(changed), changed=changed, max_abs_logit_diff=error)


def summarize_views(rows):
    def side_stats(part, method):
        scores = [r["scores"][method] for r in part]
        margins = [s["margin"] for s in scores]
        return dict(correct=sum(s["decision"] == "target" for s in scores), n=len(scores),
                    margin_mean=mean(margins), margin_min=min(margins), margin_max=max(margins),
                    mean_a_minus_b=mean(s["logits"][0]-s["logits"][1] for s in scores))

    groups = {name: [r for r in rows if r["route"] == name] for name in sorted({r["route"] for r in rows})}
    groups["all"] = rows
    groups["pair1+pair2"] = [r for r in rows if r["route"].startswith(("pair1_", "pair2_"))]
    summary = []
    for group, items in groups.items():
        if not items:
            continue
        for history in ("FR", "RF"):
            subset = [r for r in items if r["history"] == history]
            for method in METHODS:
                cases = {(r["identity"], r["camera"]): r["scores"][method]["decision"] for r in subset}
                identities = {r["identity"] for r in subset}
                both = sum(cases[uid, f] == cases[uid, tail] == "target"
                           for uid in identities for f, tail in ((1, 11), (22, 14)))
                summary.append(dict(group=group, history=history, method=method, both_correct=both,
                                    paired_n=2*len(identities),
                                    front=side_stats([r for r in subset if r["camera"] in (1, 22)], method),
                                    rear=side_stats([r for r in subset if r["camera"] in (11, 14)], method)))
    return summary
