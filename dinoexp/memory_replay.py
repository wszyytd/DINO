"""Causal query schedules and split checks for all-prefix vehicle replay."""

from collections import Counter


def partition_routes(routes):
    train = [route for route in routes if route["split"] == "train"]
    val = [route for route in routes if route["split"] == "val"]
    if not train or not val or len(train) + len(val) != len(routes):
        raise ValueError("replay training requires nonempty train and val routes")
    if len({route["id"] for route in routes}) != len(routes):
        raise ValueError("route IDs must be unique")
    if {route["group"] for route in train} & {route["group"] for route in val}:
        raise ValueError("background group crosses train/val")
    train_hashes = {sample["rgb_sha256"] for route in train for sample in route["samples"].values()}
    val_hashes = {sample["rgb_sha256"] for route in val for sample in route["samples"].values()}
    if train_hashes & val_hashes:
        raise ValueError("captured RGB crosses train/val")
    return train, val


def build_schedule(route):
    """Replay every accepted observation; independent queries read the pre-write state."""
    ids = route["identities"]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("expected two distinct candidates")
    samples = route["samples"]
    all_writes = {write["sample"] for step in route["steps"] for write in step["writes"].values()}
    write_rgb = {samples[key]["rgb_sha256"] for key in all_writes}
    seen, previous, sample_keys, replay, schedule = set(), set(), set(), [], []
    for index, step in enumerate(route["steps"]):
        if step["step"] != index:
            raise ValueError("steps must be consecutive and chronological")
        if not step["writes"]:
            continue
        writes, prewrite = [], []
        for uid, observation in step["writes"].items():
            label = ids.index(uid)
            key = observation["sample"]
            sample = samples[key]
            if sample["identity"] != uid or key in sample_keys:
                raise ValueError("invalid identity or duplicate accepted observation")
            parent_key = (uid, sample["rgb_sha256"])
            if parent_key in previous:
                raise ValueError("an accepted RGB is written twice for the same candidate")
            sample_keys.add(key)
            previous.add(parent_key)
            independent = observation["independent"]
            if independent is not None:
                probe = samples[independent]
                if probe["identity"] != uid or probe["rgb_sha256"] in write_rgb:
                    raise ValueError("independent query overlaps route writes or has the wrong identity")
                if label in seen:
                    prewrite.append((independent, label))
            writes.append((key, label))
        writes.sort(key=lambda pair: pair[1])
        seen.update(label for _, label in writes)
        replay.extend(writes)
        schedule.append(dict(step=index, writes=writes, replay=list(replay), prewrite=prewrite,
                             fixed_labels=[label if label in seen else len(ids)
                                           for label in range(len(ids))]))
    if not schedule or len(seen) != len(ids):
        raise ValueError("both candidates need at least one accepted observation")
    return schedule


def count_queries(schedules):
    count = Counter(routes=len(schedules))
    for schedule in schedules:
        count["write_steps"] += len(schedule)
        for step in schedule:
            for group in ("writes", "replay", "prewrite", "fixed_labels"):
                count[group] += len(step[group])
    return dict(count)


def audit_plan(plan, dataset_hash):
    if plan["kind"] != "memory_retention_plan_v1" or plan["dataset_sha256"] != dataset_hash:
        raise ValueError("capture manifest differs from replay plan")
    train, val = partition_routes(plan["routes"])
    schedules = {r["id"]: build_schedule(r) for r in plan["routes"]}
    coverage = {name: count_queries([schedules[r["id"]] for r in routes])
                for name, routes in (("train", train), ("val", val))}
    if not all(group["prewrite"] for group in coverage.values()):
        raise ValueError("need independent pre-write queries in each split")
    return train, val, schedules, coverage
