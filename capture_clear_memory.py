"""Collect or verify the unobstructed CARLA dataset for experiment 8."""

import argparse
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory

import numpy as np

from capture import as_transform, connect, find_blueprint
from dinoexp.clear_vehicle_views import (FAR, MIN_PIXELS, NEAR, SCHEMA, TRAIN_CASES, VAL_CASES,
                                        audit_dataset, crop_and_roi, depth_mm, make_route, orbit_views,
                                        require_clear, sample_key, verify_occupied_depth)
from dinoexp.core import camera_pose, camera_pose_matches, pose_matches, pose_values, project_box
from dinoexp.dense_capture import _receive, _rgb, managed_capture, next_pair
from dinoexp.memory_replay import audit_plan
from dinoexp.multiscene_capture import capture_lock
from dinoexp.multiscene_plan import choose_backgrounds
from dinoexp.region_ab_data import read_json, sha, write_json
from dinoexp.search_capture import (calibrate, frame_metadata, get_pair, place_camera, preflight,
                                    save_mask, save_png, spawn_sensors)
from dinoexp.search_episodes import instance_labels


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("data/experiment8_clear_orbits_v1")
STAGE_HEIGHT_M = 220


def add_depth_camera(carla, world, owned, cfg):
    bp = find_blueprint(world.get_blueprint_library(), "sensor.camera.depth")
    for k, v in dict(image_size_x=cfg["width"], image_size_y=cfg["height"], fov=cfg["fov"],
                     sensor_tick=0, **cfg["lens_attributes"]).items():
        bp.set_attribute(k, str(v))
    sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=100)))
    owned.append(sensor)
    frames = queue.Queue(maxsize=8)
    sensor.listen(_receive(frames))
    return sensor, frames


def capture_car(carla, world, case, cfg, candidate, staging):
    uid = case["scene_id"] + ":" + candidate["identity"]
    directory = staging / candidate["identity"]
    directory.mkdir()
    anchor = case["anchor_pose"]
    stage = [anchor[0], anchor[1], anchor[2] + STAGE_HEIGHT_M, 0, anchor[4], 0]
    owned, cleanup, samples = [], {"cleanup_errors": []}, {}
    with managed_capture(world, owned, cleanup):
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, .05, False
        world.apply_settings(settings)
        weather = world.get_weather()
        for k, v in cfg["weather"].items():
            setattr(weather, k, v)
        world.set_weather(weather)
        library = world.get_blueprint_library()
        bp = find_blueprint(library, candidate["blueprint"])
        bp.set_attribute("color", candidate["color"])
        actor = world.spawn_actor(bp, as_transform(carla, stage))
        owned.append(actor)
        actor.set_simulate_physics(False)
        world.tick(cfg["timeout"])
        tf = actor.get_transform()
        vertices = np.array([[p.x, p.y, p.z] for p in actor.bounding_box.get_world_vertices(tf)])
        center = vertices.mean(axis=0)
        sensors, queues = spawn_sensors(carla, world, owned, cfg)

        def photo(pose, label):
            return get_pair(world, queues, pose, cfg, directory / "sensor_retries.jsonl", label)

        mapping = calibrate(world, carla, sensors, photo, center, stage,
                            {uid: (actor, tf, center, candidate)}, cfg)
        for view in orbit_views(case["index"]):
            stem = f"{view['step']:02d}_{view['role']}"
            pose = camera_pose(center, tf.rotation.yaw, view["azimuth"], view["elevation"], view["distance"])
            place_camera(carla, sensors, pose)
            rgb, raw, meta = photo(pose, stem)
            if not pose_matches(pose_values(actor.get_transform()), pose_values(tf)):
                raise ValueError(f"vehicle moved during capture: {uid}/{stem}")
            actual = meta["rgb"]["pose"]
            projection = project_box(vertices, as_transform(carla, actual).get_inverse_matrix(),
                                     cfg["width"], cfg["height"], cfg["fov"])
            masks, _ = instance_labels(raw, mapping, [])
            crop, roi = crop_and_roi(masks[uid], projection)
            image = save_png(staging, directory / (stem + ".png"), rgb)
            mask = save_mask(staging, directory / (stem + "_mask.png"), masks[uid])
            prefix = case["path"] + "/"
            mask["path"] = prefix + mask["path"]
            key = sample_key(uid, view["step"], view["role"])
            samples[key] = dict(**view, identity=uid, pose=actual, sensor_frame=meta["rgb"]["frame"],
                                vehicle_center=center.tolist(), vehicle_yaw=tf.rotation.yaw,
                                size=[cfg["width"], cfg["height"]], bbox=crop, mask=mask,
                                image=prefix + image["path"], rgb_sha256=image["sha256"],
                                projection=projection, empty_depth=dict(roi=roi),
                                far_corner_m=float(np.linalg.norm(vertices - actual[:3], axis=1).max()))

        # Repeat every pose after removing the only vehicle. An occluder would still
        # be visible to the depth camera, including inside a hole in the instance mask.
        depth_sensor, depth_queue = add_depth_camera(carla, world, owned, cfg)
        # First check that this depth sensor measures the still-present car at the
        # last RGB pose. A blank or saturated sensor could otherwise pass every view.
        place_camera(carla, (sensors[0], depth_sensor), actual)
        _, occupied = next_pair(world, (queues[0], depth_queue), actual, cfg)
        frame_metadata(occupied, cfg)
        target_mm = depth_mm(_rgb(occupied))[masks[uid]]
        distances = np.linalg.norm(vertices - actual[:3], axis=1)
        verify_occupied_depth(target_mm, distances)
        if not actor.destroy():
            raise RuntimeError(f"could not clear {uid} for the background depth pass")
        owned.remove(actor)
        world.tick(cfg["timeout"])
        for sample in samples.values():
            pose = sample["pose"]
            place_camera(carla, (sensors[0], depth_sensor), pose)
            _, depth_frame = next_pair(world, (queues[0], depth_queue), pose, cfg)
            meta = frame_metadata(depth_frame, cfg)
            if not camera_pose_matches(meta["pose"], pose, .01, .05):
                raise ValueError("empty depth camera does not align with the original RGB")
            roi = sample["empty_depth"]["roi"]
            x0, y0, x1, y1 = roi
            depth = depth_mm(_rgb(depth_frame)[y0:y1, x0:x1])
            require_clear(depth, roi, sample["far_corner_m"])
            name = f"{sample['step']:02d}_{sample['role']}_depth.png"
            rec = save_png(staging, directory / name, depth)
            sample["empty_depth"].update(path=case["path"] + "/" + rec["path"], sha256=rec["sha256"],
                                         frame=meta["frame"], pose=meta["pose"])
    if cleanup["cleanup_errors"]:
        raise RuntimeError(str(cleanup["cleanup_errors"]))
    return samples


def collect(root, host, port):
    # Refuse unrelated directories before connecting to or mutating the simulator.
    marker = root / "capture_settings.json"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError(f"expected a new clean dataset directory: {root}")
    import carla

    config = read_json(ROOT / "configs/multiscene_roundtrip_v1.json")
    config["seed"] = 20261124
    config["sampling"].update(backgrounds=TRAIN_CASES + VAL_CASES, train_backgrounds=TRAIN_CASES,
                              min_background_distance_m=40)
    cfg = dict(config["capture"], host=host, port=port, warmup_ticks=5)
    client, world, versions = connect(carla, cfg)
    preflight(world, config["map"])
    points = [pose_values(t) for t in world.get_map().get_spawn_points()]
    backgrounds = choose_backgrounds(config, points)
    cases = []
    for i, b in enumerate(backgrounds):
        scene = f"clear_{i:02d}"
        cases.append(dict(scene_id=scene, index=i, path=f"cases/{scene}",
                          **{k: b[k] for k in ("anchor_pose", "anchor_spawn_index", "split", "group")}))
    signature = dict(schema=SCHEMA, map=config["map"], versions=versions, cases=cases,
                     stage_height_m=STAGE_HEIGHT_M, settings=config["capture"], seed=config["seed"])
    root.mkdir(parents=True, exist_ok=True)
    with capture_lock(root):
        if marker.is_file() and read_json(marker) != signature:
            raise ValueError("incompatible settings in partially collected clean dataset")
        write_json(marker, signature)
        (root / "cases").mkdir(exist_ok=True)
        routes, manifest_cases = [], []
        for case in cases:
            final = root / case["path"]
            if not final.exists():
                with TemporaryDirectory(prefix=".collect-", dir=root) as tmp:
                    staging = Path(tmp) / "case"
                    staging.mkdir()
                    samples = {}
                    for candidate in cfg["candidates"]:
                        preflight(world, config["map"])
                        print(f"Capture {case['scene_id']} ({case['split']}) {candidate['blueprint']}", flush=True)
                        samples.update(capture_car(carla, world, case, cfg, candidate, staging))
                    write_json(staging / "route.json", make_route(case, samples))
                    staging.rename(final)
            routes.append(read_json(final / "route.json"))
            manifest_cases.append(dict(**case, route_sha256=sha(final / "route.json")))
        manifest = dict(signature, cases=manifest_cases, status="complete")
        write_json(root / "dataset_manifest.json", manifest)
        digest = sha(root / "dataset_manifest.json")
        plan = dict(kind="memory_retention_plan_v1", dataset_sha256=digest, routes=routes,
                    min_visible_pixels=MIN_PIXELS, max_query_angle=NEAR, novelty_angle=FAR,
                    split="all", excluded=[])
        _, _, _, plan["coverage"] = audit_plan(plan, digest)
        write_json(root / "plan.json", plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "audit"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    if args.stage == "capture":
        collect(args.data, args.host, args.port)
    result, sheet = audit_dataset(args.data)
    preview = args.data / "validation_foregrounds.jpg"
    sheet.save(preview)
    print(json.dumps(dict(**result, preview=str(preview)), indent=2))


if __name__ == "__main__":
    main()
