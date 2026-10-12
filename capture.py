"""Static controlled CARLA views, not a flight planner or a detector."""

import argparse
import csv
import json
import math
from pathlib import Path
import queue
import re
import sys
import time

import numpy as np
from PIL import Image

from dinoexp.core import camera_pose, crop_bounds, pose_matches, pose_values, project_box


FIELDS = [
    "sample_id",
    "identity",
    "role",
    "sequence",
    "view",
    "weather",
    "position",
    "blueprint",
    "status",
    "reason",
    "image",
    "crop",
    "feature_visibility",
    "frame",
    "azimuth",
    "elevation",
    "distance",
    "camera_pose",
    "target_pose",
    "crop_box",
]


def validate(config):
    ids = [v["identity"] for v in config["vehicles"]]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("need at least two unique vehicle identities; first is the target")
    for names in (
        ids,
        [v["name"] for v in config["views"]],
        [w["name"] for w in config["weather"]],
    ):
        if (
            not names
            or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names)
        ):
            raise ValueError("identity/view/weather names must be unique safe identifiers")
    indices = config["spawn_indices"]
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
    ):
        raise ValueError("spawn_indices must be distinct nonnegative integers")
    if config["width"] < 64 or config["height"] < 64 or not 1 < config["fov"] < 179:
        raise ValueError("invalid camera dimensions/FOV")
    if not 0 < config["fixed_delta"] <= 0.1 or config["timeout"] <= 0:
        raise ValueError("invalid timing")
    if config["settle_ticks"] < 1 or config["warmup_ticks"] < 1:
        raise ValueError("need positive settling and camera warmup ticks")
    for v in config["views"] + [config["reference"]]:
        camera_pose((0, 0, 0), 0, v["azimuth"], v["elevation"], config["distance"])
    return config


def as_transform(carla, pose):
    x, y, z, pitch, yaw, roll = pose
    return carla.Transform(
        carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
    )


def connect(carla, cfg):
    client = carla.Client(cfg["host"], cfg["port"])
    client.set_timeout(cfg["timeout"])
    versions = [client.get_client_version(), client.get_server_version()]
    if versions[0].split(".")[:2] != versions[1].split(".")[:2]:
        raise RuntimeError(f"CARLA client/server mismatch: {versions}")
    return client, client.get_world(), versions


def inventory(world, versions):
    return {
        "versions": versions,
        "map": world.get_map().name,
        "existing_vehicles": len(world.get_actors().filter("vehicle.*")),
        "blueprints": [b.id for b in world.get_blueprint_library().filter("vehicle.*")],
        "spawn_points": [
            {"index": i, "pose": pose_values(t)}
            for i, t in enumerate(world.get_map().get_spawn_points())
        ],
    }


def next_image(world, sensor_queue, desired, cfg):
    # GPU images can arrive several ticks late. Validate both frame and image pose.
    barrier = None
    for _ in range(cfg["warmup_ticks"]):
        barrier = world.tick(cfg["timeout"])
    deadline = time.monotonic() + cfg["timeout"]
    while time.monotonic() < deadline:
        world.tick(min(cfg["timeout"], max(0.1, deadline - time.monotonic())))
        try:
            image = sensor_queue.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if image.frame >= barrier and pose_matches(pose_values(image.transform), desired):
            return image
    raise TimeoutError("no RGB frame matching the requested camera pose")


def capture(carla, world, versions, cfg, output, take_tick_ownership):
    initial_settings = world.get_settings()
    if initial_settings.synchronous_mode and not take_tick_ownership:
        raise RuntimeError(
            "world already synchronous: stop other tick clients, then use --take-tick-ownership"
        )
    if len(world.get_actors().filter("vehicle.*")):
        raise RuntimeError(
            "controlled probe needs a world without existing vehicles; "
            "use a separate empty scene (existing actors are never deleted)"
        )
    points = world.get_map().get_spawn_points()
    library = world.get_blueprint_library()
    for i in cfg["spawn_indices"]:
        if i >= len(points):
            raise ValueError(f"spawn index {i} not available; run --inventory")
    for vehicle in cfg["vehicles"]:
        bp = library.find(vehicle["blueprint"])
        if not bp.has_attribute("color"):
            raise ValueError(f"{bp.id} does not expose color; select another blueprint")
        if (
            bp.has_attribute("number_of_wheels")
            and bp.get_attribute("number_of_wheels").as_int() != 4
        ):
            raise ValueError(f"{bp.id} is not a four-wheel candidate")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "crops").mkdir()
    run = {
        "config": cfg,
        "versions": versions,
        "map": world.get_map().name,
        "status": "incomplete",
        "targets": [],
        "cleanup_errors": [],
        "protocol": "one candidate at a time at matched spawn points; blueprint identity",
        "reference_id": "reference",
        "target_identity": cfg["vehicles"][0]["identity"],
    }
    annotations = {}
    original_weather = world.get_weather()
    owned = []
    sensor = None
    rows = []

    def save_run():
        (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    def destroy(actor):
        try:
            if actor.type_id.startswith("sensor."):
                actor.stop()
            if not actor.destroy():
                raise RuntimeError("destroy returned false")
            owned.remove(actor)
        except Exception as error:
            run["cleanup_errors"].append(f"actor {actor.id}: {error}")

    save_run()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg["fixed_delta"]
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = max(10, math.ceil(cfg["fixed_delta"] / 0.01))
        world.apply_settings(settings)
        sensor_bp = library.find("sensor.camera.rgb")
        attributes = {
            "image_size_x": cfg["width"],
            "image_size_y": cfg["height"],
            "fov": cfg["fov"],
            "sensor_tick": 0,
            "motion_blur_intensity": 0,
            "exposure_mode": "manual",
            "exposure_compensation": 0,
            "shutter_speed": 200,
            "iso": 100,
            "fstop": 2.8,
        }
        for name, value in attributes.items():
            if sensor_bp.has_attribute(name):
                sensor_bp.set_attribute(name, str(value))
        frames = queue.Queue(maxsize=64)

        def receive(image):
            try:
                frames.put_nowait(image)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass

        sensor = world.spawn_actor(sensor_bp, carla.Transform(carla.Location(z=100)))
        owned.append(sensor)
        sensor.listen(receive)
        run["camera_attributes"] = dict(sensor.attributes)
        with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for position in cfg["spawn_indices"]:
                for w_index, weather in enumerate(cfg["weather"]):
                    current_weather = world.get_weather()
                    for name, value in weather.items():
                        if name != "name":
                            if not hasattr(current_weather, name):
                                raise ValueError(f"unknown weather attribute: {name}")
                            setattr(current_weather, name, value)
                    world.set_weather(current_weather)
                    for v_index, vehicle in enumerate(cfg["vehicles"]):
                        bp = library.find(vehicle["blueprint"])
                        bp.set_attribute("color", cfg["color"])
                        if bp.has_attribute("role_name"):
                            bp.set_attribute("role_name", "dino_probe")
                        pose = list(pose_values(points[position]))
                        pose[2] += 0.3
                        target = world.try_spawn_actor(bp, as_transform(carla, pose))
                        if target is None:
                            raise RuntimeError(f"cannot spawn {bp.id} at index {position}")
                        owned.append(target)
                        try:
                            target.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                            previous = target.get_transform()
                            for _ in range(cfg["settle_ticks"]):
                                previous = target.get_transform()
                                world.tick(cfg["timeout"])
                            tf = target.get_transform()
                            velocity = target.get_velocity()
                            if math.sqrt(
                                velocity.x**2 + velocity.y**2 + velocity.z**2
                            ) > 0.1 or not pose_matches(
                                pose_values(previous), pose_values(tf), 0.03, 0.5
                            ):
                                raise RuntimeError("vehicle did not settle; choose another spawn")
                            target.set_simulate_physics(False)
                            vertices = [
                                [p.x, p.y, p.z] for p in target.bounding_box.get_world_vertices(tf)
                            ]
                            center = np.mean(vertices, axis=0)
                            run["targets"].append(
                                {
                                    "identity": vehicle["identity"],
                                    "position": position,
                                    "weather": weather["name"],
                                    "attributes": dict(target.attributes),
                                    "actor_id": target.id,
                                    "pose": pose_values(tf),
                                }
                            )
                            views = list(cfg["views"])
                            if (
                                v_index == 0
                                and w_index == 0
                                and position == cfg["spawn_indices"][0]
                            ):
                                views = [dict(cfg["reference"], name="reference")] + views
                            for view in views:
                                ref = view["name"] == "reference"
                                seq = (
                                    -1
                                    if ref
                                    else [v["name"] for v in cfg["views"]].index(view["name"])
                                )
                                sid = (
                                    "reference"
                                    if ref
                                    else f"{vehicle['identity']}_p{position}_{weather['name']}_{seq:02d}_{view['name']}"
                                )
                                row = dict(
                                    sample_id=sid,
                                    identity=vehicle["identity"],
                                    role="reference" if ref else "query",
                                    sequence=seq,
                                    view=view["name"],
                                    weather=weather["name"],
                                    position=position,
                                    blueprint=bp.id,
                                    feature_visibility="unknown",
                                    azimuth=view["azimuth"],
                                    elevation=view["elevation"],
                                    distance=cfg["distance"],
                                    status="captured",
                                )
                                desired = camera_pose(
                                    center,
                                    tf.rotation.yaw,
                                    view["azimuth"],
                                    view["elevation"],
                                    cfg["distance"],
                                )
                                sensor.set_transform(as_transform(carla, desired))
                                image = next_image(world, frames, desired, cfg)
                                if not pose_matches(
                                    pose_values(target.get_transform()), pose_values(tf)
                                ):
                                    raise RuntimeError("target moved while capturing")
                                row.update(
                                    frame=image.frame,
                                    camera_pose=json.dumps(pose_values(image.transform)),
                                    target_pose=json.dumps(pose_values(tf)),
                                    image=f"images/{sid}.png",
                                )
                                bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                                    image.height, image.width, 4
                                )
                                rgb = Image.fromarray(bgra[:, :, :3][:, :, ::-1].copy())
                                rgb.save(output / row["image"])
                                try:
                                    box = project_box(
                                        vertices,
                                        image.transform.get_inverse_matrix(),
                                        image.width,
                                        image.height,
                                        cfg["fov"],
                                    )
                                    bounds = crop_bounds(box, image.width, image.height)
                                    row.update(crop=f"crops/{sid}.png", crop_box=json.dumps(bounds))
                                    rgb.crop(bounds).save(output / row["crop"])
                                    annotations[sid] = {"part": "", "visible": None, "box": None}
                                except ValueError as error:
                                    row.update(status="rejected", reason=str(error))
                                writer.writerow(row)
                                handle.flush()
                                rows.append(row)
                                print(f"{sid}: {row['status']}", flush=True)
                                if ref and row["status"] != "captured":
                                    raise RuntimeError(
                                        "reference crop invalid; increase distance or adjust FOV"
                                    )
                        finally:
                            destroy(target)
                            if target in owned:
                                raise RuntimeError(
                                    "candidate cleanup failed; aborting to avoid overlapping vehicles"
                                )
        if not any(r["role"] == "query" and r["status"] == "captured" for r in rows):
            raise RuntimeError("no valid query views")
        run["status"] = "complete"
    except BaseException as error:
        run["error"] = str(error)
        raise
    finally:
        for actor in list(reversed(owned)):
            destroy(actor)
        for label, restore in [
            ("weather", lambda: world.set_weather(original_weather)),
            ("settings", lambda: world.apply_settings(initial_settings)),
        ]:
            try:
                restore()
            except Exception as error:
                run["cleanup_errors"].append(f"{label}: {error}")
        run["captured"] = sum(r["status"] == "captured" for r in rows)
        run["rejected"] = sum(r["status"] == "rejected" for r in rows)
        (output / "roi_template.json").write_text(
            json.dumps(annotations, indent=2), encoding="utf-8"
        )
        save_run()
    if run["cleanup_errors"]:
        raise RuntimeError(f"capture finished but cleanup needs attention: {run['cleanup_errors']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("data/probe01"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--inventory", action="store_true", help="read-only world and asset inventory"
    )
    parser.add_argument("--dry-run", action="store_true", help="validate plan without CARLA")
    parser.add_argument("--take-tick-ownership", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = args.port
    validate(cfg)
    if args.dry_run:
        count = (
            len(cfg["vehicles"])
            * len(cfg["views"])
            * len(cfg["weather"])
            * len(cfg["spawn_indices"])
        )
        print(json.dumps({"query_count": count, "reference_count": 1, "config": cfg}, indent=2))
        return
    import carla

    _, world, versions = connect(carla, cfg)
    if args.inventory:
        print(json.dumps(inventory(world, versions), indent=2))
    else:
        capture(carla, world, versions, cfg, args.output, args.take_tick_ownership)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
