"""Two simultaneous candidates and an explicit target reference, for scene inspection."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import queue
import sys
import traceback

import numpy as np
from PIL import Image

from capture import FIELDS, as_transform, connect, find_blueprint, next_image
from dinoexp.core import (
    crop_bounds,
    image_quality,
    pose_matches,
    pose_values,
    project_box,
)
from dinoexp.scene import candidate_poses, capture_view_pose, validate_scene, with_resolution
from dinoexp.environment import select_nearby_objects


def preview(root, rows, target):
    reference = next((r for r in rows if r["role"] == "reference"), None)
    parts = []
    if reference and reference.get("crop"):
        parts.append(
            f'<h2>目标参考：{html.escape(target)}</h2><img src="{reference["crop"]}" width="400">'
        )
    scenes = {}
    for row in rows:
        if row["role"] == "query":
            scenes.setdefault(row["view"], []).append(row)
    for name, candidates in scenes.items():
        parts.append(
            f'<h2>{html.escape(name)}</h2><img src="{candidates[0]["image"]}" width="1000"><div class="crops">'
        )
        for row in candidates:
            label = "目标" if row["identity"] == target else "干扰"
            parts.append(
                f"<section><b>{html.escape(row['identity'])}：{label}；{row['status']}</b>"
            )
            if row.get("crop"):
                parts.append(f'<img src="{row["crop"]}" width="320">')
            parts.append(f"<p>{html.escape(row.get('reason', ''))}</p></section>")
        parts.append("</div>")
    doc = '<!doctype html><meta charset="utf-8"><title>两候选场景预览</title><style>body{font:16px system-ui;margin:30px auto;max-width:1100px}img{max-width:100%}.crops{display:flex;gap:30px}section{flex:1}</style>'
    doc += "<h1>两候选同场预览</h1><p>A/B 身份仅作真值展示。先检查两车真实外观、初始相似度、遮挡与参考图；尚未进行匹配或排除。</p>"
    (root / "preview.html").write_text(doc + "".join(parts), encoding="utf-8")


def capture_scene(carla, world, versions, cfg, root, take_tick):
    initial = world.get_settings()
    if initial.synchronous_mode and not take_tick:
        raise RuntimeError("stop other tick clients and use --take-tick-ownership")
    if len(world.get_actors().filter("vehicle.*")):
        raise RuntimeError("scene must have no existing vehicles; they will not be deleted")
    points = world.get_map().get_spawn_points()
    index = cfg["anchor_spawn_index"]
    if not 0 <= index < len(points):
        raise ValueError("anchor_spawn_index is unavailable")
    poses = candidate_poses(pose_values(points[index]), cfg["separation"],
                            [c.get('yaw_offset', 0) for c in cfg['candidates']])
    library = world.get_blueprint_library()
    for candidate in cfg["candidates"]:
        bp = find_blueprint(library, candidate["blueprint"])
        if not bp.has_attribute("color"):
            raise ValueError(f"{bp.id} has no color attribute")
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    (root / "crops").mkdir()
    original_weather = world.get_weather()
    owned, targets, rows = [], [], []
    hidden_ids = set()
    run = dict(
        config=cfg,
        versions=versions,
        map=world.get_map().name,
        status="incomplete",
        target_identity=cfg["target_identity"],
        reference_id="reference",
        protocol="two simultaneous candidates, shared RGB frames",
        targets=[],
        cleanup_errors=[],
    )
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg["fixed_delta"]
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = max(10, math.ceil(cfg["fixed_delta"] / 0.01))
        world.apply_settings(settings)
        weather = world.get_weather()
        for key, value in cfg["weather"].items():
            setattr(weather, key, value)
        world.set_weather(weather)
        hide = cfg.get("hide_environment_objects")
        if hide:
            center = tuple(sum(p[i] for p in poses) / len(poses) for i in range(3))
            records = {}
            for label in hide["labels"]:
                tag = getattr(carla.CityObjectLabel, label, None)
                if tag is None:
                    raise RuntimeError(f"environment label unavailable: {label}")
                nearby = select_nearby_objects(
                    world.get_environment_objects(tag), center, hide["radius"]
                )
                for record in nearby:
                    record["label"] = label
                    records[record["id"]] = record
            run["hidden_environment_objects"] = list(records.values())
            if not records:
                raise RuntimeError(
                    "no nearby pole/light objects found; obstruction removal is not confirmed"
                )
            hidden_ids = set(records)
            print(
                f"Hiding {len(hidden_ids)} nearby pole/light objects; IDs recorded in run.json",
                flush=True,
            )
            world.enable_environment_objects(hidden_ids, False)
        # Spawn BOTH before any photography; leave both alive until the run ends.
        for candidate, pose in zip(cfg["candidates"], poses, strict=True):
            bp = find_blueprint(library, candidate["blueprint"])
            bp.set_attribute("color", candidate["color"])
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "dino_scene_" + candidate["identity"])
            actor = world.try_spawn_actor(bp, as_transform(carla, pose))
            if actor is None:
                raise RuntimeError(
                    f"cannot spawn {candidate['identity']} at {pose}; choose a clear anchor"
                )
            owned.append(actor)
            targets.append((candidate, actor))
            actor.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
        previous = [a.get_transform() for _, a in targets]
        for _ in range(cfg["settle_ticks"]):
            previous = [a.get_transform() for _, a in targets]
            world.tick(cfg["timeout"])
        geometry = []
        for (candidate, actor), old in zip(targets, previous, strict=True):
            tf = actor.get_transform()
            speed = actor.get_velocity()
            if (
                not pose_matches(pose_values(tf), pose_values(old))
                or math.sqrt(speed.x**2 + speed.y**2 + speed.z**2) > 0.1
            ):
                raise RuntimeError("candidate not stationary; change anchor or increase settling")
            actor.set_simulate_physics(False)
            vertices = [[p.x, p.y, p.z] for p in actor.bounding_box.get_world_vertices(tf)]
            geometry.append((candidate, actor, tf, vertices))
            run["targets"].append(
                dict(
                    identity=candidate["identity"],
                    actor_id=actor.id,
                    pose=pose_values(tf),
                    attributes=dict(actor.attributes),
                )
            )
        group_center = np.mean([np.mean(g[3], axis=0) for g in geometry], axis=0)
        target = next(g for g in geometry if g[0]["identity"] == cfg["target_identity"])
        sensor_bp = find_blueprint(library, "sensor.camera.rgb")
        attrs = dict(
            image_size_x=cfg["width"], image_size_y=cfg["height"], fov=cfg["fov"], sensor_tick=0
        )
        if set(attrs) & set(cfg["camera_attributes"]):
            raise ValueError("camera_attributes cannot override dimensions/FOV/timing")
        attrs.update(cfg["camera_attributes"])
        for key, value in attrs.items():
            if not sensor_bp.has_attribute(key):
                raise ValueError(f"unsupported camera attribute: {key}")
            sensor_bp.set_attribute(key, str(value))
        sensor = world.spawn_actor(sensor_bp, carla.Transform(carla.Location(z=100)))
        owned.append(sensor)
        frames = queue.Queue(maxsize=32)

        def receive(image):
            try:
                frames.put_nowait(image)
            except queue.Full:
                try:
                    frames.get_nowait()
                    frames.put_nowait(image)
                except (queue.Empty, queue.Full):
                    pass

        sensor.listen(receive)
        run["camera_attributes"] = dict(sensor.attributes)
        views = [dict(cfg["reference"], name="reference")] + cfg["views"]
        with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for sequence, view in enumerate(views, start=-1):
                ref = view["name"] == "reference"
                center = np.mean(target[3], axis=0) if ref else group_center
                desired = capture_view_pose(center, points[index].rotation.yaw,
                    target[2].rotation.yaw, view, is_reference=ref,
                    query_yaw_mode=cfg.get('query_yaw_mode', 'target'))
                sensor.set_transform(as_transform(carla, desired))
                image = next_image(world, frames, desired, cfg)
                if any(
                    not pose_matches(pose_values(actor.get_transform()), pose_values(tf))
                    for _, actor, tf, _ in geometry
                ):
                    raise RuntimeError("candidate moved during capture")
                raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    image.height, image.width, 4
                )
                rgb = Image.fromarray(raw[:, :, :3][:, :, ::-1].copy())
                image_path = f"images/{view['name']}.png"
                rgb.save(root / image_path)
                quality = image_quality(rgb)
                for candidate, _, tf, vertices in [target] if ref else geometry:
                    sid = (
                        "reference"
                        if ref
                        else f"{candidate['identity']}_{sequence:02d}_{view['name']}"
                    )
                    row = dict(
                        sample_id=sid,
                        identity=candidate["identity"],
                        role="reference" if ref else "query",
                        sequence=sequence,
                        view=view["name"],
                        weather=cfg.get('weather_name', 'noon'),
                        position=index,
                        blueprint=candidate["blueprint"],
                        status="captured",
                        feature_visibility="unknown",
                        frame=image.frame,
                        image=image_path,
                        camera_pose=json.dumps(pose_values(image.transform)),
                        target_pose=json.dumps(pose_values(tf)),
                        azimuth=view["azimuth"],
                        elevation=view["elevation"],
                        distance=view["distance"],
                        rgb_mean=quality["rgb_mean"],
                        near_black_fraction=quality["near_black_fraction"],
                    )
                    try:
                        if quality["near_black"]:
                            raise ValueError("near_black_image")
                        bounds = crop_bounds(
                            project_box(
                                vertices,
                                image.transform.get_inverse_matrix(),
                                image.width,
                                image.height,
                                cfg["fov"],
                            ),
                            image.width,
                            image.height,
                        )
                        crop = rgb.crop(bounds)
                        row.update(crop=f"crops/{sid}.png", crop_box=json.dumps(bounds))
                        crop.save(root / row["crop"])
                        if image_quality(crop)["near_black"]:
                            raise ValueError("near_black_crop")
                    except ValueError as error:
                        row.update(status="rejected", reason=str(error))
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    print(f"{sid}: {row['status']}", flush=True)
                    if row["status"] != "captured":
                        raise RuntimeError(
                            f"preview invalid: {sid}: {row['reason']}; inspect saved images"
                        )
        run["status"] = "complete"
    except BaseException as error:
        run["error"] = str(error)
        raise
    finally:
        for actor in reversed(owned):
            try:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                if not actor.destroy():
                    raise RuntimeError("destroy returned false")
            except Exception as error:
                run["cleanup_errors"].append(f"actor {actor.id}: {error}")
        restore_actions = [
            ("weather", lambda: world.set_weather(original_weather)),
            ("settings", lambda: world.apply_settings(initial)),
        ]
        if hidden_ids:
            restore_actions.insert(
                0,
                ("environment_objects", lambda: world.enable_environment_objects(hidden_ids, True)),
            )
        for name, restore in restore_actions:
            try:
                restore()
            except Exception as error:
                run["cleanup_errors"].append(f"{name}: {error}")
        (root / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        preview(root, rows, cfg["target_identity"])
    if run["cleanup_errors"]:
        raise RuntimeError(f"cleanup errors: {run['cleanup_errors']}")
    print(f"Scene preview saved to {root / 'preview.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/two_candidates.json"))
    parser.add_argument("--output", type=Path, default=Path("data/scene01"))
    parser.add_argument("--take-tick-ownership", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--width", type=int, help="native RGB sensor width; requires --height")
    parser.add_argument("--height", type=int, help="native RGB sensor height; keeps aspect ratio")
    args = parser.parse_args()
    cfg = validate_scene(with_resolution(
        json.loads(args.config.read_text(encoding="utf-8-sig")), args.width, args.height
    ))
    if args.dry_run:
        print(
            json.dumps(
                dict(candidates=2, reference=1, shared_views=len(cfg["views"]), config=cfg),
                indent=2,
            )
        )
        return
    import carla

    _, world, versions = connect(carla, cfg)
    capture_scene(carla, world, versions, cfg, args.output, args.take_tick_ownership)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
