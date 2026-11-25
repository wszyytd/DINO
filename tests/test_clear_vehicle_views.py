import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import capture_clear_memory as capture
from dinoexp.clear_vehicle_views import (TRAIN_CASES, VAL_CASES, audit_dataset, crop_and_roi,
                                        depth_mm, orbit_views, require_clear, sample_key, verify_occupied_depth)
from dinoexp.core import camera_pose
from dinoexp.memory_retention import angular_gap
from dinoexp.region_ab_data import read_json, sha, write_json


def test_small_occluder_in_hole_and_frame_clipping():
    mask = np.zeros((150, 200), dtype=bool)
    mask[20:125, 30:170] = True
    mask[65:70, 50:150] = False
    projection = [25., 18., 178., 133.]
    _, roi = crop_and_roi(mask, projection)
    x0, y0, x1, y1 = roi
    depth = np.full((y1 - y0, x1 - x0), 20000, dtype=np.uint16)
    require_clear(depth, roi, 11.)
    depth[67 - y0, 51 - x0] = 8000
    with pytest.raises(ValueError, match="obstructed"):
        require_clear(depth, roi, 11.)
    with pytest.raises(ValueError, match="clipped"):
        crop_and_roi(mask, [-1, 9, 178, 133])
    with pytest.raises(ValueError, match="too small"):
        crop_and_roi(np.zeros_like(mask), projection)
    encoded = np.array([[[0, 0, 0], [128, 10, 2], [255, 255, 255]]], dtype=np.uint8)
    assert depth_mm(encoded).tolist() == [[0, int((128 + 10*256 + 2*65536)*1e6/0xffffff), 65535]]
    verify_occupied_depth(np.array([7000, 8000, 9000, 10000, 65535]), [7.5, 11.])
    with pytest.raises(ValueError, match="occupied-vehicle"):
        verify_occupied_depth(np.full(10, 65535), [7.5, 11.])


def test_old_dataset_is_rejected_before_connect_or_train(tmp_path):
    write_json(tmp_path / "dataset_manifest.json", dict(schema="multiscene_roundtrip_collection_v1"))
    with pytest.raises(ValueError, match="new clean dataset"):
        capture.collect(tmp_path, "127.0.0.1", 2000)
    with pytest.raises(ValueError, match="clear_vehicle"):
        audit_dataset(tmp_path)


def test_capture_resume_audit_and_query_separation(tmp_path, monkeypatch):
    # Substitute synthetic sensors for CARLA; exercise normal collection, publication and audit.
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[10:110, 10:110] = 255
    projection = [8., 9., 112., 111.]
    crop, roi = crop_and_roi(mask, projection)
    depth = np.full((roi[3] - roi[1], roi[2] - roi[0]), 30000, dtype=np.uint16)
    poses = [[float(i * 65), 0., 0., 0., 0., 0.] for i in range(TRAIN_CASES + VAL_CASES)]
    world = SimpleNamespace(get_map=lambda: SimpleNamespace(get_spawn_points=lambda: poses))
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace())
    monkeypatch.setattr(capture, "connect", lambda *args: (object(), world, ["mock", "mock"]))
    monkeypatch.setattr(capture, "preflight", lambda *args: None)
    monkeypatch.setattr(capture, "pose_values", lambda pose: pose)
    completed = []

    def capture_car(carla, world, case, cfg, candidate, staging):
        i, car = case["index"], candidate["identity"]
        uid = case["scene_id"] + ":" + car
        directory = staging / car
        directory.mkdir()
        # Interrupt halfway through the second location; the first is already published.
        if (i, car) == (1, "B") and completed.count((1, "A")) == 1:
            raise RuntimeError("simulated capture interruption")
        samples = {}
        center, yaw = [case["anchor_pose"][0], 0., 220.], 15.
        for j, view in enumerate(orbit_views(i)):
            step, role = view["step"], view["role"]
            stem = f"{step:02d}_{role}"
            rgb, mask_file, depth_file = (directory / (stem + ext) for ext in (".png", "_mask.png", "_depth.png"))
            Image.new("RGB", (120, 120), (i + 70, j + 60, ord(car))).save(rgb)
            Image.fromarray(mask).save(mask_file)
            Image.fromarray(depth).save(depth_file)
            pose = list(camera_pose(center, yaw, view["azimuth"], view["elevation"], view["distance"]))
            prefix = f"{case['path']}/{car}/"
            samples[sample_key(uid, step, role)] = dict(
                **view, identity=uid, sensor_frame=j, pose=pose, size=[120, 120], bbox=crop,
                vehicle_center=center, vehicle_yaw=yaw, mask=dict(path=prefix+mask_file.name, sha256=sha(mask_file)),
                image=prefix+rgb.name, rgb_sha256=sha(rgb), projection=projection, far_corner_m=12.,
                empty_depth=dict(roi=roi, frame=j+32, pose=pose, path=prefix+depth_file.name, sha256=sha(depth_file)))
        completed.append((i, car))
        return samples

    monkeypatch.setattr(capture, "capture_car", capture_car)
    with pytest.raises(RuntimeError, match="interruption"):
        capture.collect(tmp_path, "local", 2000)
    assert not (tmp_path / "plan.json").exists()
    capture.collect(tmp_path, "local", 2000)
    assert completed.count((0, "A")) == completed.count((0, "B")) == 1
    routes = read_json(tmp_path / "plan.json")["routes"]
    for route in routes:
        samples = route["samples"]
        for step in route["steps"]:
            for w in step["writes"].values():
                obs, query = [samples[w[k]] for k in ("sample", "independent")]
                assert 3.999 < angular_gap(obs["azimuth"], query["azimuth"]) < 8.001
                assert obs["role"] == "write" and query["role"] == "query"
    counts, sheet = audit_dataset(tmp_path)
    assert sheet.size == (1295, 750)
    assert counts["images"] == 704
    assert counts["coverage"]["val"]["postwrite"] == 96
    assert counts["coverage"]["val"]["prewrite"] == 90
    sample = next(iter(routes[0]["samples"].values()))
    (tmp_path / sample["empty_depth"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="depth file changed"):
        audit_dataset(tmp_path)
