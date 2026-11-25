"""Integration checks against the original photos when the local dataset is present."""

import shutil

import pytest
from PIL import Image

from dinoexp.memory_retention import angular_gap, open_crop
from dinoexp.region_ab_data import read_json
from prepare_experiment8_offline import REVIEWED, SOURCE, prepare, read_case


pytestmark = pytest.mark.skipif(not (SOURCE / "collection.json").is_file(), reason="requires fixed-height capture")


def test_reviewed_boxes_have_disjoint_cameras_and_location_splits(tmp_path):
    output = tmp_path / "boxes"
    counts = prepare(SOURCE, output)
    assert [(counts[k]["writes"], counts[k]["postwrite"]) for k in ("train", "val")] == [(48, 30), (48, 36)]
    groups = {}
    for route in read_json(output / "plan.json")["routes"]:
        groups.setdefault(route["split"], set()).add(route["group"])
        samples = route["samples"]
        write_frames, query_frames = set(), set()
        for step in route["steps"]:
            for w in step["writes"].values():
                image = samples[w["sample"]]
                assert open_crop(output, image).size == tuple(image["size"])
                write_frames.add(image["camera_index"])
                if w["independent"]:
                    query = samples[w["independent"]]
                    assert 12 < angular_gap(image["azimuth"], query["azimuth"]) < 20
                    query_frames.add(query["camera_index"])
        assert write_frames.isdisjoint(query_frames)
    assert groups == {"train": {"spawn-42"}, "val": {"spawn-64"}}


def test_editing_a_reviewed_crop_requires_new_review(tmp_path):
    case = next(c for c in read_json(SOURCE / "collection.json")["plan"]["cases"]
                if c["name"] == "pair0_location0_refA")
    copy = tmp_path / case["name"]
    shutil.copytree(SOURCE / case["name"], copy, ignore=shutil.ignore_patterns("images"))
    Image.new("RGB", (100, 100)).save(copy / "crops/A_00_orbit_000.png")
    stage = tmp_path / "prepared"
    stage.mkdir()
    with pytest.raises(ValueError, match="reviewed images or camera metadata differ"):
        read_case(tmp_path, stage, case, 0, REVIEWED[0][0])
