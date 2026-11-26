import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from PIL import Image

from dinoexp.memory_retention import open_crop
from dinoexp.region_ab_data import sha
from prepare_experiment8_cutouts import attach_masks
from run_experiment8_followup import compare_cutouts


def test_mask_pack_is_bound_to_original_pixels_and_removed_background(tmp_path):
    root = tmp_path / "data"
    (root / "scene").mkdir(parents=True)
    rgb = Image.new("RGB", (20, 16), (91, 65, 39))
    rgb.save(root / "scene/A_00.png")
    sample = dict(image="scene/A_00.png", bbox=[0, 0, 20, 16], size=[20, 16],
                  rgb_sha256=sha(root / "scene/A_00.png"))
    plan = dict(routes=[dict(samples={"scene:A/00": sample})])
    mask = Image.new("L", rgb.size)
    mask.paste(255, (5, 3, 16, 13))
    data = BytesIO()
    mask.save(data, format="PNG")
    path = tmp_path / "masks.zip"
    member = "scene/A_00.mask.png"
    records = {"scene:A/00": dict(mask=member, rgb_sha256=sample["rgb_sha256"],
                                   mask_sha256=hashlib.sha256(data.getvalue()).hexdigest())}
    with ZipFile(path, "w") as pack:
        pack.writestr("index.json", json.dumps(dict(records=records)))
        pack.writestr(member, data.getvalue())
    attach_masks(root, plan, path)
    actual = open_crop(root, sample, foreground=True)
    assert actual.getpixel((0, 0)) == (0, 0, 0)
    assert actual.getpixel((10, 10)) == rgb.getpixel((10, 10))
    assert open_crop(root, sample).tobytes() == rgb.tobytes()
    sample["rgb_sha256"] = "other-photo"
    with pytest.raises(ValueError, match="RGB"):
        attach_masks(root, plan, path)


def test_fixed_weight_and_retraining_results_are_paired_by_photo():
    base = [dict(route="r", probe=p, kind="new_view", after={"decision": d})
            for p, d in (("carA", "target"), ("carB", "other"))]
    fixed = [dict(route="r", probe=p, kind="new_view", scores={"original": {"decision": d}})
             for p, d in (("carB", "target"), ("carA", "other"))]
    tuned = [dict(row, scores={"original": {"decision": "target"}}) for row in fixed]
    rows, report = compare_cutouts(base, fixed, tuned)
    assert "all | 1 / 1 / 2 | 2 | 1/1 | 1/0" in report
    assert rows[0]["probe"] == "carA" and rows[0]["same_weights_cutout"]["decision"] == "other"
    with pytest.raises(ValueError, match="questions differ"):
        compare_cutouts(base, fixed[:1], tuned)
