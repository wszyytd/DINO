import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from dinoexp.parts import discover_scenes, export_prediction, make_roi_annotations
from segment_parts import run_inference, selected_classes


def dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            sample_id="reference",
            identity="A",
            role="reference",
            status="captured",
            crop="r.png",
        ),
        dict(
            sample_id="query",
            identity="B",
            role="query",
            status="captured",
            crop="q.png",
        ),
        dict(sample_id="bad", identity="B", role="query", status="rejected", crop=""),
    ]
    for name in ["r.png", "q.png"]:
        Image.new("RGB", (80, 40), "white").save(root / name)
    with (root / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


class PartsTests(unittest.TestCase):
    def test_discovery_and_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root / "left")
            dataset(root / "right")
            scenes = discover_scenes(root)
            self.assertEqual([s["name"] for s in scenes], ["left", "right"])
            self.assertEqual(len(scenes[0]["rows"]), 2)
            self.assertEqual(scenes[0]["rejected"][0]["sample_id"], "bad")
            self.assertEqual(len(discover_scenes(root / "left")), 1)
            (root / "right" / "run.json").write_text(
                json.dumps({"status": "incomplete"})
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                discover_scenes(root)

    def test_native_mask_and_no_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = Image.new("RGB", (80, 40), "white")
            mask = np.zeros((40, 80), dtype=bool)
            mask[10:20, 30:50] = True
            r = export_prediction(
                im,
                [{"class_name": "left_front-door", "confidence": 0.8, "mask": mask}],
                root,
                "sample",
            )[0]
            self.assertEqual(r["bbox_xyxy"], [30, 10, 50, 20])
            self.assertEqual(r["mask_pixels"], 200)
            with Image.open(root / r["mask"]) as saved:
                self.assertEqual(saved.size, (80, 40))
            with Image.open(root / r["crop"]) as saved:
                self.assertEqual(saved.size, (20, 10))
            empty = export_prediction(im, [], root, "empty")[0]
            self.assertEqual(empty["status"], "no_detection")
            self.assertIsNone(empty["confidence"])
            self.assertFalse(empty["testability_validated"])

    def test_bad_mask_size_and_empty_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            im = Image.new("RGB", (80, 40), "white")
            with self.assertRaisesRegex(ValueError, "native"):
                export_prediction(
                    im,
                    [
                        {
                            "class_name": "roof",
                            "confidence": 0.8,
                            "mask": np.ones((64, 64)),
                        }
                    ],
                    Path(tmp),
                    "bad",
                )
            r = export_prediction(
                im,
                [{"class_name": "roof", "confidence": 0.8, "mask": np.zeros((40, 80))}],
                Path(tmp),
                "empty",
            )[0]
            self.assertEqual(r["status"], "empty_mask")

    def test_roi_ambiguity_and_opposite_side(self):
        records = [
            dict(
                sample_id="reference",
                class_name="left_front-door",
                status="detected",
                bbox_normalized=[0.1, 0.2, 0.5, 0.8],
            ),
            dict(
                sample_id="opposite",
                class_name="right_front-door",
                status="detected",
                bbox_normalized=[0.1, 0.2, 0.5, 0.8],
            ),
            dict(
                sample_id="duplicate",
                class_name="left_front-door",
                status="detected",
                bbox_normalized=[0.1, 0.2, 0.5, 0.8],
            ),
            dict(
                sample_id="duplicate",
                class_name="left_front-door",
                status="detected",
                bbox_normalized=[0.2, 0.2, 0.6, 0.8],
            ),
        ]
        roi = make_roi_annotations(
            records,
            "reference",
            ["reference", "opposite", "duplicate"],
            "left_front-door",
        )
        self.assertTrue(roi["reference"]["visible"])
        self.assertFalse(roi["opposite"]["visible"])
        self.assertFalse(roi["duplicate"]["visible"])
        self.assertEqual(roi["duplicate"]["reason"], "ambiguous_multiple_detections")

    def test_one_pixel_part_is_saved_but_not_exported_as_comparable_roi(self):
        records = [
            dict(
                sample_id="reference",
                class_name="roof",
                status="detected",
                bbox_normalized=[0.1, 0.2, 0.5, 0.8],
                crop_width=30,
                crop_height=20,
            ),
            dict(
                sample_id="tiny",
                class_name="roof",
                status="detected",
                bbox_normalized=[0.1, 0.2, 0.11, 0.21],
                crop_width=1,
                crop_height=1,
            ),
        ]
        roi = make_roi_annotations(records, "reference", ["reference", "tiny"], "roof")
        self.assertFalse(roi["tiny"]["visible"])
        self.assertEqual(roi["tiny"]["reason"], "roi_too_small_for_existing_matcher")

    def test_dry_run_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root / "data")
            result = subprocess.run(
                [
                    sys.executable,
                    "segment_parts.py",
                    "--data",
                    str(root / "data"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["total_crops"], 2)
            self.assertEqual(plan["rejected_captures"], 1)

    def test_injected_result_adapter_writes_portable_report_and_roi(self):
        class TensorFixture:
            def __init__(self, values):
                self.values = np.asarray(values)

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class BoxesFixture:
            cls = TensorFixture([0])
            conf = TensorFixture([0.8])

            def __len__(self):
                return 1

        class ModelFixture:
            names = {0: "left_front-door", 1: "left_front-window", 2: "roof"}
            task = "segment"
            bad_shape = False

            def predict(self, source, **settings):
                self.settings = settings
                shape = (64, 64) if self.bad_shape else (source.height, source.width)
                mask = np.zeros(shape)
                mask[10:20, 30:50] = 1
                return [
                    SimpleNamespace(
                        orig_shape=(source.height, source.width),
                        boxes=BoxesFixture(),
                        masks=SimpleNamespace(data=TensorFixture([mask])),
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root / "data")
            weights = root / "fixture.pt"
            weights.write_bytes(b"synthetic adapter fixture, never a model")
            args = SimpleNamespace(
                output=root / "out",
                data=root / "data",
                weights=weights,
                parts=["door", "window", "roof"],
                imgsz=768,
                conf=0.25,
                iou=0.7,
                device="cpu",
            )
            model = ModelFixture()
            run_inference(
                args, discover_scenes(args.data), model, weights, {"fixture": True}
            )
            run = json.loads((args.output / "parts_run.json").read_text())
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["samples"], 2)
            self.assertTrue(model.settings["retina_masks"])
            self.assertEqual(len(run["scenes"][0]["rejected_captures"]), 1)
            roi = json.loads(
                (args.output / run["scenes"][0]["roi_files"][0]).read_text()
            )
            self.assertTrue(roi["query"]["visible"])
            records = json.loads((args.output / "parts.json").read_text())
            for sample in records:
                for r in sample["records"]:
                    for key in ["source", "overlay", "crop", "mask", "masked_crop"]:
                        self.assertTrue((args.output / r[key]).is_file())
            self.assertTrue((args.output / "index.html").is_file())
            import re

            links = re.findall(
                r'(?:src|href)="([^"]+)"',
                (args.output / "index.html").read_text(encoding="utf-8"),
            )
            self.assertTrue(all((args.output / link).is_file() for link in links))
            model.bad_shape = True
            args.output = root / "failed"
            with self.assertRaisesRegex(ValueError, "native"):
                run_inference(
                    args, discover_scenes(args.data), model, weights, {"fixture": True}
                )
            failed = json.loads((args.output / "parts_run.json").read_text())
            self.assertEqual(failed["status"], "incomplete")

    def test_coco_weights_cannot_be_mistaken_for_vehicle_parts(self):
        with self.assertRaisesRegex(ValueError, "vehicle"):
            selected_classes({0: "person", 1: "car"}, ["door"])


if __name__ == "__main__":
    unittest.main()
