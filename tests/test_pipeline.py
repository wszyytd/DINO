import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from dinoexp.dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        rows = [
            dict(
                sample_id="reference",
                identity="A",
                role="reference",
                status="captured",
                crop="ref.png",
            ),
            dict(sample_id="B1", identity="B", role="query", status="captured", crop="b.png"),
            dict(sample_id="bad", identity="B", role="query", status="rejected", crop=""),
        ]
        for name in ["ref.png", "b.png"]:
            Image.new("RGB", (32, 32), "white").save(root / name)
        with (root / "metadata.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def test_rejected_views_are_skipped_and_identity_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            ref, queries, rejected = load_dataset(root)
            self.assertEqual(ref["identity"], "A")
            self.assertEqual([q["sample_id"] for q in queries], ["B1"])
            self.assertEqual(len(rejected), 1)

    def test_missing_query_image_does_not_silently_disappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            (root / "b.png").unlink()
            with self.assertRaises(FileNotFoundError):
                load_dataset(root)

    def test_near_black_reference_is_rejected_even_with_captured_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            image = Image.new("RGB", (100, 100), "black")
            image.putpixel((50, 50), (255, 255, 255))
            image.save(root / "ref.png")
            with self.assertRaisesRegex(ValueError, "near_black"):
                load_dataset(root)

    def test_duplicate_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.make_dataset(root)
            with (root / "metadata.csv").open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writerow(dict(rows[0], sample_id="ref2"))
            with self.assertRaises(ValueError):
                load_dataset(root)

    def test_incomplete_capture_cannot_be_evaluated_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            (root / "run.json").write_text(json.dumps({"status": "incomplete"}))
            with self.assertRaises(ValueError):
                load_dataset(root)

    def test_report_copies_images_and_generates_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, results = root / "data", root / "out"
            data.mkdir()
            results.mkdir()
            rows = self.make_dataset(data)
            query = dict(
                rows[1],
                sequence=0,
                view="side",
                weather="noon",
                position=0,
                feature_visibility="unknown",
                whole_score=0.85,
                local_score="",
                local_status="unavailable",
            )
            with (results / "scores.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=query.keys())
                writer.writeheader()
                writer.writerow(query)
            (results / "match_run.json").write_text(
                json.dumps(
                    dict(
                        reference=rows[0],
                        data=str(data),
                        model="synthetic-report-fixture",
                        local_compared=0,
                        rejected_views=1,
                    )
                )
            )
            completed = subprocess.run(
                [sys.executable, "report.py", "--results", str(results)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with Image.open(results / "scores.png") as plot:
                self.assertGreater(plot.width, 1000)
            with Image.open(results / "gallery/0001.png") as copied:
                self.assertEqual(copied.getpixel((0, 0)), (255, 255, 255))
            self.assertTrue((results / "report.html").is_file())

    def test_dry_run_works_without_carla_and_does_not_create_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "absent"
            completed = subprocess.run(
                [sys.executable, "capture.py", "--dry-run", "--output", str(out)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["query_count"], 12)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
