import json
import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from dinoexp.suite import build_cases, reuse_reference


class SuiteTests(unittest.TestCase):
    def test_shared_reference_preserves_source_metadata_and_query_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / "source", root / "dest"
            for directory, frame, color in [(source, "100", "white"), (dest, "200", "gray")]:
                (directory / "images").mkdir(parents=True)
                (directory / "crops").mkdir()
                for rel in ["images/reference.png", "crops/reference.png", "crops/query.png"]:
                    Image.new("RGB", (32, 32), color).save(directory / rel)
                rows = [
                    dict(
                        sample_id="reference",
                        identity="A",
                        blueprint="car",
                        role="reference",
                        frame=frame,
                        image="images/reference.png",
                        crop="crops/reference.png",
                    ),
                    dict(
                        sample_id="query",
                        identity="B",
                        blueprint="car",
                        role="query",
                        frame="202",
                        image="images/reference.png",
                        crop="crops/query.png",
                    ),
                ]
                with (directory / "metadata.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                (directory / "run.json").write_text(json.dumps(dict(status="complete")))
            query_before = (dest / "crops/query.png").read_bytes()
            reuse_reference(source, dest)
            self.assertEqual(
                (dest / "crops/reference.png").read_bytes(),
                (source / "crops/reference.png").read_bytes(),
            )
            self.assertEqual((dest / "crops/query.png").read_bytes(), query_before)
            with (dest / "metadata.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["frame"], "100")
            self.assertEqual(rows[1]["frame"], "202")
            with Image.open(dest / "crops/local_reference.png") as backup:
                self.assertEqual(backup.getpixel((0, 0)), (128, 128, 128))
            run = json.loads((dest / "run.json").read_text())
            self.assertEqual(run["reference_source"]["scene"], "source")

    def test_swapping_position_keeps_target_identity_and_color(self):
        base = json.loads(Path("configs/two_candidates.json").read_text())
        spec = dict(
            cases=[
                dict(name="left", distractor_color="185,185,185", swap=False),
                dict(name="right", distractor_color="185,185,185", swap=True),
            ]
        )
        cases = build_cases(base, spec)
        self.assertEqual(cases[0][1]["candidates"][0]["identity"], "A")
        self.assertEqual(cases[1][1]["candidates"][0]["identity"], "B")
        for _, cfg in cases:
            self.assertEqual(cfg["target_identity"], "A")
            a = next(c for c in cfg["candidates"] if c["identity"] == "A")
            b = next(c for c in cfg["candidates"] if c["identity"] == "B")
            self.assertEqual(a["color"], "255,255,255")
            self.assertEqual(b["color"], "185,185,185")
        self.assertEqual(base["candidates"][1]["color"], "220,220,220")

    def test_suite_rejects_names_that_escape_output_directory(self):
        base = json.loads(Path("configs/two_candidates.json").read_text())
        with self.assertRaises(ValueError):
            build_cases(
                base,
                dict(cases=[dict(name="../elsewhere", distractor_color="185,185,185", swap=False)]),
            )


if __name__ == "__main__":
    unittest.main()
