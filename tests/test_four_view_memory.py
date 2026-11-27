import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class FourViewTests(unittest.TestCase):
    def setUp(self):
        from dinoexp.four_view_memory import VIEWS
        self.route = dict(id="pair1_val", group="spawn-64", split="val",
                          identities=["pair1_val:A", "pair1_val:B"], samples={})
        for uid in self.route["identities"]:
            for _, w, q in VIEWS:
                for cam in (w, q):
                    key = f"{uid}/{cam:02}"
                    self.route["samples"][key] = dict(identity=uid, camera_index=cam, azimuth=cam*15.,
                        rgb_sha256=key, frame=str(cam), mask={"path": key + ".png"})

    def test_protocol_checks_every_vehicle_and_ignores_natural_route_roles(self):
        from dinoexp.four_view_memory import select_views
        pairs = select_views(self.route, 25)
        self.assertEqual([(p["write_camera"], p["query_camera"]) for p in pairs[:4]],
                         [(0, 1), (20, 21), (16, 15), (12, 11)])
        self.assertEqual(len(pairs), 8)
        self.route["samples"]["pair1_val:B/15"]["azimuth"] = 0.
        with self.assertRaisesRegex(ValueError, "angle"):
            select_views(self.route, 25)
        self.route["samples"]["pair1_val:B/15"]["azimuth"] = 225.
        self.route["samples"]["pair1_val:A/15"]["rgb_sha256"] = "pair1_val:A/12"
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_views(self.route, 25)

    def test_four_prefixes_use_only_seen_photos_and_revisit_identical_questions(self):
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.four_view_memory import inspect_sequence, select_views, summarize
        from dinoexp.symmetric_memory_eval import gallery_state
        torch.manual_seed(12)
        torch.set_num_threads(1)
        model = EvidenceMatcher(dim=12, heads=3, slots=4).eval()
        pairs = select_views(self.route, 25)
        def feature():
            return torch.randn(3, 12), torch.ones(3, dtype=torch.bool)
        obs = {p["write"]: feature() for p in pairs}
        queries = {p["query"]: feature() for p in pairs}
        rows, gates = inspect_sequence(model, self.route, pairs, obs, queries)
        self.assertEqual((len(rows), len(gates)), (32, 6))
        state, singles = None, []
        with torch.inference_mode():
            for step, cam in enumerate((0, 20, 16, 12), 1):
                frame = [obs[f"{uid}/{cam:02}"] for uid in self.route["identities"]]
                x, valid = (torch.stack(items)[None] for items in zip(*frame, strict=True))
                state = model.update(x, valid, state)
                singles.append(model.update(x, valid))
                subset = [r for r in rows if r["step"] == step]
                self.assertEqual({r["probe"] for r in subset}, set(queries))
                for row in subset:
                    q, m = queries[row["probe"]]
                    def read(s, query=q, mask=m):
                        return model.score(query[None], mask[None], s)[0]
                    expected = dict(updated=read(state), written=read(singles[-1]), gallery=read(gallery_state(singles)),
                                    score_mean=torch.stack([read(s) for s in singles]).mean(0))
                    for method, logits in expected.items():
                        torch.testing.assert_close(torch.tensor(row["scores"][method]["logits"]), logits,
                                                   atol=2e-6, rtol=1e-5)
        stats = summarize(rows)
        summary = [r for r in stats if r["group"] == "all" and r["method"] == "updated"]
        self.assertEqual([r["old_changes"]["n"] for r in summary], [0, 2, 4, 6])
        self.assertEqual([r["new_changes"]["n"] for r in summary], [0, 2, 2, 2])
        self.assertEqual([r["all_right_n"] for r in summary], [2]*4)

    def test_server_entrypoint_writes_each_checkpoint_with_fake_features(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        import run_experiment8_four_views as entry
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.region_ab_data import read_json, write_json
        cfg = dict(plan_sha256="plan", dataset_sha256="dataset", teacher_sha256="teacher",
                   backbone_sha256="backbone", source_checkpoints={"7": "init"}, crop_mode="foreground")
        model = EvidenceMatcher(dim=12, heads=3, slots=4)
        def encode(runtime, root, route, pairs):
            def features():
                return torch.randn(2, 12), torch.ones(2, dtype=torch.bool)
            return {p["write"]: features() for p in pairs}, {p["query"]: features() for p in pairs}
        with TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            run = dict(cfg, baseline=tmp, status="complete", variant=entry.VARIANT, seed=7,
                       completed_epochs=list(entry.MILESTONES), source_checkpoint_sha256="init")
            write_json(checkpoints / "run.json", run)
            for epoch in entry.MILESTONES:
                folder = checkpoints / f"epoch_{epoch:03}"
                folder.mkdir()
                cp = dict(run, epoch=epoch, kind="memory_replay_matcher_v1", config=model.config, model_state=model.state_dict())
                torch.save(cp, folder / "last.pt")
            args = SimpleNamespace(checkpoints=checkpoints, baseline=None, output=Path(tmp)/"results")
            runtime = (SimpleNamespace(device="cpu"), cfg, "backbone")
            with patch.object(entry, "read_reference", return_value=(cfg, {"max_query_angle": 25})), \
                    patch.object(entry, "audit_plan", return_value=([], [self.route], {}, {})), \
                    patch("dinoexp.memory_retention_eval.frozen_runtime", return_value=runtime), \
                    patch("dinoexp.four_view_memory.encode_sequence", side_effect=encode) as encoder, redirect_stdout(StringIO()):
                dest = entry.run(args)
            self.assertEqual(encoder.call_count, 1)
            info = read_json(dest / "run.json")
            self.assertEqual(info["status"], "complete")
            self.assertEqual(set(info["checkpoints"]), {"30","50","75","100"})
            self.assertIn("32 槽", (dest / "report.md").read_text(encoding="utf-8"))
            for epoch in entry.MILESTONES:
                folder = dest / f"epoch_{epoch:03}"
                self.assertEqual(len(read_json(folder / "questions.json")), 32)
                self.assertEqual(len(read_json(folder / "gates.json")), 6)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                entry.load_writer(checkpoints/"epoch_030", cfg, 7, "cpu", epoch=100)
