import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class LongWriterTests(unittest.TestCase):
    def test_second_write_reads_the_actual_four_states_and_averages_raw_logits(self):
        import torch
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.experiment8_long_eval import inspect_route
        from dinoexp.symmetric_memory_eval import gallery_state

        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = EvidenceMatcher(dim=12, heads=3, slots=4).eval()
        ids = ["scene:A", "scene:B"]
        samples, writes, queries = {}, {}, {}
        for uid in ids:
            for cam in (0, 12, 1, 22, 11, 14):
                key = f"{uid}/{cam:02}"
                samples[key] = dict(identity=uid, camera_index=cam)
                cache = writes if cam in (0, 12) else queries
                cache[key] = torch.randn(3, 12), torch.ones(3, dtype=torch.bool)
        route = dict(id="scene", identities=ids, samples=samples)
        rows, gates = inspect_route(model, route, (writes, queries))
        self.assertEqual((len(rows), len(gates)), (16, 4))
        with torch.inference_mode():
            def frame(cam):
                return tuple(torch.stack(ts)[None] for ts in zip(*(writes[f"{uid}/{cam:02}"] for uid in ids),
                                                                   strict=True))
            for history, first, second in (("FR", 0, 12), ("RF", 12, 0)):
                previous = model.update(*frame(first))
                written = model.update(*frame(second))
                updated = model.update(*frame(second), previous)
                p, w = previous["slots"][0], written["slots"][0]
                u, _ = model.updater(p, w, torch.ones(w.shape[:2], dtype=torch.bool))
                g = model.gate(torch.cat((p, u), -1)).sigmoid()
                observed = [r["gate"] for r in gates if r["history"] == history]
                torch.testing.assert_close(torch.tensor(observed), g.squeeze(-1))
                for row in (r for r in rows if r["history"] == history):
                    x, mask = queries[row["probe"]]
                    states = dict(previous=previous, written=written, proposal=dict(previous, slots=u[None]),
                                  updated=updated, gallery=gallery_state([previous, written]))
                    logits = {name: model.score(x[None], mask[None], state)[0] for name, state in states.items()}
                    logits["score_mean"] = (logits["previous"] + logits["written"]) / 2
                    for method, expected in logits.items():
                        result = row["scores"][method]
                        torch.testing.assert_close(torch.tensor(result["logits"]), expected, atol=2e-6, rtol=1e-5)
                        label = ids.index(row["identity"])
                        self.assertAlmostEqual(result["margin"], (expected[label]-expected[1-label]).item(), places=5)
        self.assertFalse(model.gate._forward_hooks or model.updater._forward_hooks)

    def test_reproduction_detects_offsetting_identity_changes_and_logit_drift(self):
        from dinoexp.experiment8_long_eval import compare_replay

        def row(probe, decision, margin):
            return dict(kind="new_view", route="r", probe=probe,
                        after=dict(decision=decision, target=margin, other=0., null=-1.))
        reference = [row("A", "target", 1.), row("B", "other", -.5)]
        current = [row("B", "target", .25), row("A", "other", -1.)]
        check = compare_replay(reference, current)
        self.assertEqual((check["reference_correct"], check["current_correct"], check["agree"]), (1, 1, 0))
        self.assertEqual(check["max_abs_logit_diff"], 2.)
        with self.assertRaises(ValueError):
            compare_replay(reference, current[:1])

    def test_route_losses_are_recorded_before_batch_averaging(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        import torch
        from dinoexp.memory_replay_training import _fit
        from dinoexp.region_ab_data import read_json

        # Stub the objective and optimizer; exercise only logging and the epoch callback.
        net = torch.nn.Linear(1, 1)
        values = {"pair0": 0.1, "pair1": 0.6, "pair2": 0.8}
        train = [dict(id=k) for k in values]
        args = SimpleNamespace(epochs=2, batch_routes=2, lr=.0001, weight_decay=.01,
                               independent_weight=1., postwrite_weight=1.)
        def terms(model, value, *_):
            loss = model.weight.sum()*0 + value
            return loss*3, loss, loss, loss
        callback = Mock()
        with TemporaryDirectory() as tmp, patch("torch.optim.AdamW"), \
                patch("dinoexp.memory_replay_training.route_objective", side_effect=terms):
            _fit(net, "replay_prepost", args, 7, train, values, values, {}, Path(tmp), after_epoch=callback)
            log = read_json(Path(tmp) / "train.json")
        self.assertEqual([v["updates"] for v in log], [2, 4])
        self.assertEqual(callback.call_count, 2)
        for epoch in log:
            self.assertAlmostEqual(epoch["main_ce"], .5)
            for key, expected in values.items():
                self.assertAlmostEqual(epoch["by_route"][key]["main_ce"], expected)

    def test_four_checkpoints_through_entry_point_without_training(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch
        import run_experiment8_long as exp
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.memory_replay import audit_plan
        from dinoexp.memory_replay_training import _validate
        from dinoexp.region_ab_data import read_json, sha, write_json

        # Synthetic observations and queries only. Exercise evaluation, persistence and
        # old-run matching while replacing the optimizer loop and the image encoder.
        torch.set_num_threads(1)
        torch.manual_seed(7)
        base_model = EvidenceMatcher().eval()
        routes, features = [], {}
        for split in ("train", "val"):
            name = "pair2_" + split
            ids = [name + ":A", name + ":B"]
            route = dict(id=name, split=split, group=split, identities=ids, samples={}, steps=[])
            observations, queries = {}, {}
            for uid in ids:
                for cam in (0, 23, 12, 13, 1, 22, 11, 14):
                    key = f"{uid}/{cam:02}"
                    route["samples"][key] = dict(identity=uid, azimuth=cam*15, rgb_sha256=key, camera_index=cam)
                    observations[key] = torch.randn(16, 256), torch.ones(16, dtype=torch.bool)
                    queries[key] = torch.randn(64, 256), torch.ones(64, dtype=torch.bool)
            for step, (w, q) in enumerate(((0, 1), (13, 14))):
                route["steps"].append(dict(step=step, frame=str(step), writes={
                    uid: dict(sample=f"{uid}/{w:02}", independent=f"{uid}/{q:02}", previous_view_gap=None,
                              probe_previous_view_gap=None) for uid in ids}))
            routes.append(route)
            features[name] = observations, queries

        def encode(_runtime, _data, route, *, foreground, extra_writes):
            self.assertTrue(foreground)
            self.assertEqual(len(extra_writes), 4 if route["split"] == "val" else 0)
            return features[route["id"]]

        def no_fit(model, variant, args, seed, train, schedules, enc, fixed, folder, *, after_epoch):
            self.assertEqual((variant, args.epochs, seed), ("replay_prepost", 100, 7))
            for epoch in range(1, 101):
                after_epoch(model, {"epoch": epoch, "updates": epoch * 2})

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, fg = root / "data", root / "foreground"
            data.mkdir()
            orig = fg / "replay_prepost-s7"
            orig.mkdir(parents=True)
            write_json(data / "dataset_manifest.json", {"schema": "fixed_height_auto_cutouts_v1"})
            digest = sha(data / "dataset_manifest.json")
            plan = dict(kind="memory_retention_plan_v1", routes=routes, dataset_sha256=digest,
                        max_query_angle=25, novelty_angle=45)
            write_json(data / "plan.json", plan)
            _, val, _, coverage = audit_plan(plan, digest)
            config = dict(status="complete", crop_mode="foreground", epochs=30, variants=[exp.VARIANT],
                          plan_sha256=sha(data / "plan.json"), dataset_sha256=digest, coverage=coverage,
                          source_checkpoints={"7": "initial"}, teacher_sha256="teacher", backbone_sha256="backbone",
                          lr=.0001, batch_routes=2, weight_decay=.01, independent_weight=1., postwrite_weight=1.)
            write_json(fg / "summary.json", config)
            write_json(orig / "validation.json", _validate(base_model, val, features, 45))
            torch.save({"model_state": base_model.state_dict()}, orig / "last.pt")
            initial = [({"checkpoint_sha256": "initial"}, base_model.config, base_model.state_dict())]
            runtime = (SimpleNamespace(device="cpu"), {"teacher_sha256": "teacher"}, "backbone")
            args = SimpleNamespace(baseline=root, source_run=root, output=root/"long")
            with patch("dinoexp.memory_retention_eval.load_checkpoints", return_value=initial), \
                    patch("dinoexp.memory_retention_eval.frozen_runtime", return_value=runtime), \
                    patch("dinoexp.memory_retention_eval.encode_route_features", side_effect=encode), \
                    patch("dinoexp.memory_replay_training._fit", side_effect=no_fit), redirect_stdout(StringIO()):
                output = exp.run(args)
            self.assertEqual(read_json(output / "run.json")["completed_epochs"], [30, 50, 75, 100])
            self.assertEqual(sorted(p.name for p in output.glob("epoch_*")),
                             ["epoch_030", "epoch_050", "epoch_075", "epoch_100"])
            for epoch in exp.MILESTONES:
                dest = output / f"epoch_{epoch:03}"
                saved = torch.load(dest / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual((saved["epoch"], saved["updates"]), (epoch, 2*epoch))
                summary = read_json(dest / "summary.json")
                self.assertIn("pair2_train", summary["train_at_checkpoint"])
                self.assertEqual(len(summary["lincoln_b"]), 4)
                self.assertEqual(len(read_json(dest / "gates.json")), 4)
                self.assertEqual(len(read_json(dest / "questions.json")), 16)
            check = read_json(output / "epoch_030/summary.json")["reproduction"]
            self.assertEqual(check["agree"], check["n"])
            self.assertTrue(check["weights_identical"])
            self.assertIn("score_mean", (output / "report.md").read_text(encoding="utf-8"))
