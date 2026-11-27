import unittest
import importlib.util
from itertools import combinations

if not importlib.util.find_spec("torch"):
    raise unittest.SkipTest("requires PyTorch")

import torch

from dinoexp.evidence_memory import EvidenceMatcher
from dinoexp.four_view_memory import VIEWS, inspect_sequence, select_views
from dinoexp.gallery_reader import (TRAIN_VIEWS, subset_schedule, freeze_for_gallery, cache_route,
                                   train_epoch, evaluate, report)
from dinoexp.symmetric_memory_eval import gallery_state


def fixture(split):
    views = VIEWS if split == "val" else TRAIN_VIEWS
    name = f"pair2_{split}"
    ids = [name + ":A", name + ":B"]
    route = dict(id=name, split=split, group="spawn-64" if split == "val" else "spawn-42", identities=ids, samples={})
    for uid in ids:
        for _, w, q in views:
            for cam in (w, q):
                key = f"{uid}/{cam:02}"
                route["samples"][key] = dict(identity=uid, camera_index=cam, frame=str(cam),
                    azimuth=(cam*15) % 360, mask={"path": key}, rgb_sha256=key)
    pairs = select_views(route, 25, views=views, split=split)
    def features():
        return torch.randn(4, 12), torch.ones(4, dtype=torch.bool)
    return route, pairs, {p["write"]: features() for p in pairs}, {p["query"]: features() for p in pairs}


class GalleryReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.model = EvidenceMatcher(dim=12, heads=3, slots=8)
        freeze_for_gallery(self.model)

    def test_rear_context_has_no_query_and_all_subsets_mix_directions(self):
        route, _, obs, queries = fixture("train")
        views = (*TRAIN_VIEWS, ("rear", 16, 17), ("tail", 11, None))
        for uid in route["identities"]:
            for camera in (11, 16, 17):
                key = f"{uid}/{camera:02}"
                route["samples"][key] = dict(identity=uid, camera_index=camera, frame=str(camera),
                    azimuth=camera*15, rgb_sha256=key, mask={"path": key})
                bank = queries if camera == 17 else obs
                bank[key] = torch.randn(4, 12), torch.ones(4, dtype=torch.bool)
        pairs = select_views(route, 25, views=views, split="train")
        cached = cache_route(self.model, route, pairs, obs, queries)
        self.assertEqual(len(cached["frames"]), 7)
        self.assertEqual(len(cached["pairs"]), 12)
        self.assertEqual([p["write_camera"] for p in pairs if p["query"] is None], [11, 11])
        subsets = subset_schedule(7, 30, 3, rear_views=(5, 6))
        for row in subsets:
            for subset in row:
                self.assertTrue(any(i < 5 for i in subset) and any(i >= 5 for i in subset))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad])
        group = ((0, 6), (0, 5, 6), (0, 1, 5, 6))
        seen = []
        hook = self.model.reader.register_forward_pre_hook(lambda m, x: seen.append((x[0].shape[0], x[1].shape[1])))
        try:
            train_epoch(self.model, optimizer, {route["id"]: cached}, {route["id"]: group}, 7)
        finally:
            hook.remove()
        self.assertEqual(seen, [(4, 16), (8, 24), (12, 32)])

    def test_review_uses_camera10_only_when_enabled_and_rejects_validation_images(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from dinoexp.region_ab_data import write_json
        from prepare_experiment8_gallery_rear import KIND, load_review

        train, *_ = fixture("train")
        val, *_ = fixture("val")
        samples = {f"{uid}/{cam:02}": dict(identity=uid, camera_index=cam, role="query",
                    rgb_sha256=f"rear {uid}/{cam:02}") for uid in train["identities"] for cam in (17, 10)}
        review = dict(kind=KIND, base_plan_sha256="base", cameras=[17, 10],
                      samples={train["id"]: samples})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/"review.json", review)
            for with_tail, expected in ((False, {17}), (True, {10, 17})):
                routes, roots = load_review(root, "base", [train], [val], include_tail10=with_tail)
                self.assertEqual(set(roots.values()), {root})
                self.assertEqual({routes[0]["samples"][k]["camera_index"] for k in roots}, expected)
                self.assertTrue(set(roots).isdisjoint(train["samples"]))
            key = train["identities"][0] + "/17"
            samples[key]["rgb_sha256"] = next(iter(val["samples"].values()))["rgb_sha256"]
            write_json(root/"review.json", review)
            with self.assertRaisesRegex(ValueError, "invalid or missing"):
                load_review(root, "base", [train], [val])

    def test_training_reads_only_included_views_and_only_reader_and_match_change(self):
        route, pairs, obs, queries = fixture("train")
        cached = cache_route(self.model, route, pairs, obs, queries)
        schedule = subset_schedule(5, 30, 7)
        for size in (2, 3, 4):
            used = [row[size-2] for row in schedule]
            for subset in combinations(range(5), size):
                self.assertEqual(used.count(subset), 30 // len(set(used)))
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        seen_shapes = []
        hook = self.model.reader.register_forward_pre_hook(lambda m, x: seen_shapes.append(
            (x[0].shape[0], x[1].shape[1])))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        try:
            stats = train_epoch(self.model, optimizer, {route["id"]: cached}, {route["id"]: schedule[0]}, 7)
        finally:
            hook.remove()
        self.assertEqual(stats["updates"], 1)
        self.assertEqual(seen_shapes, [(4*s, 8*s) for s in (2, 3, 4)])
        changed = [k for k,v in self.model.state_dict().items() if not torch.equal(before[k], v)]
        self.assertTrue(any(k.startswith("match.") for k in changed))
        self.assertTrue(any(k.startswith("reader.") for k in changed))
        self.assertTrue(all(k.startswith(("reader.", "match.")) for k in changed))

    def test_ablation_has_identical_queries_and_targets_for_both_cars(self):
        route, pairs, obs, queries = fixture("val")
        cached = cache_route(self.model, route, pairs, obs, queries)
        rows = evaluate(self.model, {route["id"]: cached})
        reference, _ = inspect_sequence(self.model, route, pairs, obs, queries)
        prefix_scores = {(r["probe"], r["step"]): r["scores"]["gallery"]["logits"] for r in reference}
        self.assertEqual(len(rows), 8)
        for r in rows:
            for step in range(1, 5):
                torch.testing.assert_close(torch.tensor(r["conditions"][f"prefix_{step}"]["logits"]),
                                           torch.tensor(prefix_scores[r["probe"],step]), atol=2e-6, rtol=1e-5)
            full = r["conditions"]["prefix_4"]
            self.assertEqual(full["cameras"], [0, 20, 16, 12])
            for omit, (_, w, _) in enumerate(VIEWS, 1):
                dropped = r["conditions"][f"drop_{w:02}"]
                self.assertEqual(dropped["cameras"], [c for c in full["cameras"] if c != w])
                self.assertAlmostEqual(dropped["margin_change"], dropped["margin"]-full["margin"])
                self.assertEqual(len(dropped["logits"]), 3)
                query = r["probe"]
                frames = [cached["frames"][i] for i in range(4) if i != omit-1]
                tokens, valid = queries[query]
                with torch.no_grad():
                    expected = self.model.score(tokens[None],valid[None],gallery_state(frames))[0]
                torch.testing.assert_close(torch.tensor(dropped["logits"]), expected, atol=2e-6, rtol=1e-5)
        self.assertIn("drop_16", report({"before": rows, "after": rows}))

    def test_server_entrypoint_and_saved_weights_on_synthetic_inputs(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        import train_experiment8_gallery as entry
        from dinoexp.region_ab_data import read_json
        config = dict(dataset_sha256="data", teacher_sha256="teacher", backbone_sha256="backbone", plan_sha256="plan")
        training, valid = fixture("train"), fixture("val")
        feats = {r["id"]: (o, q) for r, _, o, q in (training, valid)}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root/"epoch_100"
            folder.mkdir()
            (folder/"last.pt").write_bytes(b"synthetic checkpoint")
            args = SimpleNamespace(epochs=2, checkpoints=root, baseline=None, output=root/"output")
            with patch.object(entry, "experiment_source", return_value=(root, config, {"max_query_angle":25})), \
                    patch.object(entry, "audit_plan", return_value=([training[0]], [valid[0]], {}, {})), \
                    patch.object(entry, "load_writer", return_value=self.model), \
                    patch("dinoexp.memory_retention_eval.frozen_runtime",return_value=(SimpleNamespace(device="cpu"),config,"backbone")), \
                    patch("dinoexp.four_view_memory.encode_sequence",side_effect=lambda _, __, r, p: feats[r["id"]]), \
                    redirect_stdout(StringIO()):
                out = entry.run(args)
            info = read_json(out/"run.json")
            self.assertEqual((info["status"], info["source_epoch"], info["updates"]), ("complete", 100, 2))
            self.assertEqual(len(read_json(out/"train.json")), 2)
            saved = torch.load(out/"last.pt", map_location="cpu", weights_only=True)
            new = EvidenceMatcher(**saved["config"])
            new.load_state_dict(saved["model_state"], strict=True)
            freeze_for_gallery(new)
            recomputed = evaluate(new, {valid[0]["id"]: cache_route(new, *valid)})
            recorded = read_json(out/"after_questions.json")
            for actual, expected in zip(recorded, recomputed, strict=True):
                self.assertEqual(actual["probe"], expected["probe"])
                for name, scores in expected["conditions"].items():
                    observed = actual["conditions"][name]
                    self.assertEqual(observed["decision"], scores["decision"])
                    torch.testing.assert_close(torch.tensor(observed["logits"]), torch.tensor(scores["logits"]),
                                               rtol=1e-5, atol=1e-6)
            self.assertEqual(len(read_json(out/"before_questions.json")), len(recomputed))
