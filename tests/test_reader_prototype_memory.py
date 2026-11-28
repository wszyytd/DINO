import importlib.util
import unittest

if not importlib.util.find_spec("torch"):
    raise unittest.SkipTest("requires PyTorch")

import torch

from dinoexp.reader_prototype_memory import max_reader_scores, replay_reader, update_reader_bank
from test_prototype_memory import prototype


class ReaderPrototypeTests(unittest.TestCase):
    def test_reader_max_scores_each_candidate_independently_and_keeps_null_once(self):
        scores = torch.tensor([[3., 7., 6., 2., 9., .4], [8., 1., 0., 2., 3., .4]])
        values, indices = max_reader_scores(scores, (2, 3))
        torch.testing.assert_close(values, torch.tensor([[7., 9., .4], [8., 3., .4]]))
        self.assertEqual(indices.tolist(), [[1, 2], [0, 2]])

    def test_dedup_requires_good_scores_in_both_directions(self):
        old = prototype(0, 0, 1)
        tail = prototype(12, 1, 3)
        similar = prototype(23, 0, 5)
        bank = [old]
        directed = {(12, 0): 10., (0, 12): -1., (23, 0): 6., (0, 23): 7., (23, 12): 4., (12, 23): 2.}
        self.assertEqual(update_reader_bank(bank, tail, directed, 5.)["event"], "opened")
        result = update_reader_bank(bank, similar, directed, 5.)
        self.assertEqual(result["event"], "kept")
        self.assertEqual(result["bidirectional"], 6.)
        self.assertEqual([p.camera for p in bank], [0, 12])
        self.assertIs(bank[0], old)

    def test_full_bank_eviction_is_separate_and_keeps_one_whole_prototype(self):
        bank = [prototype(0, 0, 1), prototype(2, 0, 2)]
        tail = prototype(12, 1, 3)
        directed = {(12, 0): 1., (0, 12): 1., (12, 2): 1., (2, 12): 1., (0, 2): 4., (2, 0): 4.}
        old = bank[0]
        self.assertEqual(update_reader_bank(bank, tail, directed, 5., capacity=2)["event"], "rejected_full")
        result = update_reader_bank(bank, tail, directed, 5., capacity=2, evict=True)
        self.assertEqual(result["event"], "replaced_redundant")
        self.assertEqual(result["evicted_camera"], 2)
        self.assertEqual([p.camera for p in bank], [0, 12])
        self.assertIs(bank[0], old)
        self.assertIs(bank[1], tail)

    def test_new_comparison_reuses_original_grid_bank_and_queries(self):
        from dinoexp.evidence_memory import EvidenceMatcher
        from dinoexp.prototype_memory import replay
        from test_gallery_reader import fixture

        torch.manual_seed(19)
        torch.set_num_threads(1)
        model = EvidenceMatcher(dim=12, heads=3, slots=8).eval()
        route, probes, _, _ = fixture("val")
        directions = {0: 0, 23: 0, 2: 0, 22: 0, 20: 1, 16: 2, 17: 2, 12: 3, 13: 3, 14: 3}
        evidence = {(uid, cam): (torch.randn(3, 12), torch.ones(3, dtype=torch.bool), prototype(cam, axis, cam))
                    for uid in route["identities"] for cam, axis in directions.items()}
        reference, valid = torch.randn(8, 4, 12), torch.ones(8, 4, dtype=torch.bool)
        keys = [prototype(p["write_camera"], p["position"] - 1, 0).keys for p in probes]
        encoded = evidence, reference, valid, keys
        directed = {uid: {(a, b): float(directions[a] == directions[b]) for a in directions for b in directions if a != b}
                    for uid in route["identities"]}
        previous_rows, previous_events = replay(model, route, probes, encoded, .8)
        rows, events = replay_reader(model, route, probes, encoded, directed, {
            "grid": {"threshold": .8}, "reader": {"threshold": .8},
        })
        for old, event in zip(previous_events, events, strict=True):
            grid = event["writes"]["grid_keep_4x4"]
            self.assertEqual(old["writes"]["keep"], grid)
            self.assertEqual(grid, event["writes"]["grid_keep_max"])
            self.assertEqual(event["writes"]["unlimited_4x4"], event["writes"]["unlimited_max"])
        for old, row in zip(previous_rows, rows, strict=True):
            self.assertEqual(old["query"], row["query"])
            self.assertEqual(old["scores"]["keep"], row["scores"]["grid_keep_4x4"])
