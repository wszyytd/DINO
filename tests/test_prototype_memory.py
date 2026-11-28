import importlib.util
import unittest

if not importlib.util.find_spec("torch"):
    raise unittest.SkipTest("requires PyTorch")

import torch

from dinoexp.prototype_memory import CAPACITY, SEQUENCES, Prototype, choose_threshold, replay, summarize, update_bank


def prototype(camera, axis, slots):
    key = torch.eye(4)[axis]
    return Prototype(camera, camera, {"foreground_mean": key, "foreground_grid": key.repeat(16, 1)},
                     torch.full((8, 12), float(slots)))


class PrototypeTests(unittest.TestCase):
    def test_duplicate_updates_only_one_slot_and_novel_late_view_can_enter(self):
        for policy, camera, value in (("keep", 0, 1), ("mean", 23, 2), ("latest", 23, 3)):
            with self.subTest(policy=policy):
                bank = [prototype(0, 0, 1), prototype(20, 1, 2)]
                unaffected = bank[1]
                update_bank(bank, prototype(23, 0, 3), policy, .8, 3)
                self.assertEqual(len(bank), 2)
                self.assertIs(bank[1], unaffected)
                self.assertEqual(bank[0].camera, camera)
                self.assertEqual(bank[0].slots.mean().item(), value)
                result = update_bank(bank, prototype(12, 3, 5), policy, .8, 3)
                self.assertEqual(result["event"], "opened")
                self.assertEqual(bank[-1].camera, 12)
                result = update_bank(bank, prototype(16, 2, 7), policy, .8, 3)
                self.assertEqual(result["event"], "rejected_full")
                self.assertEqual(len(bank), 3)

    def test_fifo_and_unlimited_admit_new_observation(self):
        for policy, expected in (("fifo", [20, 23]), ("unlimited", [0, 20, 23])):
            bank = [prototype(0, 0, 1), prototype(20, 1, 2)]
            update_bank(bank, prototype(23, 0, 3), policy, .8, 2)
            self.assertEqual([p.camera for p in bank], expected)

    def test_threshold_from_labeled_pairs_balances_duplicate_and_novel(self):
        threshold, stats = choose_threshold([.81, .84, .95], [.65, .71, .90])
        self.assertGreater(threshold, .71)
        self.assertLessEqual(threshold, .81)
        self.assertEqual((stats["duplicate_correct"], stats["novel_correct"]), (3, 2))

    def test_asynchronous_replay_leaves_queries_out_of_writes_and_counts_all_steps(self):
        from dinoexp.evidence_memory import EvidenceMatcher
        from test_gallery_reader import fixture

        torch.manual_seed(9)
        torch.set_num_threads(1)
        model = EvidenceMatcher(dim=12, heads=3, slots=8).eval()
        route, probes, _, _ = fixture("val")
        ids = route["identities"]
        directions = {0: 0, 23: 0, 2: 0, 22: 0, 20: 1, 16: 2, 17: 2, 12: 3, 13: 3, 14: 3}
        evidence = {(uid, cam): (torch.randn(3, 12), torch.ones(3, dtype=torch.bool), prototype(cam, axis, cam))
                    for uid in ids for cam, axis in directions.items()}
        references, valid = torch.randn(8, 4, 12), torch.ones(8, 4, dtype=torch.bool)
        keys = [prototype(p["write_camera"], p["position"]-1, 0).keys for p in probes]
        rows, events = replay(model, route, probes, (evidence, references, valid, keys), .8)
        self.assertEqual((len(rows), len(events)), (304, 40))
        self.assertEqual({p["query"] for p in probes}, {r["query"] for r in rows})
        for sequence in SEQUENCES:
            self.assertEqual([r["identity"] for r in events if r["sequence"] == sequence], ids*10)
        self.assertTrue(all(event["camera"] not in {p["query_camera"] for p in probes} for event in events))
        for event in events:
            for method in ("fifo", "keep", "mean", "latest"):
                self.assertTrue(all(len(bank) <= CAPACITY for bank in event["writes"][method]["banks"]))
            if event["sequence"] == "rear_arrives_late" and event["step"] == 14:
                self.assertTrue(all(12 in bank for bank in event["writes"]["keep"]["banks"]))
        self.assertEqual(len(summarize(rows, events)), 12)
