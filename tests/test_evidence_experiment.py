import tempfile
from pathlib import Path
import unittest

import torch

from dinoexp.evidence_data import Episode
from dinoexp.evidence_state import Action, Observation, ReplayState
from dinoexp.evidence_model import EvidenceModel, teacher_targets
from dinoexp.evidence_training import train_memory, train_value, load_memory, load_value
from dinoexp.evidence_labels import FrozenRuntime, generate_labels, fingerprint
from dinoexp.evidence_policy import rollout, summarize_rollouts


class SmallBank:
    def __init__(self):
        torch.manual_seed(7)
        self.config = dict(input_dim=8, layers=[12], pooled_grid=2, backbone_sha256='test')
        self.items = {}
        for i in range(9):
            x = torch.randn(1, 1, 4, 8)
            mask = torch.ones(1, 4, dtype=torch.bool)
            cls = x[:, -1].mean(1)
            target, target_valid = teacher_targets(x, mask, cls)
            self.items[str(i)] = dict(tokens=x[0], valid=mask[0], cls=cls[0],
                                      teacher=target[0], teacher_valid=target_valid[0])

    def get(self, digest):
        return self.items[digest]


def small_episode(name='scene', target=0):
    pose = (0., 1., 0., 1., 0.)
    a0, b0 = Observation('a0', 'A', '1', pose), Observation('b0', 'B', '2', pose)
    a1 = Observation('a1', 'A', '3', (1., 0., 0., 1., 0.))
    b1 = Observation('b1', 'B', '4', (-1., 0., 0., 1., 0.))
    failed = Observation('fail', 'A', None, pose, valid=False)
    actions = (Action('a1', 0, a1.pose, 'side'), Action('b1', 1, b1.pose, 'side'),
               Action('fail', 0, pose, 'blocked'))
    return Episode(name, name, name, '0', ('A', 'B'), target,
                   ReplayState(('A', 'B'), ((a0,), (b0,))), actions,
                   {'a1': a1, 'b1': b1, 'fail': failed})


class EvidenceExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        self.bank = SmallBank()
        self.model = EvidenceModel(input_dim=8, dim=16, layers=1, heads=4,
                                   memory_tokens=3, blocks=1)
        self.episode = small_episode()

    def test_labels_have_no_future_or_target_in_policy_inputs_and_failure_is_zero(self):
        runtime = FrozenRuntime(self.model, self.bank)
        records = generate_labels(runtime, [self.episode], states_per_episode=2, seed=4)
        self.assertTrue(records)
        first = records[0]
        self.assertEqual(set(first['inputs']),
                         {'q', 'memories', 'scores', 'counts', 'subjects', 'poses'})
        failed_index = first['action_ids'].index('fail')
        self.assertEqual(first['targets'][0, failed_index].item(), 0.)
        original = fingerprint(self.model)
        with tempfile.TemporaryDirectory() as tmp:
            train_value(runtime, records, Path(tmp) / 'value', steps=4, seed=4)
        self.assertEqual(original, fingerprint(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_repeated_rollouts_do_not_inflate_episode_count(self):
        runtime = FrozenRuntime(self.model, self.bank)
        traces = [rollout(runtime, self.episode, 'random', budget=3, seed=seed)
                  for seed in (7, 8, 9)]
        report = summarize_rollouts(traces, budget=3)
        self.assertEqual(report['episodes'], 1)
        self.assertEqual(report['rollouts'], 3)
        self.assertEqual(report['independent_groups'], 1)

    def test_staged_training_reload_rejects_wrong_feature_or_upstream_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_memory(self.model, self.bank, [self.episode], root / 'memory',
                         steps=3, alpha=.1, seed=5)
            loaded, meta = load_memory(root / 'memory' / 'last.pt', self.bank.config)
            self.assertEqual(meta['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'feature'):
                load_memory(root / 'memory' / 'last.pt', dict(self.bank.config, pooled_grid=1))
            runtime = FrozenRuntime(loaded, self.bank)
            records = generate_labels(runtime, [self.episode], states_per_episode=2, seed=5)
            train_value(runtime, records, root / 'value', steps=3, seed=5)
            value = load_value(root / 'value' / 'last.pt', runtime.fingerprint)
            self.assertIsNotNone(value)
            with self.assertRaisesRegex(ValueError, 'upstream'):
                load_value(root / 'value' / 'last.pt', 'wrong-model')

    def test_rollout_budget_failure_accounting_and_summary_include_all_episodes(self):
        runtime = FrozenRuntime(self.model, self.bank)
        trace = rollout(runtime, self.episode, 'fixed', budget=3)
        self.assertEqual(trace['attempts'], 3)
        self.assertEqual(trace['failed_attempts'], 1)
        self.assertEqual(trace['cost'], 3.)
        self.assertEqual(len(trace['steps']), 4)
        self.assertEqual(trace['status'], 'UNRESOLVED')
        report = summarize_rollouts([trace], budget=3)
        self.assertEqual(report['episodes'], 1)
        self.assertEqual(len(report['rank_accuracy_by_budget']), 4)
        self.assertEqual(report['penalized_confirmation_attempts'], 3)
        a = rollout(runtime, self.episode, 'random', budget=2, seed=10)
        b = rollout(runtime, self.episode, 'random', budget=2, seed=10)
        self.assertEqual([s['action'] for s in a['steps']], [s['action'] for s in b['steps']])

    def test_auxiliary_ablation_uses_identical_matching_curriculum(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = [self.episode, small_episode('another', target=1)]
            histories = []
            for index, alpha in enumerate((0., .1)):
                model = EvidenceModel(**self.model.config)
                train_memory(model, self.bank, episodes, root / str(index),
                             steps=5, alpha=alpha, seed=12)
                records = json.loads((root / str(index) / 'history.json').read_text())
                histories.append([(r['episode_id'], r['history_ids']) for r in records])
            self.assertEqual(histories[0], histories[1])


if __name__ == '__main__':
    unittest.main()
