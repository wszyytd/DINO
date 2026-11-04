from dataclasses import replace
import unittest

from test_evidence_experiment import small_episode
from dinoexp.evidence_roles import make_manifests, select_roles


def episode(split, position, pair, target):
    name = f'{split}_pair{pair}_location{position}_light0_ref{"AB"[target]}'
    e = small_episode(name, target)
    initial = replace(e.initial, histories=tuple(tuple(replace(o, image_hash=f'{name}:{o.id}')
                       for o in history) for history in e.initial.histories))
    outcomes = {k: replace(o, image_hash=f'{name}:{k}' if o.valid else None)
                for k, o in e.outcomes.items()}
    return replace(e, position=str(position), group=f'position:{position}',
                   reference_hash=f'{name}:ref', initial=initial, outcomes=outcomes,
                   identities=(f'{split}_car0', f'{split}_car1'), source_split=split,
                   capture_group=name.rsplit('_ref', 1)[0])


class EvidenceRoleTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [episode(s, p, pair, t) for s, p, pairs in
                         [('train', 42, 4), ('train', 64, 2), ('val', 70, 2), ('test', 80, 2)]
                         for pair in range(pairs) for t in (0, 1)]

    def test_nested_subsets_keep_pairs_and_fixed_validation(self):
        manifests = make_manifests(self.episodes, ['64'], [.25, .5, 1.], seed=7)
        previous = set()
        for m in manifests:
            roles, audit = select_roles(self.episodes, m)
            ids = set(m['roles']['repr'])
            self.assertTrue(previous <= ids)
            previous = ids
            self.assertEqual(len(roles['value']), 4)
            self.assertEqual(len(roles['eval']), 4)
            self.assertEqual(len(audit['reserved_test']), 4)
            self.assertEqual(len({e.capture_group for e in roles['repr']})*2, len(ids))
        self.assertEqual(len(previous), 8)

    def test_rejects_split_pair_unknown_id_and_identity_leak(self):
        m = make_manifests(self.episodes, ['64'], [1.])[0]
        m['roles']['value'].append(m['roles']['repr'].pop())
        with self.assertRaisesRegex(ValueError, 'position|group'):
            select_roles(self.episodes, m)
        m = make_manifests(self.episodes, ['64'], [1.])[0]
        m['roles']['eval'].append('unknown')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            select_roles(self.episodes, m)
        m = make_manifests(self.episodes, ['64'], [1.])[0]
        leaked = [replace(e, identities=('train_car0',)) if e.source_split=='val' else e
                  for e in self.episodes]
        with self.assertRaisesRegex(ValueError, 'identity'):
            select_roles(leaked, m)

    def test_incomplete_ab_group_cannot_form_learning_curve(self):
        with self.assertRaisesRegex(ValueError, 'A/B'):
            make_manifests(self.episodes[1:], ['64'], [1.])

    def test_missing_identity_in_one_evaluation_episode_is_rejected(self):
        m = make_manifests(self.episodes, ['64'], [1.])[0]
        changed = [replace(e, identities=()) if e.id==m['roles']['eval'][0] else e
                   for e in self.episodes]
        with self.assertRaisesRegex(ValueError, 'identity metadata'):
            select_roles(changed, m)


if __name__ == '__main__':
    unittest.main()
