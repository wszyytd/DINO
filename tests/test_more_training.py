import argparse
import json
from pathlib import Path
import tempfile
import unittest

from train_more_photos import find_collection, make_plan


class MoreTrainingTests(unittest.TestCase):
    def test_selects_latest_complete_collection_not_an_interrupted_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, status in [('more_photos_01', 'complete'),
                                 ('more_photos_02', 'incomplete')]:
                path = root / name
                path.mkdir()
                (path / 'collection.json').write_text(json.dumps(dict(status=status)))
            self.assertEqual(find_collection(root).name, 'more_photos_01')

    def test_training_uses_validation_and_evaluates_best_checkpoint_on_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'collection.json').write_text(json.dumps(dict(status='complete')))
            for split in ('train', 'val', 'test'):
                rows = [dict(reference=f'{split}/r.png', query=f'{split}/{i}.png',
                    reference_sha256=({'train': 'a', 'val': 'b', 'test': 'c'}[split]) * 64,
                    query_sha256=({'train': 'd', 'val': 'e', 'test': 'f'}[split]) * 63 + str(i),
                    label=i, group=split, comparison_group=split, view='side', sample_id=str(i))
                    for i in range(2)]
                (root / f'pairs_{split}.json').write_text(json.dumps(dict(format_version=1, pairs=rows)))
            args = argparse.Namespace(output=root / 'new', device='cpu', epochs=2,
                batch_size=2, threads=1, model_repo=None, backbone=None)
            plan = make_plan(root, args)
            self.assertEqual(len(plan), 4)
            self.assertIn('--val-manifest', plan[1])
            self.assertNotIn('--allow-train-only', plan[1])
            self.assertIn(str(root / 'pairs_test.json'), plan[-1])
            self.assertTrue(any(str(x).endswith('best.pt') for x in plan[-1]))
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
