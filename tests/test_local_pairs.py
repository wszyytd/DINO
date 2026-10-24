import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from dinoexp import local_pairs as pairs


class LocalPairsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scene = self.root / 'pilot'
        scene.mkdir()
        (scene / 'run.json').write_text(json.dumps({'status': 'complete'}))
        rows = []
        for name, identity, role, color in [
            ('reference', 'A', 'reference', 'white'),
            ('A_side', 'A', 'query', 'gray'),
            ('B_side', 'B', 'query', 'red'),
        ]:
            Image.new('RGB', (60, 30), color).save(scene / (name + '.png'))
            rows.append(dict(sample_id=name, identity=identity, role=role, status='captured',
                             crop=name + '.png', view='side', sequence='0'))
        with (scene / 'metadata.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_manifest_labels_are_relative_to_reference_not_filename(self):
        m = pairs.build_manifest(self.root, 'pilot', 'same-capture')
        self.assertEqual([r['label'] for r in m['pairs']], [1, 0])
        self.assertEqual(m['pairs'][0]['reference'], 'pilot/reference.png')
        self.assertEqual(m['pairs'][0]['group'], 'same-capture')
        pairs.verify_images(m, self.root)

    def test_modified_image_is_rejected(self):
        m = pairs.build_manifest(self.root, 'pilot', 'capture')
        Image.new('RGB', (60, 30), 'blue').save(self.root / 'pilot/A_side.png')
        with self.assertRaisesRegex(ValueError, 'hash'):
            pairs.verify_images(m, self.root)

    def test_repeated_named_view_in_different_conditions_is_not_one_ranking_group(self):
        target = self.root / 'pilot/metadata.csv'
        with target.open(newline='') as f:
            rows = list(csv.DictReader(f))
        rows[1]['weather'], rows[2]['weather'] = 'noon', 'rain'
        with target.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[1]))
            writer.writeheader()
            writer.writerows(rows)
        m = pairs.build_manifest(self.root, 'pilot', 'capture')
        self.assertNotEqual(m['pairs'][0]['comparison_group'], m['pairs'][1]['comparison_group'])

    def test_group_overlap_and_duplicate_image_block_validation(self):
        m = pairs.build_manifest(self.root, 'pilot', 'capture')
        with self.assertRaisesRegex(ValueError, 'group'):
            pairs.check_split(m['pairs'], m['pairs'])
        other = [dict(r, group='different-name') for r in m['pairs']]
        with self.assertRaisesRegex(ValueError, 'image'):
            pairs.check_split(m['pairs'], other)

    def test_unknown_label_and_escaping_path_are_rejected(self):
        m = pairs.build_manifest(self.root, 'pilot', 'capture')
        m['pairs'][0]['label'] = -1
        with self.assertRaises(ValueError):
            pairs.validate_manifest(m)
        m['pairs'][0]['label'] = 1
        m['pairs'][0]['query'] = '../escape.png'
        with self.assertRaises(ValueError):
            pairs.verify_images(m, self.root)

    def test_metrics_do_not_force_target_absent_group_to_have_a_winner(self):
        rows = [dict(comparison_group='v1', label=1, view='side'),
                dict(comparison_group='v1', label=0, view='side'),
                dict(comparison_group='v2', label=0, view='top')]
        result = pairs.ranking_metrics(rows, [0.7, 0.2, 0.9])
        self.assertEqual(result['ranking_groups'], 1)
        self.assertEqual(result['target_absent_groups'], 1)
        self.assertEqual(result['target_top1'], 1.0)
        self.assertEqual(result['by_view']['side']['ranking_groups'], 1)


if __name__ == '__main__':
    unittest.main()
