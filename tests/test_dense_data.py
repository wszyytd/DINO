import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from dinoexp.dense_data import read_dense_manifest, make_targets


class DenseDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new('RGB', (4, 4), (120, 10, 30)).save(self.root / 'ref.png')
        Image.new('RGB', (4, 4), (30, 50, 70)).save(self.root / 'image.png')
        a = np.zeros((4, 4), np.uint8); a[0, 0] = 255
        b = np.zeros((4, 4), np.uint8); b[2:, 2:] = 255
        Image.fromarray(a).save(self.root / 'a.png'); Image.fromarray(b).save(self.root / 'b.png')
        self.sample = dict(id='sample', split='train', group='physical-scene',
            reference='ref.png', image='image.png',
            reference_sha256=self.digest('ref.png'), image_sha256=self.digest('image.png'),
            positive_masks=['a.png', 'b.png'], negative_masks=[], presence='present',
            supervision='simulation', mask_source='carla_instance_verified',
            mask_sha256={p: self.digest(p) for p in ('a.png', 'b.png')})

    def digest(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def write(self, samples):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(dict(format_version=1, task='reference_guided_dense_search', samples=samples)))
        return path

    def test_real_instances_receive_equal_mass_regardless_of_size(self):
        self.sample.update(supervision='confirmed_regions', reviewed=True, mask_source='reviewed_manual')
        manifest = read_dense_manifest(self.write([self.sample]))
        labels = make_targets(manifest['samples'][0], self.root, 4)
        self.assertAlmostEqual(float(labels['target'][0]), .5)
        self.assertAlmostEqual(float(labels['target'][10]), .125)
        self.assertFalse(labels['known'][1])
        self.assertEqual(float(labels['target'][-1]), 0.)

    def test_missing_masks_and_absence_contradictions_are_rejected(self):
        self.sample['positive_masks'] = []
        with self.assertRaisesRegex(ValueError, 'mask|present'):
            read_dense_manifest(self.write([self.sample]))
        self.sample.update(positive_masks=['a.png'], presence='absent')
        with self.assertRaisesRegex(ValueError, 'absent'):
            read_dense_manifest(self.write([self.sample]))

    def test_shared_reference_hash_cannot_cross_split_even_if_group_names_differ(self):
        other = dict(self.sample, id='other', split='val', group='different')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            read_dense_manifest(self.write([self.sample, other]))

    def test_changed_mask_or_nonbinary_mask_cannot_silently_change_labels(self):
        Image.new('L', (4, 4), 128).save(self.root / 'a.png')
        with self.assertRaisesRegex(ValueError, 'hash|binary'):
            read_dense_manifest(self.write([self.sample]))

    def test_absent_sample_targets_null(self):
        self.sample.update(positive_masks=[], presence='absent', mask_sha256={})
        manifest = read_dense_manifest(self.write([self.sample]))
        labels = make_targets(manifest['samples'][0], self.root, 2)
        np.testing.assert_array_equal(labels['target'], [0, 0, 0, 0, 1])
