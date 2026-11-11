"""CPU-only geometry/label checks: deliberately no torch or neural inference."""
import copy
import unittest
import numpy as np

from dinoexp.region_geometry import (DEFAULTS, clip_box, propose, pool_weights,
                                     label_regions, remap_positives, validate_config)


class RegionGeometryTests(unittest.TestCase):
    def test_fixed_config_rejects_degenerate_thresholds(self):
        validate_config(DEFAULTS)
        for section, key, value in [('proposal', 'max_candidates', 0), ('proposal', 'expansion', .5),
                                    ('label', 'min_coverage', 2), ('training', 'heads', 3),
                                    ('training', 'seeds', [7, 7])]:
            cfg = copy.deepcopy(DEFAULTS); cfg[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_clip_and_invalid(self):
        self.assertEqual(clip_box([-2, 3, 25, 80], (20, 40)), [0, 3, 20, 40])
        self.assertIsNone(clip_box([4, 5, 3, 8], (20, 40)))
        self.assertIsNone(clip_box([0, 0, float('nan'), 8], (20, 40)))

    def test_connected_cells_and_empty(self):
        heat = np.zeros((32, 32)); heat[5:7, 7:9] = 1; heat[25, 26] = 1
        result = propose(heat, (1024, 768), DEFAULTS['proposal'])
        self.assertEqual(len(result['before_merge']), 2)
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual(propose(np.ones((32, 32)), (1024, 768), DEFAULTS['proposal'])['candidates'], [])
        self.assertEqual(propose(np.zeros((32, 32)), (1024, 768), DEFAULTS['proposal'])['candidates'], [])

    def test_overlap_merging_records_sources(self):
        heat = np.zeros((32, 32)); heat[8, 8] = 1; heat[8, 10] = 1
        cfg = dict(DEFAULTS['proposal'], min_side=200)
        result = propose(heat, (1024, 768), cfg)
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0]['merged_from'], [0, 1])

    def test_letterbox_exact_area_and_padding(self):
        weights = pool_weights((1024, 768), [0, 0, 1024, 768])
        self.assertEqual(weights.shape, (16, 37*37))
        self.assertAlmostEqual(float(weights.sum()), 518*388, places=2)
        self.assertTrue((weights[:, :4*37] == 0).all())
        narrow = pool_weights((31, 101), [1, 2, 30, 100])
        self.assertTrue((narrow.sum(1) > 0).all())
        self.assertTrue((pool_weights((1024, 768), [3, 4, 2, 5]) == 0).all())

    def test_no_background_mask_in_pool(self):
        weights = pool_weights((518, 518), [0, 0, 518, 518])
        pooled = weights @ np.ones((1369, 2)) / weights.sum(1, keepdims=True)
        np.testing.assert_allclose(pooled, 1)

    def test_multi_positive_empty_ambiguous_and_fragment(self):
        a = np.zeros((100, 100), bool); a[10:30, 10:30] = True
        b = np.zeros_like(a); b[10:30, 40:60] = True
        boxes = [[8, 8, 32, 32], [5, 5, 35, 35], [0, 0, 70, 40], [10, 10, 12, 12], [90, 90, 90, 95]]
        label = label_regions(boxes, {'a': a, 'b': b}, 'a', DEFAULTS['label'])
        self.assertEqual(label['positives'], [0, 1])
        self.assertTrue(label['candidates'][2]['ambiguous'])
        self.assertTrue(label['candidates'][3]['fragment'])
        self.assertFalse(label['candidates'][4]['valid_box'])
        missing = label_regions([], {'a': a, 'b': b}, 'a', DEFAULTS['label'])
        self.assertTrue(missing['physical_visible'])
        self.assertEqual(missing['positives'], [])
        self.assertEqual(remap_positives([0, 2], [2, 1, 0]), [0, 2])
        self.assertEqual(remap_positives([0], [2, 0, 1]), [1])

    def test_ignore_is_not_confirmed_identity(self):
        a = np.ones((32, 32), bool)
        label = label_regions([[0, 0, 32, 32]], {'a': a}, 'a', DEFAULTS['label'], a)
        self.assertEqual(label['positives'], [])
        self.assertTrue(label['candidates'][0]['ambiguous'])


if __name__ == '__main__':
    unittest.main()
