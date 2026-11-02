import unittest
import numpy as np
from PIL import Image
from dinoexp.dense_views import bilinear_map, make_views


class DenseViewTests(unittest.TestCase):
    def test_four_neighbor_weights_and_image_boundary_are_hand_checked(self):
        indices, weights = bilinear_map(np.array([[.5, .5], [0., 0.], [.25, .25]]), 2)
        self.assertEqual(indices[0].tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(weights[0], [.25, .25, .25, .25])
        sums = np.bincount(indices[1], weights=weights[1], minlength=4)
        np.testing.assert_allclose(sums, [1, 0, 0, 0])
        np.testing.assert_allclose(np.bincount(indices[2], weights=weights[2], minlength=4), [1, 0, 0, 0])

    def test_letterbox_padding_cannot_contribute_to_source_grid(self):
        result = make_views(Image.new('RGB', (56, 28)), image_size=56, tile_size=28,
                            rows=1, cols=2, grid_size=4)
        self.assertEqual(len(result['images']), 3)
        self.assertEqual(result['valid'].shape, (3, 16))
        self.assertFalse(result['valid'][0, :4].any())
        self.assertFalse(result['valid'][0, -4:].any())
        self.assertTrue(result['valid'][0, 4:12].all())
        self.assertEqual(float(result['weights'][0, :4].sum()), 0.)
        np.testing.assert_allclose(result['weights'].sum(-1), result['valid'].astype(float))

    def test_small_images_keep_full_plus_fifteen_views_without_coordinates_outside_image(self):
        result = make_views(Image.new('RGB', (30, 20)), image_size=28, grid_size=2)
        self.assertEqual(len(result['images']), 16)
        self.assertTrue((result['indices'] >= 0).all())
        self.assertTrue((result['indices'] < 4).all())
        self.assertTrue(np.isfinite(result['geometry']).all())
        with self.assertRaises(ValueError):
            make_views(Image.new('RGB', (30, 20)), image_size=29)

    def test_partial_padding_uses_visible_area_as_geometry_weight(self):
        result=make_views(Image.new('RGB',(28,14)),image_size=28,rows=1,cols=1,grid_size=2)
        np.testing.assert_allclose(result['weights'][0].sum(-1),[.5,.5,.5,.5])

    def test_partial_patch_maps_visible_content_centroid_without_native_grid_holes(self):
        result=make_views(Image.new('RGB',(28,14)),image_size=28,rows=1,cols=1,grid_size=4)
        covered=np.bincount(result['indices'][0].reshape(-1),weights=result['weights'][0].reshape(-1),minlength=16)
        self.assertTrue((covered>0).all(),covered.reshape(4,4))
