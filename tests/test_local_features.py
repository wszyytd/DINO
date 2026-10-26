import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional training dependencies absent')
class LocalFeaturesTests(unittest.TestCase):
    def test_cls_baseline_always_reads_final_layer_even_for_early_patch_layers(self):
        import json
        from unittest.mock import patch
        import torch
        from dinoexp.local_features import extract_cache, FeatureCache
        from dinoexp.local_pairs import sha256
        class Backbone(torch.nn.Module):
            embed_dim = 8

            def get_intermediate_layers(self, x, n, **kwargs):
                return [(torch.full((len(x), 4, 8), float(i)),
                         torch.full((len(x), 8), float(i))) for i in n]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'hubconf.py').write_text('# fixture')
            for i in range(2):
                Image.new('RGB', (28, 14), (100 + i * 20, 50, 60)).save(root / f'{i}.png')
            r, q = (sha256(root / f'{i}.png') for i in range(2))
            manifest = root / 'pairs.json'
            manifest.write_text(json.dumps(dict(format_version=1, pairs=[dict(reference='0.png',
                query='1.png', reference_sha256=r, query_sha256=q, label=1, group='g',
                comparison_group='v', view='side', sample_id='q')])))
            torch.save({}, root / 'backbone.pt')
            with patch('torch.hub.load', return_value=Backbone()):
                extract_cache([manifest], root, root / 'features', root, root / 'backbone.pt',
                              size=28, layers=[3, 6], device='cpu')
            item = FeatureCache(root / 'features').get(q)
            np.testing.assert_array_equal(item['cls'], np.full(8, 11))
            self.assertEqual(item['tokens'].shape, (2, 4, 8))

    def test_letterbox_mask_excludes_padding_not_vehicle(self):
        from dinoexp.local_features import preprocess
        image, valid = preprocess(Image.new('RGB', (140, 28), 'white'), 140)
        self.assertEqual(image.shape, (3, 140, 140))
        self.assertEqual(valid.shape, (100,))
        self.assertFalse(valid[:30].any())
        self.assertTrue(valid[40:60].all())

    def test_cache_checks_file_hash_and_complete_state(self):
        import json
        from dinoexp.local_features import FeatureCache
        from dinoexp.local_pairs import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / ('a' * 64 + '.npz')
            np.savez_compressed(target, tokens=np.ones((1, 4, 8), dtype=np.float16),
                                valid=np.ones(4, dtype=bool), cls=np.ones(8, np.float32))
            config = dict(model='test', image_size=28, layers=[1], input_dim=8,
                          backbone_sha256='b' * 64, preprocess='letterbox-imagenet-v1')
            meta = dict(status='complete', format_version=1, config=config,
                        images={'a' * 64: {'sha256': sha256(target)}})
            (root / 'cache.json').write_text(json.dumps(meta))
            cache = FeatureCache(root)
            self.assertEqual(cache.get('a' * 64)['tokens'].shape, (1, 4, 8))
            target.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'hash'):
                cache.get('a' * 64)


if __name__ == '__main__':
    unittest.main()
