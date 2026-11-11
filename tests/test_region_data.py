"""Filesystem/provenance checks only; no neural modules imported."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dinoexp.region_ab_data import (write_json, read_json, save_features, load_bundle,
                                    sha, code_record)


class RegionDataTests(unittest.TestCase):
    def build(self, root):
        for name in ('proposals', 'A', 'B', 'references'):
            (root/name).mkdir()
        rows = [dict(id='pair', candidates=[{'box': [0, 0, 30, 30]}], reference_sha256='ref')]
        write_json(root/'proposals/records.json', {'records': rows})
        write_json(root/'labels.json', {'pair': {'positives': [0], 'physical_visible': True}})
        artifacts = {name: sha(root/name) for name in ('proposals/records.json', 'labels.json')}
        np.savez_compressed(root/'references/ref.npz', tokens=np.ones((64, 256), 'float32'), valid=np.ones(64, bool))
        for variant in ('A', 'B'):
            entry = save_features(root/variant, 'pair', np.ones((1, 16, 256), 'float32'), np.ones((1, 16), bool), [])
            write_json(root/variant/'cache.json', dict(kind='region_encoding_'+variant+'_v1',
                       proposal_sha256=artifacts['proposals/records.json'], labels_sha256=artifacts['labels.json'], entries={'pair': entry}))
            artifacts[variant+'/cache.json'] = sha(root/variant/'cache.json')
        write_json(root/'prepared.json', dict(status='complete', kind='region_encoding_ab_v1', artifacts=artifacts,
                   sources={}, code=code_record(), references={'ref': {'file': 'references/ref.npz', 'sha256': sha(root/'references/ref.npz')}}))

    def test_shared_ab_contract_and_changed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.build(root)
            _, rows, labels, _, features = load_bundle(root, root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(labels['pair']['positives'], [0])
            self.assertEqual(features['A']['pair']['observation'].shape, (1, 16, 256))
            write_json(root/'labels.json', {'pair': {'positives': []}})
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                load_bundle(root, root)

    def test_wrong_b_proposal_file_rejected_even_if_index_hash_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.build(root)
            index = read_json(root/'B/cache.json'); index['proposal_sha256'] = 'wrong'
            write_json(root/'B/cache.json', index)
            meta = read_json(root/'prepared.json'); meta['artifacts']['B/cache.json'] = sha(root/'B/cache.json')
            write_json(root/'prepared.json', meta)
            with self.assertRaisesRegex(ValueError, 'A/B proposal/label'):
                load_bundle(root, root)


if __name__ == '__main__':
    unittest.main()
