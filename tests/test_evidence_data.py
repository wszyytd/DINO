import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dinoexp.evidence_data import load_episodes, split_by_position, FeatureBank
from dinoexp.local_pairs import sha256


def make_capture(root):
    scene, cache = root / 'captures' / 'scene', root / 'cache'
    scene.mkdir(parents=True)
    cache.mkdir()
    (scene / 'run.json').write_text(json.dumps({'status': 'complete'}))
    images, rows = {}, []
    for index, (name, role, identity, status) in enumerate([
            ('ref', 'reference', 'A', 'captured'),
            ('a0', 'query', 'A', 'captured'), ('b0', 'query', 'B', 'captured'),
            ('a1', 'query', 'A', 'captured'), ('b1', 'query', 'B', 'rejected')]):
        path = scene / f'{name}.png'
        if status == 'captured':
            Image.new('RGB', (16, 16), (40 + 30*index, 60, 80)).save(path)
            digest = sha256(path)
            tokens = np.ones((1, 16, 8), dtype=np.float32)
            tokens[:, :4] = 1e9  # Padded pixels must not affect pooled features.
            target = cache / f'{digest}.npz'
            np.savez_compressed(target, tokens=tokens,
                                valid=np.array([False]*4 + [True]*12),
                                cls=np.ones(8, dtype=np.float32))
            images[digest] = {'sha256': sha256(target)}
        rows.append(dict(sample_id=name, role=role, identity=identity, status=status,
                         crop=path.name, sequence=index, position='42', view=name,
                         camera_pose=json.dumps([10, 0, 10, 0, 0, 0]),
                         target_pose=json.dumps([0, 0, 0, 0, 0, 0])))
    with (scene / 'metadata.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = dict(model='test', image_size=56, layers=[12], input_dim=8,
                  backbone_sha256='b'*64, preprocess='letterbox-imagenet-v1')
    (cache / 'cache.json').write_text(json.dumps(dict(
        status='complete', format_version=1, config=config, images=images)))
    return scene, cache


class EvidenceDataTests(unittest.TestCase):
    def test_rejected_actions_remain_but_missing_captured_image_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, cache = make_capture(Path(tmp))
            episodes, audit = load_episodes(scene.parent, cache)
            self.assertEqual(audit['episodes'], 1)
            self.assertEqual({a.id for a in episodes[0].actions}, {'a1', 'b1'})
            self.assertFalse(episodes[0].outcomes['b1'].valid)
            (scene / 'a1.png').unlink()
            with self.assertRaises(FileNotFoundError):
                load_episodes(scene.parent, cache)

    def test_exact_reference_replay_and_cross_role_image_leak_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, cache = make_capture(Path(tmp))
            episode = load_episodes(scene.parent, cache)[0][0]
            other = replace(episode, id='other', position='64')
            with self.assertRaisesRegex(ValueError, 'image overlap'):
                split_by_position([episode, other], {'repr': ['42'], 'value': ['64']})
            (scene / 'a1.png').write_bytes((scene / 'ref.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'exact replay'):
                load_episodes(scene.parent, cache)

    def test_feature_pooling_masks_padding_and_rejects_modified_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, cache = make_capture(Path(tmp))
            digest = sha256(scene / 'ref.png')
            bank = FeatureBank(cache, [digest], grid=2)
            np.testing.assert_array_equal(bank.get(digest)['tokens'].numpy(),
                                          np.ones((1, 4, 8)))
            with (cache / f'{digest}.npz').open('ab') as stream:
                stream.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'hash'):
                FeatureBank(cache, [digest], grid=2)


if __name__ == '__main__':
    unittest.main()
