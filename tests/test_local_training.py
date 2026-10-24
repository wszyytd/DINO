import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional training dependencies absent')
class LocalTrainingTests(unittest.TestCase):
    def test_training_requires_both_classes_and_explicit_smoke_mode(self):
        from dinoexp.local_training import validate_training_setup
        rows = [dict(group='a', label=1, reference_sha256='x', query_sha256='y')]
        with self.assertRaisesRegex(ValueError, 'both'):
            validate_training_setup(rows, None, True)
        rows.append(dict(rows[0], label=0, query_sha256='z'))
        with self.assertRaisesRegex(ValueError, 'train-only'):
            validate_training_setup(rows, None, False)
        validate_training_setup(rows, None, True)
        with self.assertRaisesRegex(ValueError, 'group'):
            validate_training_setup(rows, rows, False)

    def test_evaluation_rejects_training_images_unless_explicit(self):
        from dinoexp.local_training import check_evaluation_scope
        rows = [dict(group='a', reference_sha256='x', query_sha256='y')]
        training = dict(training_groups=['a'], training_images=['x', 'y'])
        with self.assertRaisesRegex(ValueError, 'training'):
            check_evaluation_scope(rows, training, False)
        self.assertEqual(check_evaluation_scope(rows, training, True), 'training_overlap')
        clean = [dict(group='b', reference_sha256='z', query_sha256='w')]
        self.assertEqual(check_evaluation_scope(clean, training, False), 'held_out')

    def test_train_validate_checkpoint_reload_and_portable_report(self):
        import argparse
        import json
        from pathlib import Path
        import tempfile
        import numpy as np
        from PIL import Image
        from dinoexp.local_pairs import sha256
        from dinoexp.local_training import train_head, evaluate_head
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            cache.mkdir()
            images, rows = {}, []
            rng = np.random.default_rng(12)
            for i in range(6):
                Image.new('RGB', (28, 14), (20 + i * 30, 40, 90)).save(root / f'{i}.png')
                digest = sha256(root / f'{i}.png')
                images[i] = digest
                target = cache / (digest + '.npz')
                np.savez_compressed(target, tokens=rng.normal(size=(1, 4, 8)).astype('float32'),
                                    valid=np.ones(4, dtype=bool), cls=rng.normal(size=8).astype('float32'))
            config = dict(model='test', image_size=28, layers=[12], input_dim=8,
                          backbone_sha256='b' * 64, preprocess='letterbox-imagenet-v1')
            (cache / 'cache.json').write_text(json.dumps(dict(status='complete', format_version=1,
                config=config, images={d: {'sha256': sha256(cache / (d + '.npz'))}
                                        for d in images.values()})))
            for ref, queries, group in [(0, [1, 2], 'train'), (3, [4, 5], 'val')]:
                rows = [dict(reference=f'{ref}.png', query=f'{q}.png',
                    reference_sha256=images[ref], query_sha256=images[q], label=int(j == 0),
                    group=group, comparison_group=group, view='side', sample_id=str(q))
                    for j, q in enumerate(queries)]
                (root / f'{group}.json').write_text(json.dumps(dict(format_version=1, pairs=rows)))
            args = argparse.Namespace(train_manifest=root / 'train.json', val_manifest=root / 'val.json',
                cache=cache, output=root / 'train', allow_train_only=False, epochs=2, batch_size=2,
                lr=1e-3, threads=1, seed=7, device='cpu', dim=16, heads=4, probes=2,
                reference_grid=2, blocks=1, weight_decay=0.01)
            result = train_head(args)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue((args.output / 'best.pt').is_file())
            evaluate = argparse.Namespace(manifest=root / 'val.json', data_root=root, cache=cache,
                checkpoint=args.output / 'best.pt', allow_training_data=False,
                batch_size=2, threads=1, device='cpu', output=root / 'report')
            report = evaluate_head(evaluate)
            self.assertEqual(report['scope'], 'held_out')
            self.assertTrue((evaluate.output / 'index.html').is_file())


if __name__ == '__main__':
    unittest.main()
