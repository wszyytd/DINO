from pathlib import Path
import tempfile
import unittest

from dinoexp.local_paths import resolve_backbone_paths


class LocalPathTests(unittest.TestCase):
    def fixtures(self, tmp):
        project = Path(tmp) / 'DINO'
        project.mkdir()
        hub = Path(tmp) / 'cache/torch/hub'
        repo = hub / 'facebookresearch_dinov2_main'
        repo.mkdir(parents=True)
        (repo / 'hubconf.py').write_text('# fixture')
        weights = hub / 'checkpoints/dinov2_vits14_pretrain.pth'
        weights.parent.mkdir()
        weights.write_bytes(b'fixture')
        return project, hub, repo, weights

    def test_discovers_existing_hub_cache_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, hub, repo, weights = self.fixtures(tmp)
            result = resolve_backbone_paths(None, None, 'dinov2_vits14', project,
                                            hub_dir=hub, environ={})
            self.assertEqual(result, (repo.resolve(), weights.resolve()))
            self.assertFalse((project / 'out').exists())

    def test_pasted_environment_placeholders_do_not_override_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, hub, repo, weights = self.fixtures(tmp)
            result = resolve_backbone_paths(None, None, 'dinov2_vits14', project,
                hub_dir=hub, environ={'DINO_REPO': '/实际路径/dinov2_repo',
                                     'DINO_BACKBONE': '/实际路径/weights.pth'})
            self.assertEqual(result, (repo.resolve(), weights.resolve()))

    def test_explicit_invalid_path_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, hub, _, weights = self.fixtures(tmp)
            with self.assertRaisesRegex(ValueError, 'hubconf.py'):
                resolve_backbone_paths(project / 'missing', weights, 'dinov2_vits14',
                                       project, hub_dir=hub, environ={})

    def test_small_backbone_is_not_selected_for_base_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, hub, _, _ = self.fixtures(tmp)
            with self.assertRaisesRegex(ValueError, 'dinov2_vitb14_pretrain.pth'):
                resolve_backbone_paths(None, None, 'dinov2_vitb14', project,
                                       hub_dir=hub, environ={})

    def test_explicit_paths_take_precedence_over_stale_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, hub, repo, weights = self.fixtures(tmp)
            self.assertEqual(resolve_backbone_paths(repo, weights, 'dinov2_vits14', project,
                hub_dir=hub, environ={'DINO_REPO': '/missing', 'DINO_BACKBONE': '/missing'}),
                (repo.resolve(), weights.resolve()))


if __name__ == '__main__':
    unittest.main()
