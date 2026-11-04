from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_evidence_experiment import SmallBank, small_episode
from dinoexp.evidence_data import episode_hashes
from dinoexp.evidence_model import EvidenceModel
from dinoexp.evidence_training import train_memory
from run_evidence_pilot import parser, run


class EvidenceCLITests(unittest.TestCase):
    def test_reuse_upstream_without_loading_evaluation_features_or_running_policies(self):
        torch.set_num_threads(1)
        bank = SmallBank()
        episodes = []
        for role, position in [('repr', '42'), ('value', '64'), ('eval', '105')]:
            e = small_episode(role)
            for digest, item in list(bank.items.items()):
                if ':' not in digest:
                    bank.items[f'{role}:{digest}'] = item
            def rename(o, role=role):
                return replace(o, image_hash=f'{role}:{o.image_hash}' if o.image_hash else None)
            episodes.append(replace(e, position=position, group=f'position:{position}',
                reference_hash=f'{role}:{e.reference_hash}',
                initial=replace(e.initial, histories=tuple(tuple(rename(o) for o in h)
                                                          for h in e.initial.histories)),
                outcomes={k:rename(o) for k,o in e.outcomes.items()}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            (source/'run.json').write_text(json.dumps({'status':'complete'}))
            model = EvidenceModel(input_dim=8, dim=16, layers=1, heads=4,
                                  memory_tokens=3, blocks=1)
            train_memory(model, bank, episodes[:1], source/'alpha_0/memory', steps=2, alpha=0)
            checkpoint = source/'alpha_0/memory/last.pt'
            before = checkpoint.read_bytes()
            args = parser().parse_args(['--data','unused','--cache','unused',
                '--output',str(root/'new'),'--memory-run',str(source),
                '--skip-evaluation','--allow-pilot-location-split','--alphas','0',
                '--states-per-episode','2','--value-steps','2','--threads','1'])
            def feature_bank(path, digests, grid):
                self.assertEqual(digests, episode_hashes(episodes[:2]))
                return bank
            with patch('run_evidence_pilot.load_corpus', return_value=(episodes,
                       {'skipped':[], 'scope':'test'})), \
                    patch('run_evidence_pilot.FeatureBank', side_effect=feature_bank), \
                    patch('run_evidence_pilot._policies', side_effect=AssertionError('no evaluation')):
                result = run(args)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['policies'], {})
            self.assertEqual(result['value_diagnostics'], {})
            # Fully exhausted sampled states have no remaining actions and are skipped.
            self.assertGreaterEqual(result['training_value_diagnostics']['alpha_0']['states'], 1)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual((root/'new/alpha_0/memory/last.pt').read_bytes(), before)
            self.assertIn('relative_softmax_v2', (root/'new/report.md').read_text(encoding='utf-8'))
            (source/'run.json').write_text(json.dumps(
                {'status':'complete', 'audit':{'cache_manifest_sha256':'old'}}))
            with patch('run_evidence_pilot.load_corpus', return_value=(episodes,
                       {'skipped':[], 'scope':'test', 'cache_manifest_sha256':'different'})):
                with self.assertRaisesRegex(ValueError, 'original feature cache'):
                    run(args)


if __name__ == '__main__':
    unittest.main()
