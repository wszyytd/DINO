import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from dinoexp.evidence import build_cards, apply_reviews, fuse_cards, review_template
from evidence import write_report, score_cards
from dinoexp.scene import with_resolution


def fixture(root):
    root.mkdir()
    Image.new('RGB', (40, 20), 'white').save(root / 'source.png')
    Image.new('RGB', (15, 12), 'white').save(root / 'door.png')
    Image.new('L', (40, 20), 255).save(root / 'mask.png')
    record = dict(status='detected', class_name='right_front-door', confidence=.8,
                  source='source.png', overlay='source.png', crop='door.png',
                  masked_crop='door.png', mask='mask.png', native_width=40,
                  native_height=20, crop_width=15, crop_height=12,
                  bbox_xyxy=[3, 2, 18, 14], mask_pixels=180)
    samples = []
    for sid, identity, role, seq in [('reference', 'A', 'reference', -1),
                                     ('A_0', 'A', 'query', 0),
                                     ('B_0', 'B', 'query', 0)]:
        samples.append(dict(scene='scene', sample_id=sid, identity=identity,
                            role=role, capture_metadata=dict(sequence=str(seq),
                            view='side', frame=str(seq)), records=[copy.deepcopy(record)]))
    (root / 'parts.json').write_text(json.dumps(samples))
    (root / 'parts_run.json').write_text(json.dumps(dict(status='complete')))
    return samples


def review(card, status):
    item = review_template([card])['reviews'][0]
    item.update(status=status, reviewer='human', basis='Checked corresponding visible door.')
    item['checks'] = {key: True for key in item['checks']}
    return item


class EvidenceTests(unittest.TestCase):
    def test_diagnostic_cosine_uses_corresponding_crops_without_changing_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            samples = fixture(root)
            Image.new('RGB', (15, 12), 'black').save(root / 'other.png')
            samples[2]['records'][0]['crop'] = 'other.png'
            (root / 'parts.json').write_text(json.dumps(samples))
            cards = build_cards(root, ['right_front-door'])
            features, paths = score_cards(root, cards,
                lambda images: np.array([[1., 0.], [0., 1.]]))
            self.assertEqual(paths, ['door.png', 'other.png'])
            self.assertEqual([c['cosine'] for c in cards], [1., 0.])
            self.assertTrue(all(c['status'] == 'unknown' for c in cards))
            self.assertEqual(features.shape, (2, 2))

    def test_no_detection_and_duplicates_are_unknown_not_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            samples = fixture(root)
            samples[1]['records'][0]['class_name'] = 'left_front-door'
            samples[2]['records'] *= 2
            (root / 'parts.json').write_text(json.dumps(samples))
            cards = build_cards(root, ['right_front-door'])
            self.assertEqual([c['availability'] for c in cards],
                             ['query_missing', 'query_ambiguous'])
            self.assertTrue(all(c['status'] == 'unknown' for c in cards))
            self.assertTrue(all(c['cosine'] is None for c in cards))
            self.assertTrue(all(c['decision'] == 'retain' for c in fuse_cards(cards)))

    def test_unique_prediction_and_low_score_do_not_establish_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            cards = build_cards(root, ['right_front-door'])
            cards[0]['cosine'] = -.5
            self.assertEqual(cards[0]['availability'], 'needs_review')
            self.assertEqual(cards[0]['status'], 'unknown')
            self.assertEqual(fuse_cards(cards)[0]['decision'], 'retain')

    def test_missing_reference_and_incomplete_run_are_rejected_or_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            cards = build_cards(root, ['right_back-door'])
            self.assertEqual(cards[0]['availability'], 'reference_missing')
            (root / 'parts_run.json').write_text('{"status":"incomplete"}')
            with self.assertRaisesRegex(ValueError, 'complete'):
                build_cards(root, ['right_front-door'])

    def test_review_requires_checks_and_is_bound_to_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            cards = build_cards(root, ['right_front-door'])
            item = review(cards[0], 'contradictory')
            item['checks']['same_part_side'] = False
            with self.assertRaisesRegex(ValueError, 'checks'):
                apply_reviews(cards, {'reviews': [item]})
            item['checks']['same_part_side'] = True
            item['fingerprint'] = 'old-run'
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                apply_reviews(cards, {'reviews': [item]})
            item['fingerprint'] = cards[0]['fingerprint']
            reviewed = apply_reviews(cards, {'reviews': [item]})
            self.assertEqual(reviewed[0]['status'], 'contradictory')
            self.assertEqual(cards[0]['status'], 'unknown')
            self.assertEqual(fuse_cards(reviewed)[0]['decision'], 'retain')

    def test_review_cannot_promote_missing_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            cards = build_cards(root, ['right_back-door'])
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                apply_reviews(cards, {'reviews': [review(cards[0], 'contradictory')]})

    def test_repeated_frames_do_not_vote_and_unknown_does_not_erase_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            first = build_cards(root, ['right_front-door'])[0]
            first = apply_reviews([first], {'reviews': [review(first, 'compatible')]})[0]
            later = copy.deepcopy(first)
            later.update(card_id='later', status='unknown', sequence=2)
            states = fuse_cards([first, first, later])
            self.assertEqual(states[0]['parts']['right_front-door']['status'], 'compatible')
            self.assertEqual(states[0]['parts']['right_front-door']['observation_groups'], 1)
            other = copy.deepcopy(first)
            other.update(card_id='other', status='contradictory', sequence=3)
            states = fuse_cards([first, other])
            self.assertEqual(states[0]['parts']['right_front-door']['status'], 'conflict')
            self.assertEqual(states[0]['decision'], 'retain')

    def test_scene_ids_do_not_merge_and_similarities_do_not_cancel_contradiction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            first = build_cards(root, ['right_front-door'])[0]
            first = apply_reviews([first], {'reviews': [review(first, 'contradictory')]})[0]
            other = copy.deepcopy(first)
            other.update(card_id='window', part='right_front-window', status='compatible')
            swapped = copy.deepcopy(first)
            swapped.update(scene='swapped', card_id='swapped', status='compatible')
            states = fuse_cards([first, other, swapped])
            self.assertEqual(len(states), 2)
            self.assertEqual(states[0]['state'], 'contradiction_recorded')
            self.assertTrue(all(s['decision'] == 'retain' for s in states))

    def test_sources_and_reviews_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            samples = fixture(root)
            cards = build_cards(root, ['right_front-door'])
            old = review(cards[0], 'compatible')
            Image.new('RGB', (15, 12), 'black').save(root / 'door.png')
            changed = build_cards(root, ['right_front-door'])
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                apply_reviews(changed, {'reviews': [old]})
            samples[1]['records'][0]['source'] = '../escape.png'
            (root / 'parts.json').write_text(json.dumps(samples))
            with self.assertRaisesRegex(ValueError, 'escapes'):
                build_cards(root, ['right_front-door'])

    def test_portable_report_and_cli_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parts'
            fixture(root)
            output = Path(tmp) / 'report'
            result = subprocess.run([sys.executable, 'evidence.py', '--parts-run', str(root),
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = json.loads((output / 'evidence_run.json').read_text())
            self.assertEqual(run['status'], 'complete')
            page = (output / 'index.html').read_text(encoding='utf-8')
            links = re.findall(r'(?:src|href)="([^"]+)"', page)
            self.assertTrue(all((output / link).is_file() for link in links))
            cards = json.loads((output / 'evidence.json').read_text())
            self.assertTrue(all(c['status'] == 'unknown' for c in cards))
            with self.assertRaises(FileExistsError):
                write_report(root, output, cards, {})


class ResolutionTests(unittest.TestCase):
    def test_resolution_changes_only_dimensions_and_rejects_incomplete_pair(self):
        cfg = dict(width=1024, height=768, fov=70, views=[dict(distance=24)])
        high = with_resolution(cfg, 2048, 1536)
        self.assertEqual((high['width'], high['height']), (2048, 1536))
        self.assertEqual(high['views'], cfg['views'])
        self.assertEqual(cfg['width'], 1024)
        for w, h in [(2048, None), (0, 0), (2048, 768)]:
            with self.assertRaises(ValueError):
                with_resolution(cfg, w, h)


if __name__ == '__main__':
    unittest.main()
