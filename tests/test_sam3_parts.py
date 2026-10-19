import json
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from dinoexp.sam3_parts import Sam3PartAdapter, decode_prompt, load_prompts
from dinoexp.parts import discover_scenes
from segment_parts import run_inference
from compare_parts import compare_runs
from test_parts import dataset


class ProcessorFixture:
    """External SAM API boundary fixture, not a model or experiment result."""
    def __init__(self):
        self.encoded = 0
        self.reset = 0
        self.prompts = []

    def set_image(self, image):
        self.encoded += 1
        return {'width': image.width, 'height': image.height}

    def reset_all_prompts(self, state):
        self.reset += 1
        state.pop('previous', None)

    def set_text_prompt(self, prompt, state):
        if 'previous' in state:
            raise AssertionError('prompts contaminated each other')
        state['previous'] = prompt
        self.prompts.append(prompt)
        mask = np.zeros((1, 1, state['height'], state['width']), bool)
        mask[0, 0, 10:20, 30:50] = True
        return dict(masks=mask, scores=np.array([.8]))


class Sam3PartsTests(unittest.TestCase):
    def test_native_masks_and_scores_are_not_reshaped_or_relabelled(self):
        mask = np.ones((2, 1, 20, 40), bool)
        results = decode_prompt(dict(masks=mask, scores=np.array([.7, .8])),
                                'front_door', (40, 20))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['class_name'], 'front_door')
        self.assertEqual(results[0]['mask'].shape, (20, 40))
        with self.assertRaisesRegex(ValueError, 'native'):
            decode_prompt(dict(masks=np.ones((1, 1, 1008, 1008)), scores=np.array([.8])),
                          'front_door', (40, 20))
        with self.assertRaisesRegex(ValueError, 'score'):
            decode_prompt(dict(masks=mask, scores=np.array([float('nan'), .8])),
                          'front_door', (40, 20))
        empty = decode_prompt(dict(masks=np.zeros((0, 1, 20, 40)), scores=np.zeros(0)),
                              'car_door', (40, 20))
        self.assertEqual(empty, [])

    def test_image_is_encoded_once_and_each_prompt_is_reset(self):
        prompts = [{'label': 'car_door', 'text': 'car door'},
                   {'label': 'front_door', 'text': 'front car door'}]
        processor = ProcessorFixture()
        adapter = Sam3PartAdapter(processor, prompts)
        result = adapter.predict_parts(Image.new('RGB', (80, 40)))
        self.assertEqual(processor.encoded, 1)
        self.assertEqual(processor.reset, 2)
        self.assertEqual(processor.prompts, ['car door', 'front car door'])
        self.assertEqual(len(result), 2)
        self.assertEqual(adapter.names, {0: 'car_door', 1: 'front_door'})

    def test_prompt_file_requires_unique_nonempty_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.json'
            path.write_text('[{"label":"car_door","text":"car door"},'
                            '{"label":"car_door","text":"door"}]')
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_prompts(path)

    def test_adapter_runs_existing_native_export_and_pair_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root / 'data')
            weights = root / 'fixture.pt'
            weights.write_bytes(b'API boundary fixture, never actual model weights')
            prompts = [{'label': 'car_door', 'text': 'car door'},
                       {'label': 'side_window', 'text': 'side window'},
                       {'label': 'roof', 'text': 'car roof'}]
            args = SimpleNamespace(output=root / 'one', data=root / 'data', weights=weights,
                                   parts=['all'], imgsz=1008, conf=.5, iou=None, device='cpu')
            model = Sam3PartAdapter(ProcessorFixture(), prompts)
            run_inference(args, discover_scenes(args.data), model, weights, {'fixture': True})
            run = json.loads((args.output / 'parts_run.json').read_text())
            self.assertEqual(run['backend'], 'sam3')
            self.assertEqual(run['prompts'], prompts)
            self.assertEqual(run['status'], 'complete')
            shutil.copytree(args.output, root / 'baseline')
            compare_runs(root / 'baseline', args.output, root / 'comparison')
            report = json.loads((root / 'comparison/comparison.json').read_text())
            self.assertEqual(report['samples'], 2)
            self.assertIsNone(report['accuracy'])
            # A different source image cannot be presented as a controlled model comparison.
            records = json.loads((args.output / 'parts.json').read_text())
            records[0]['records'][0]['source_sha256'] = 'changed'
            (args.output / 'parts.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'source'):
                compare_runs(root / 'baseline', args.output, root / 'bad')

    def test_dry_run_needs_no_sam3_torch_or_weight_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root / 'data')
            result = subprocess.run([sys.executable, 'segment_sam3.py', '--data',
                                     str(root / 'data'), '--output', str(root / 'out'),
                                     '--dry-run'], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('car door', result.stdout)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
