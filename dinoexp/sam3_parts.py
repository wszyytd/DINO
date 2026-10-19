"""Independent SAM 3 text prompts produce proposals, never verified part identities."""

import json
import math

import numpy as np


def load_prompts(path):
    prompts = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('prompt file must be a nonempty list')
    labels = []
    for prompt in prompts:
        if (not isinstance(prompt, dict) or not isinstance(prompt.get('label'), str)
                or not isinstance(prompt.get('text'), str)
                or not prompt['label'].strip() or not prompt['text'].strip()):
            raise ValueError('each prompt needs nonempty label and text')
        labels.append(prompt['label'])
    if len(labels) != len(set(labels)):
        raise ValueError('prompt labels must be unique')
    return prompts


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'is_floating_point') and value.is_floating_point():
        value = value.float()  # NumPy does not directly support torch.bfloat16 scores.
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def decode_prompt(result, label, native_size):
    width, height = native_size
    masks = array(result['masks'])
    scores = array(result['scores'])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(f'SAM 3 masks must be native Nx{height}x{width}; got {masks.shape}')
    if scores.ndim != 1 or len(scores) != len(masks):
        raise ValueError('SAM 3 mask/score count mismatch')
    if not np.isfinite(masks).all():
        raise ValueError('SAM 3 masks contain nonfinite values')
    detections = []
    for mask, score in zip(masks, scores, strict=True):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('SAM 3 score must be finite and in [0,1]')
        detections.append(dict(class_name=label, confidence=score, mask=mask.copy()))
    return detections


class Sam3PartAdapter:
    task = 'segment'
    backend = 'sam3'

    def __init__(self, processor, prompts):
        self.processor = processor
        self.prompts = prompts
        self.names = {i: prompt['label'] for i, prompt in enumerate(prompts)}
        self.prompt_texts = {prompt['label']: prompt['text'] for prompt in prompts}

    def predict_parts(self, image):
        state = self.processor.set_image(image)
        predictions = []
        for prompt in self.prompts:
            print(f"  [sam3 prompt] {prompt['text']}", flush=True)
            self.processor.reset_all_prompts(state)
            result = self.processor.set_text_prompt(prompt=prompt['text'], state=state)
            predictions.extend(decode_prompt(result, prompt['label'], image.size))
        # Preserve all masks and cross-prompt overlaps. No appearance-based part assignment.
        return predictions
