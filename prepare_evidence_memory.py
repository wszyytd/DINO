"""Extract frozen full-image-branch evidence; does not train the teacher."""
import argparse
from pathlib import Path
from dinoexp.evidence_data import build_evidence_cache
from dinoexp.local_features import resolve_device
import torch


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--model-repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dense-cache', type=Path)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    build_evidence_cache(a.data_root, a.teacher, a.backbone, a.model_repo, a.output,
                         resolve_device(a.device, torch), a.dense_cache, a.threads)
