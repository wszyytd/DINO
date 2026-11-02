"""Explicitly fetch official DINOv2 backbone weights; no teacher task weights."""
import argparse
from pathlib import Path
import urllib.request
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['dinov2_vitb14','dinov2_vits14'],default='dinov2_vitb14')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    target=(args.output or Path('models')/(args.model+'_pretrain.pth')).resolve()
    expected=768 if args.model=='dinov2_vitb14' else 384
    def check(path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if tuple(state['patch_embed.proj.weight'].shape)!=(expected,3,14,14):
            raise ValueError('downloaded backbone shape is incompatible')
    if target.exists():
        check(target);print('Existing official backbone:',target);return
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(target.suffix+'.part')
    url=f'https://dl.fbaipublicfiles.com/dinov2/{args.model}/{args.model}_pretrain.pth'
    print('Downloading official backbone:',url,flush=True)
    urllib.request.urlretrieve(url,partial)
    check(partial);partial.replace(target)
    print('Saved:',target)


if __name__=='__main__':
    main()
