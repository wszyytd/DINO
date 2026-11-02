"""Cache frozen features for full RGB +15 tiles, with verified visible-mask labels."""
import argparse
from pathlib import Path
from dinoexp.dense_cache import extract_dense_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['dinov2_vitb14','dinov2_vits14'],default='dinov2_vitb14')
    p.add_argument('--image-size',type=int,default=518);p.add_argument('--grid-size',type=int,default=32)
    p.add_argument('--tile-size',type=int,default=512);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--dtype',choices=['float32','float16'],default='float32')
    p.add_argument('--device',default='auto');p.add_argument('--model-repo',type=Path);p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    extract_dense_features(args.manifest,args.output,data_root=args.data_root,model_repo=args.model_repo,
        checkpoint=args.checkpoint,model=args.model,image_size=args.image_size,grid_size=args.grid_size,
        tile_size=args.tile_size,batch_size=args.batch_size,device=args.device,dtype=args.dtype)


if __name__=='__main__':
    main()
