"""Train the spatial encoder, not the previous crop BCE classifier."""
import argparse
from pathlib import Path
from dinoexp.dense_training import train_dense


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data-root',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--stage',choices=['simulation','real'],default='simulation')
    p.add_argument('--epochs',type=int);p.add_argument('--lr',type=float)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--dim',type=int,default=256)
    p.add_argument('--heads',type=int,default=8);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--gamma',type=float,default=2.)
    p.add_argument('--null-weight',type=float,default=1.);p.add_argument('--device',default='auto')
    args=p.parse_args()
    args.epochs=args.epochs if args.epochs is not None else (5 if args.stage=='simulation' else 60)
    args.lr=args.lr if args.lr is not None else (2e-4 if args.stage=='simulation' else 5e-6)
    train_dense(args)


if __name__=='__main__':
    main()
