"""Export held-out original-image search heatmaps and Null metrics."""
import argparse
from pathlib import Path
from dinoexp.dense_training import evaluate_dense


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','cache','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=4)
    evaluate_dense(p.parse_args())


if __name__=='__main__':
    main()
