import argparse
import os
import pandas as pd
import numpy as np

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--csv',
                    type=str,
                    default='../data/train_fold.csv',
                    help='path to csv file')
parser.add_argument('--out',
                    type=str,
                    default='../data/folds/',
                    help='output directory')
args = parser.parse_args()

df = pd.read_csv(args.csv)

if not os.path.exists(args.out):
    os.mkdir(args.out)

for fid in df['fold'].unique():
    df[df['fold'] == fid].to_csv(f'{args.out}/{fid}_val.csv', index=False)
    df[df['fold'] != fid].to_csv(f'{args.out}/{fid}_train.csv', index=False)
