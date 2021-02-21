import argparse

import pandas as pd
import numpy as np

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--csv',
                    type=str,
                    help='path to csv file')
parser.add_argument('--out',
                    type=str,
                    help='output directory')
args = parser.parse_args()

df = pd.read_csv(args.csv)

for fid in df['fold'].unique():
    df[df['fold'] == fid].to_csv(f'{args.out}/{fid}_val.csv', index=False)
    df[df['fold'] != fid].to_csv(f'{args.out}/{fid}_train.csv', index=False)
