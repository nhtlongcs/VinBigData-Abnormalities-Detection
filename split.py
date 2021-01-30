import pandas as pd
import numpy as np

root = 'data'
df = pd.read_csv(root + '/' + 'train_fold.csv')

for fid in df['fold'].unique():
    df[df['fold'] == fid].to_csv(f'{root}/{fid}_val.csv', index=False)
    df[df['fold'] != fid].to_csv(f'{root}/{fid}_train.csv', index=False)

