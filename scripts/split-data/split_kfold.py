import os
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import csv
import argparse
import pandas as pd

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=3698, type=int)
parser.add_argument('--csv', default='data/train.csv', type=str)
parser.add_argument('--k', default=5, type=int)
args = parser.parse_args()

# Set randomization seed
np.random.seed(args.seed)

# Read CSV
df = pd.read_csv(args.csv)

# Get appearing classes for each image
data = dict()
for image_id, _,class_id, *_ in df.values.tolist():
    data.setdefault(image_id, [])
    data[image_id].append(class_id)
data = [[k, v] for k, v in data.items()]

# Transform into numpy arrays
X, y = map(np.array, zip(*data))

# Get unique classes
original_tokens = sum(y, [])
original_cnt = dict(Counter(original_tokens))

# Transform into binary vectors (1=class appears)
unique_tokens = sorted(list(set(original_tokens)))
mlb = MultiLabelBinarizer(classes=unique_tokens)
y_bin = mlb.fit_transform(y)

# k-fold split
df['fold'] = np.nan

cnt_df = pd.DataFrame()
cnt_df['class_id'] = original_cnt.keys()
cnt_df['total'] = original_cnt.values()
cnt_df = cnt_df.sort_values(by='class_id')

X_indices = np.array(range(len(X))).reshape(-1, 1)
k_fold = IterativeStratification(n_splits=args.k, order=1)

for i, (train_indices, test_indices) in enumerate(k_fold.split(X_indices, y_bin)):
    # Get train-val splits
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Assign fold
    for x in X_test:
        df['fold'][df['image_id'] == x] = i
    cnt_df[f'fold_{i}'] = df.groupby('fold')['class_id'].value_counts()[i].sort_index().values
df['fold'] = df['fold'].astype(int)
df.to_csv('train_fold.csv', index=False)
cnt_df.to_csv('train_fold_cnt.csv', index=False)