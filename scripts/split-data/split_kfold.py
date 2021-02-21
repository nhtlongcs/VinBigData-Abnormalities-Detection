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
parser = argparse.ArgumentParser(
    description='Split a given data (in CSV format) into folds, add a column to indicate the fold each sample belongs to.'
)
parser.add_argument('--seed', 
                    type=int, 
                    default=3698,
                    help='random seed (default: 3698)')
parser.add_argument('--csv',
                    type=str,
                    help='path to csv file')
parser.add_argument('--k', 
                    type=int,
                    default=5,
                    help='number of folds (default: 5)')
parser.add_argument('--out',
                    type=str,
                    help='output filename and directory')
args = parser.parse_args()

# Set randomization seed
np.random.seed(args.seed)

# Read CSV
df = pd.read_csv(args.csv)

# Get appearing classes for each image
data = dict()
df_obj_class = df[['image_id', 'class_id']]
for image_id, class_id in df_obj_class.values:
    data.setdefault(image_id, [])
    data[image_id].append(class_id)
data = [[k, v] for k, v in data.items()]

# Transform into numpy arrays
X, y = zip(*data)
X = np.array(X)

# Get unique classes
original_tokens = sum(y, [])
original_cnt = dict(Counter(original_tokens))

# Transform into binary vectors (1=class appears)
unique_tokens = sorted(list(set(original_tokens)))
mlb = MultiLabelBinarizer(classes=unique_tokens)
y_bin = mlb.fit_transform(y)

# k-fold split
X_indices = np.array(range(len(X))).reshape(-1, 1)
k_fold = IterativeStratification(n_splits=args.k, order=1)
k_fold_split = k_fold.split(X_indices, y_bin)

# Assign fold to each id
id2fold = dict()
for i, (_, val_indices) in enumerate(k_fold_split):
    for x in X[val_indices]:
        id2fold[x] = i

# Add new 'fold' column
df['fold'] = [id2fold[im_id] for im_id in df['image_id']]
df['fold'] = df['fold'].astype(int)

# Save to csv
df.to_csv(args.out, index=False)
