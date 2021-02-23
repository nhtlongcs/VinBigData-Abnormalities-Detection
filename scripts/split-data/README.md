# Dataset splitting

## Stratified k-fold split for Detection by class existence

### Input CSV format

The input CSV needs to have 2 columns `image_id` and `class_id`.

**Example**:

| image_id                         | class_id | ... |
| -------------------------------- | -------- | --- |
| 50a418190bc3fb1ef1633bf9678929b3 | 14       | ... |
| 9a5094b2563a1ef3ff50dc5c7ff71345 | 3        | ... |
| 9a5094b2563a1ef3ff50dc5c7ff71345 | 5        | ... |

### Split and add column

We first generate a new CSV file which includes the fold indexing information.

```
python split_kfold.py --csv=data/train.csv --out=data/train_fold.csv --k=5
```

- **Parameters**:
  - **_--csv_**: path to csv file
  - **_--out_**: output filename and directory
  - **_--seed_**: random seed (default: 3698)
  - **_--k_**: number of folds (default: 5)

The resulting CSV will have an additional `fold` column which specifies the fold each sample point belongs to.

**Example**:

| image_id                         | class_id | ... | fold |
| -------------------------------- | -------- | --- | ---- |
| 50a418190bc3fb1ef1633bf9678929b3 | 14       | ... | 0    |
| 9a5094b2563a1ef3ff50dc5c7ff71345 | 3        | ... | 1    |
| 9a5094b2563a1ef3ff50dc5c7ff71345 | 5        | ... | 1    |

### Split into individual files

Using the generated file, we can split into a pair of CSV files (train, val) for each fold.

```
python split.py --csv=data/train_fold.csv --out=data/folds
```

- **Parameters**:
  - **_--csv_**: path to csv file
  - **_--out_**: output directory

The output directory will be like this.

```
<out>/
├─ <fold_id>/
│  ├─ <fold_id>_<phase>.csv
```

where:

- **`<out>`**: output directory name
- **`<fold_id>`**: fold identifier
- **`<phase>`**: train or val

## Stratified k-fold split for Classification by class existence

Assuming we have the same CSVs and directory format from the step above, we can generate the split for the Classification task.

```
python det2clsl.py --in=data/folds
```

- **Parameters**:
  - **_--in_**: input directory

The output directory will have additional files for the classification task.

```
<out>/
├─ <fold_id>/
│  ├─ <fold_id>_<phase>.csv
│  ├─ <fold_id>_<phase>_cls.csv // <-- new files
```

The format of the classification CSV files:

```
Format: image_id, class_[0-14]
where:
  image_id: image identifier
  class_[0-13]: whether the image contains region of that the disease
  class_14: 1 if there is no disease (all class_[0-13] = 0),
            0 otherwise
```

Example:

| image_id                         | class_0 | class_1 | class_2 | class_3 | class_4 | class_5 | class_6 | class_7 | class_8 | class_9 | class_10 | class_11 | class_12 | class_13 | class_14 |
| -------------------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 50a418190bc3fb1ef1633bf9678929b3 | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0        | 0        | 0        | 0        | 1        |
| 21a10246a5ec7af151081d0cd6d65dc9 | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0        | 0        | 0        | 0        | 1        |
| 9a5094b2563a1ef3ff50dc5c7ff71345 | 1       | 0       | 0       | 1       | 0       | 0       | 0       | 0       | 0       | 0       | 1        | 1        | 0        | 0        | 0        |
