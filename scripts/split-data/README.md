# Dataset splitting

## Stratified k-fold split by class existence  

### Input CSV format

The input CSV needs to have 2 columns `image_id` and `class_id`.

**Example**:

image_id | class_id | ...
--- | --- | ---
50a418190bc3fb1ef1633bf9678929b3 | 14 | ...
9a5094b2563a1ef3ff50dc5c7ff71345 | 3 | ...
9a5094b2563a1ef3ff50dc5c7ff71345 | 5 | ...

### Split and add column

We first generate a new CSV file which includes the fold indexing information.

```
python split_kfold.py --csv=train.csv --out=train_fold.csv --k=5
```
- **Parameters**:
    - ***--csv***:	path to csv file
    - ***--out***:      output filename and directory
    - ***--seed***:     random seed (default: 3698)
    - ***--k***:        number of folds (default: 5)

The resulting CSV will have an additional `fold` column which specifies the fold each sample point belongs to.

**Example**:

image_id | class_id | ... | fold
--- | --- | --- | ---
50a418190bc3fb1ef1633bf9678929b3 | 14 | ... | 0
9a5094b2563a1ef3ff50dc5c7ff71345 | 3 | ... | 1
9a5094b2563a1ef3ff50dc5c7ff71345 | 5 | ... | 1

### Split into individual files

Using the generated file, we can split into a pair of CSV files (train, val) for each fold.

```
python split.py --csv=train_fold.csv --out=.
```
- **Parameters**:
    - ***--csv***:	path to csv file
    - ***--out***:      output directory

Generate multiple CSV files with the same format as the original CSV, each is a `<fid>_train.csv` and `<fid>_val.csv` of the `<fid>`-th fold.
