## EDA
- Use ```eda.ipynb``` to analyze dataset

## Split dataset into K-FOLDs

## CSV Format:

image_id | class_name | class_id | rad_id | x_min | y_min | x_max | y_max | width | height
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
50a418190bc3fb1ef1633bf9678929b3 | No Finding | 14 | R11 | | | | | 2332 | 2580
9a5094b2563a1ef3ff50dc5c7ff71345 | Cardiomegaly | 3 | R10 | 691 | 1375 | 1653 | 1831 | 2080 | 2336

- Run below commands to split dataset into k-folds

```
python split_kfold.py --csv=train.csv --k=5
```
- **Parameters**:
    - ***--csv***:        path to csv file
    - ***--seed***:       random seed
    - ***--k***:          number of folds
