python split_kfold.py --csv=data/train.csv --out=data/train_fold.csv --k=5
python split.py --csv=data/train_fold.csv --out=data/folds
python det2clsl.py --in=data/folds