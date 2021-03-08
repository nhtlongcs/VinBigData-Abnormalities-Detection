for x in 0 1 2 3 4
do 
    python scripts/merge-bboxes/merge.py --type=wbf --csv_in data/raw/folds/${x}/${x}_val.csv --class_mapping data/meta/class_mapping.csv
    python scripts/merge-bboxes/merge.py --type=nms --csv_in data/raw/folds/${x}/${x}_val.csv --class_mapping data/meta/class_mapping.csv
done