for x in 0 1 2 3 4
do 
    python scripts/convert-format/convert.py --resize 1024 --ignored [14] --csv data/raw/folds/${x}/${x}_train.csv --format coco --output_path data/detection/coco/${x}/
    python scripts/convert-format/convert.py --resize 1024 --ignored [] --csv data/raw/folds/${x}/${x}_val.csv --format coco --output_path data/detection/coco/${x}/
    # python scripts/convert-format/convert.py --resize 1024 --ignored [14] --csv data/raw/folds/${x}/${x}_train.csv --format yolo --output_path data/detection/yolo/${x}/train/
    # python scripts/convert-format/convert.py --resize 1024 --ignored [] --csv data/raw/folds/${x}/${x}_val.csv --format yolo --output_path data/detection/yolo/${x}/val/
done