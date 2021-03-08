img_size=1024
for x in 0 1 2 3 4
do 
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [14] --csv data/raw/folds/${x}/${x}_train.csv --format coco --output_path data/detection/${img_size}/coco/${x}/
    
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [] --csv data/raw/folds/${x}/${x}_val.csv --format coco --output_path data/detection/${img_size}/coco/${x}/
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [] --csv data/raw/folds/${x}/${x}_val_nms.csv --format coco --output_path data/detection/${img_size}/coco/${x}/
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [] --csv data/raw/folds/${x}/${x}_val_wbf.csv --format coco --output_path data/detection/${img_size}/coco/${x}/
    
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [14] --csv data/raw/folds/${x}/${x}_train.csv --format yolo --output_path data/detection/${img_size}/yolo/${x}/train/
    
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [] --csv data/raw/folds/${x}/${x}_val.csv --format yolo --output_path data/detection/${img_size}/yolo/${x}/val/
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [] --csv data/raw/folds/${x}/${x}_val_nms.csv --format yolo --output_path data/detection/${img_size}/yolo/${x}/val_nms/
    python scripts/convert-format/convert.py --resize ${img_size} --ignored [] --csv data/raw/folds/${x}/${x}_val_wbf.csv --format yolo --output_path data/detection/${img_size}/yolo/${x}/val_wbf/
done