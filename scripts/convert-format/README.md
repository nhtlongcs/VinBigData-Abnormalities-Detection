# Convert dataset to COCO/YOLO format
- Input: .csv file
- Output: .json file (COCO format) or .txt file (YOLO format)
- Use below command to convert to other format
```
python convert.py --csv_in=train.csv --format=coco
```
- ***Parameters***:
    - ***--csv_in***:          path to .csv file or folder to kfold .csv files
    - ***--format***:          convert to coco/yolo format
    - ***--resize****:         resize boxes to fit image size (for COCO only)     
    - ***--keep_ratio***:      whether to keep the original aspect ratio (for COCO only)
    - ***--output_path***:     output path to store annotations
