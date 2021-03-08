# Merge overlapping boxes
- Input: .csv need to remove overlapping boxes, class-index mapping .csv
- Output: .csv after being processed
- Use below command to merge overlapping boxes

```
python merge.py --csv_in=train.csv --type=wbf
```
- ***Parameters***:
    - ***--csv_in***:          path to csv file
    - ***--csv_out***:         path to ouptut csv file
    - ***--skip_threshold***:   skip box whose confidence lower than threshold (for WBF only)
    - ***--iou_threshold***:   iou threshold to remove overlappings
    - ***--type***:            box fusion method, support ['nms', 'wbf']
    - ***--class_mapping***:   class name csv file
    - ***--ignored_classes***: string of list of ignored classes. Example: '[14, 15]'
