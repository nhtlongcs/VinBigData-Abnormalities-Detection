# Evaluate Submission

## Calculate mAP
- Input: 2 .csv file, ground truth and prediction
- Output: mAP@0.4 score report


- Run below command to calculate mAP@0.4
```
    python evaluate.py --gt_csv=0_val.json --pred_csv=0_predict.csv
```
- ***Parameters***:
    - ***--gt_csv***:          path to ground truth .csv
    - ***--pred_csv***:        path to prediction .csv
    
