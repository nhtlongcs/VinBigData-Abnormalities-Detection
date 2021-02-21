# Evaluate Submission

## Calculate mAP
- Input: 2 .json file, ground truth and prediction
- Output: mAP@0.4 score report
- Prediction .json file format example:
```
[
    {
        "image_id": 0,
        "category_id": 1,
        "score": 0.894,
        "bbox": [0, 0, 1, 1] #(x_topleft, y_topleft, width, height)
    },
    ...
]
```

- Run below command to calculate mAP@0.4
```
    python evaluate_map.py --gt_json=1024_noratio_fold0_val.json --pred_json=bbox_results.json
```
- ***Parameters***:
    - ***--gt_json***:          path to ground truth .json
    - ***--pred_json***:        path to prediction .json
    
