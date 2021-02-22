# Convert output format to submission format

- Input: model output file, test info file, expected filename, keep ratio (trick)
- Output: Submission file

```
!python output-to-submission.py --df='output.csv' --test_info='test_info.csv' --filename='submission.csv' --trick=True --keep_ratio=False
```

- Parameters:
    - ***df***: path to model output csv file
    - ***test_info***: path to test info csv file (to get image_id, width, height, class14prob)
    - ***filename***: output csv file name
    - ***trick: add*** class 14 to all images
    - ***keep_ratio***: Your output is keep ratio
