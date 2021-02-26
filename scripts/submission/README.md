# Convert output format to submission format

- Input: model output file, test info file, expected filename, number of decimal places to round, flags: use_trick, keep_ratio
- Output: Submission file

```
!python output-to-submission.py --df='output.csv' --test_info='test_info.csv' --filename='submission.csv' --round_to 3 --use_trick --keep_ratio
```

- Parameters:
    - ***df***: path to model output csv file
    - ***test_info***: path to test info csv file (to get image_id, width, height, class14prob)
    - ***filename***: output csv file name
    - ***round_to***: number of decimal places to round confidence values to
    - ***use_trick: add*** class 14 to all images
    - ***keep_ratio***: undo central pad
