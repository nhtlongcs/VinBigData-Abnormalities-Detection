# Convert output format to submission format

Input: model output file, test info file, expected filename, keep ratio (trick) </br>
Output: Submission file

```
!python output-to-submission.py --df='output.csv' --test_info='test_info.csv' --filename='submission.csv' --trick=True --keep_ratio=False
```
