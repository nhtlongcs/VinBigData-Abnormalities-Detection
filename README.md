# VinBigData-Abnormalities-Detection
## EfficientDet D7
### Train
Train: 4000 files, validation: 400 files </br>
Image: load dicom -> resize 512x512 </br>
Aug: 
1. ToGray
2. HueSaturationValue
3. MedianBlur or Blur
4. HorizontalFlip
5. VerticalFlip
6. RandomRotate90
7. Transpose
8. Cutout

wbf: iou_thr = 0.4
### Infer (0.134)
DETECTION_THRESHOLD = 0.001 </br>
[14 1 0 0 1 1 trick](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/211971) </br>
~~TTA~~
~~wbf~~
