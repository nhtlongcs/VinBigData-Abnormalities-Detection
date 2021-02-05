# VinBigData-Abnormalities-Detection

Model: Yolov5

Repo gốc: https://github.com/ultralytics/yolov5

Link dataset và annotations:\
    - **512px** : [dataset512]() [annotations512]() \
    - **1024px**: [dataset1024]() [annotations1024]() 

## **Requirements**

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

## **Pretrained Checkpoints**

| Model | size | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>40</sub> | Speed<sub>V100</sub> | FPS<sub>V100</sub> || params | GFLOPS |
|---------- |------ |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases)    |640 |36.8     |36.8     |55.6     |**2.2ms** |**455** ||7.3M   |17.0
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases)    |640 |44.5     |44.5     |63.1     |2.9ms     |345     ||21.4M  |51.3
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    |640 |48.1     |48.1     |66.4     |3.8ms     |264     ||47.0M  |115.4
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    |640 |**50.1** |**50.1** |**68.7** |6.0ms     |167     ||87.7M  |218.8
| | | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases) + TTA |832 |**51.9** |**51.9** |**69.6** |24.9ms |40      ||87.7M  |1005.3

<!--- 
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |640 |49.0     |49.0     |67.4     |4.1ms     |244     ||77.2M  |117.7
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |1280 |53.0     |53.0     |70.8     |12.3ms     |81     ||77.2M  |117.7
--->


## **Organize Directories**

```
  workspace
  └───this repo
  │   └───notebooks   
  │         ...
  │   └───yolov5     
  │         ...
  |
  │   config.yaml 
  │   train.txt 
  │   val.txt 
  |
  └───inputs                        # Dataset folder
  │   └───data                       
  │       └───images
  │           └───train
  │              ***.png
  │           └───val
  │              ***.png
  |
  │       └───labels
  │           └───train
  │              ***.txt
  │           └───val
  │              ***.txt
  ...
```
## **Setting configs (Edit in \*.yaml file):**
> setting config for yolo model
```
names:
- Aortic enlargement
- Atelectasis
- Calcification
- Cardiomegaly
- Consolidation
- ILD
- Infiltration
- Lung Opacity
- Nodule/Mass
- Other lesion
- Pleural effusion
- Pleural thickening
- Pneumothorax
- Pulmonary fibrosis
nc: 14
train: <path to train.txt>  # ex: /content/train.txt
val: <path to val.txt>      # ex: /content/val.txt
```
### **train.txt**
> contain train image paths, ex:
```
/content/inputs/data/images/train/0.png
/content/inputs/data/images/train/1.png
...
```
### **val.txt**
> contain val image paths, ex:
```
/content/inputs/data/images/val/0.png
/content/inputs/data/images/val/1.png
...
```
## **Environments**

- **Google Colab Notebook** 

    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## **Inference**

detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
To run inference on example images in `data/images`:
```bash
$ python detect.py --weights 'best.pt' \
--img 512 \
--conf 0.0 \
--iou 0.4 \
--save-txt \
--save-conf \
--source /content/inputs/test \
--exist-ok
```
## **Training**

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ !python train.py --save-dir $SAVE_DIR --batch $BATCH_SIZE --weights yolov5x.pt --data config.yaml --epochs 50 --cache --img 640 --name $EXP_NAME
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">


