# **VinBigData-Abnormalities-Detection**

*Model: CenterNet*

Repo gốc: https://github.com/xingyizhou/CenterNet

Link dataset và annotations:\
    - **512px** : [dataset512]() [annotations512]() \
    - **1024px**: [dataset1024]() [annotations1024]() 

## **Installation**

Please refer to [INSTALL.md](centernet/readme/INSTALL.md) for installation instructions.

## **Pretrained Checkpoints**

| Model | size | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>40</sub> | Dataset | Annotations || something | something |
|---------- |------ |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    |640 |48.1     |48.1     |66.4     |3.8ms     |264     ||47.0M  |115.4
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    |640 |**50.1** |**50.1** |**68.7** |6.0ms     |167     ||87.7M  |218.8
| | | | | | | || |

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
  │   └───centernet     
  │         ...
  │   └───data                       
  │       └───images
  │              ***.png
  |
  │       └───annotations
  │              train.json
  │              val.json
  ...
```
### **train.json**
> contain train image paths, ex:
```
...
```
### **val.json**
> contain train image paths, ex:
```
...
```
## **Environments**

- **Google Colab Notebook** 

    <a href="https://colab.research.google.com/drive/1ptcmwvV1eDwDoJWCBPSvoSUOImgjNcLI?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## **Inference**

To run inference on example images in $TEST_DIR`:
```bash
$ python demo.py ctdet \
--demo $TEST_DIR \
--load_model $MODEL_PATH
```

To use this CenterNet in src code, you can 

~~~
import sys
CENTERNET_PATH = /path/to/CenterNet/src/lib/
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = /path/to/model
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

img = image/or/path/to/your/image/
ret = detector.run(img)['results']
~~~
`ret` will be a python dict: `{category_id : [[x1, y1, x2, y2, score], ...], }`
## **Training**

Run commands below to to something
```bash
$ !python main.py ctdet --metric MAP --val_intervals 1 --exp_id $exp_id --batch_size $BATCH_SIZE --lr 1.25e-4  --gpus 0 --num_epochs $NUM_EPOCHS  --load_model $PRETRAIN_PATH #optional
```


