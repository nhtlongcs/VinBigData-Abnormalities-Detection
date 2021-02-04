# Efficientdet Abnormalities Detection

## Notebooks: [link](https://colab.research.google.com/drive/1CZ4q_14QlfM9EcFya6Q_PPidszT7_AmC?usp=sharing)

## Dataset Structure:

```
  this repo
  │   train.py
  │   eval.py
  │   eval_coco.py
  │   ...
  │
  └───configs
  │      vin512.yaml                      # Your dataset configs
  │      configs.py
  │      ...
  │
  └───datasets  
  │   └───vinai512                        # Dataset folder
  │       └───images
  │           └───train
  │           │   ***.png
  │           └───test
  │           │   ***.png
  │       └───annotations
  │           |  vinai512_train.json
  │           |  vinai512_val.json
  ...
```
  
## Setting configs (Edit in *.yaml file):

Example for above folder structure:
```
  settings:
    project_name: "vinai512"  # also the folder name of the dataset that under datasets folder
    train_imgs: images/train  
    val_imgs: images/train
    test_imgs: images/test
    train_anns: annotations/vinai512_train.json
    val_anns: annotations/vinai512_val.json

    # this is coco anchors, change it if necessary
    anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

    # must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'Atelectasis' here is 1, while category_id of is 2
    obj_list: [
      'Aortic enlargement', 
      'Atelectasis', 
      'Calcification',
      'Cardiomegaly',
      'Consolidation',
      'ILD',
      'Infiltration',
      'Lung Opacity',
      'Nodule/Mass',
      'Other lesion',
      'Pleural effusion',
      'Pleural thickening',
      'Pneumothorax',
      'Pulmonary fibrosis',
    ]

    # augmentations settings
    
    image_size: [512,512]
    keep_ratio: False         # If True, use Resize + Padding. Else Resize normally

    cutmix: False             # If True, use Cutmix Augmentation
    mixup: False              # If True, use Mixup Augmentation
```

## Start training
- Run command
```
python train.py -c=<version number of EfficientDet> --config=<path to project config yaml file>
```
- **Extra Parameters**:
    - ***--resume***:       path to checkpoint to resume training
    - ***--lr***:           learning rate
    - ***--batch_size***:   batch size, recommend 4 - 8
    - ***--head_only***:    if train only the head, freeze others
    - ***--num_epochs***:   number of epochs
    - ***--saved_path***:   path to save weight
    - ***--val_interval***: validate per number of epochs
    - ***--save_interval***: save per number of iterations
    - ***--log_path***:     tensorboard logging path
    
- To log training process, use Tensorboard:
```
%reload_ext tensorboard
%tensorboard --logdir='./loggers/runs/vinai512'
```

## Inference on testset:

- This requires test_info.csv inside datasets folder

- test_info.csv format:

image_id | width | height 
--- | --- | --- 
002a34c58c5b758217ed1f584ccbcfe9 | 2345 | 2584
... | ... | ... 

- Run command
```
python eval.py -c=<version number of EfficientDet> --config=<path to project config yaml file>
```
- **Extra Parameters**:
    - ***--weight***:       path to model checkpoint
    - ***--min_conf***:     minimum confidence for an object to be detect
    - ***--min_iou***:      minimum iou threshold for non max suppression
    - ***--output_path***:  path to folder to save visualization, if None, no visualization to be made
    - ***--submission***:   flag, for output submission.csv file
    
## Result:   

version | datasets | configs | val mAP@0.5| LB score 
--- | --- | --- | --- | --- 
efficiendet-d7 | nms-512-noratio + khong class "No Finding" | use Cutmix + Mixup, min-conf=0.3 | 0.186 | 0.113
... | ... | ... | ... | ... 
