"""
COCO Mean Average Precision Evaluation
Example execution:

python evaluate.py --gt_csv=0_val.csv --pred_csv=0_predict.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser('Evaluate COCO mAP')
parser.add_argument('--gt_csv', type=str, help='path to ground truth csv file')
parser.add_argument('--pred_csv', type=str, help='path to prediction csv file')
args = parser.parse_args()


LABELS = [
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
    'No Finding'
]


def csv2json(gt_csv, pred_csv):
    """
    Convert .csv to .json
    Input: .csv file and mode ('gt' or 'pred')
    Output .json format example: (source: https://cocodataset.org/#format-results)
    [{
        "image_id": int, 
        "category_id": int, 
        "bbox": [x,y,width,height], 
        "score": float,
    }]


    image_id,class_id,x_min,y_min,x_max,y_max,width,height
    image_id,class_id,x_min,y_min,x_max,y_max,score
    """


    def make_gt_file(df, save_filename='./temp/gt.json'):
        database = df

        try:
            os.makedirs(os.path.dirname(save_filename))
        except:
            pass

        my_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        img_count = 0
        item_count = 0
        image_dict = {}
        labels = LABELS

        for label_idx, label in enumerate(labels):
            class_dict = {
                "supercategory": None,
                "id": label_idx + 1,  # Coco starts from 1
                "name": label,
            }
            my_dict["categories"].append(class_dict)

        annotations = [
            row
            for row in zip(
                database["image_id"],
                database["class_id"],
                database["x_min"],
                database["y_min"],
                database["x_max"],
                database["y_max"],
                database["width"],
                database["height"],
            )
        ]

        for row in annotations:
            image_name, class_id, xmin, ymin, xmax, ymax, width, height = row

            if image_name not in image_dict.keys():
                image_dict[image_name] = img_count
                img_count += 1
                image_id = image_dict[image_name]
                img_dict = {
                    "file_name": image_name + ".png",
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
                my_dict["images"].append(img_dict)

            ann_w = xmax - xmin
            ann_h = ymax - ymin
            image_id = image_dict[image_name]
            ann_dict = {
                "id": item_count,
                "image_id": image_id,
                "bbox": [xmin, ymin, ann_w, ann_h],
                "area": ann_w * ann_h,
                "category_id": int(class_id) + 1,  # Coco starts from 1
                "iscrowd": 0,
            }
            item_count += 1
            my_dict["annotations"].append(ann_dict)

        if os.path.isfile(save_filename):
            os.remove(save_filename)
        with open(save_filename, "w") as outfile:
            json.dump(my_dict, outfile)
        return save_filename, image_dict

    def make_pred_file(df, image_dict, save_filename='./temp/pred.json'):
        zipped = zip(df["image_id"], df["class_id"], df["x_min"], df["y_min"], df['x_max'], df["y_max"], df["score"])
        annotations = [row for row in zipped]
        results = []
        for ann in annotations:
            image_id, class_id, xmin, ymin, xmax, ymax, score = ann

            results.append({
                "image_id": int(image_dict[image_id]), 
                "category_id": class_id + 1, 
                "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], 
                "score": float(score)
                })
        if os.path.isfile(save_filename):
            os.remove(save_filename)
        with open(save_filename, "w") as outfile:
            json.dump(results, outfile)
        return save_filename

    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)
    

    # Fill class 14 with [0,0,1,1]
    gt_df = gt_df.fillna(0)
    pred_df = pred_df.fillna(0)

    gt_df.loc[gt_df['class_id'] == 14, 'x_max'] = 1
    pred_df.loc[pred_df['class_id'] == 14, 'x_max'] = 1
    gt_df.loc[gt_df['class_id'] == 14, 'y_max'] = 1
    pred_df.loc[pred_df['class_id'] == 14, 'y_max'] = 1


    # Make json file
    gt_json, image_dict = make_gt_file(gt_df)
    pred_json = make_pred_file(pred_df, image_dict)

    return gt_json, pred_json

class mAPScore:
    """
    Arguments:
        gt_csv (str):      path to ground truth csv file (in COCO format)
        pred_csv (str):    path to prediction csv file
    """
    def __init__(self, gt_csv, pred_csv):
        self.gt_csv = gt_csv
        self.pred_csv = pred_csv
        
        # Convert csv to json
        self.gt_json, self.pred_json = csv2json(self.gt_csv, self.pred_csv)
        
        self.coco_gt = COCO(self.gt_json)
        self.get_image_ids()

    def get_image_ids(self):
        self.image_ids = []
        with open(self.pred_json, 'r') as f:
            data = json.load(f)
            for item in data:
                image_id = item['image_id']
                self.image_ids.append(image_id)

    def evaluate(self):
        # load results in COCO evaluation tool
        coco_pred = self.coco_gt.loadRes(self.pred_json)

        # run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_pred, 'bbox')
        coco_eval.params.imgIds = self.image_ids

        # Run at IOU=0.4
        coco_eval.params.iouThrs = np.array([0.4])

        # Some other params for COCO eval
        # imgIds = []
        # catIds = []
        # iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        # recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        # maxDets = [1, 10, 100]
        # areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        # areaRngLbl = ['all', 'small', 'medium', 'large']
        # useCats = 1

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        return stats



if __name__ =='__main__':
    metric = mAPScore(args.gt_csv, args.pred_csv)
    metric.evaluate()
