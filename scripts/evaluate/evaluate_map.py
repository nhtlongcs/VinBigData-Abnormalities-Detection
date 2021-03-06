"""
COCO Mean Average Precision Evaluation

True Positive (TP): Predicted as positive as was correct
False Positive (FP): Predicted as positive but was incorrect
False Negative (FN): Failed to predict an object that was there

if IOU prediction >= IOU threshold, prediction is TP
if 0 < IOU prediction < IOU threshold, prediction is FP

Precision measures how accurate your predictions are. Precision = TP/(TP+FP)
Recall measures how well you find all the positives. Recal = TP/(TP+FN)

Average Precision (AP) is finding the area under the precision-recall curve.
Mean Average  Precision (MAP) is AP averaged over all categories.

AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05
AP@.75 means the AP with IoU=0.75

*Under the COCO context, there is no difference between AP and mAP

Example execution:

python evaluate.py --gt_csv=0_val.csv --pred_csv=0_predict.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser("Evaluate COCO mAP")
parser.add_argument("--gt_csv", type=str, help="path to ground truth csv file")
parser.add_argument("--pred_csv", type=str, help="path to prediction csv file")
args = parser.parse_args()


LABELS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No Finding",
]

class mAPScore:
    """
    Arguments:
        gt_df (pd.DataFrame):       ground truth dataframe, format: [image_id,class_id,x_min,y_min,x_max,y_max,width,height]
        pred_df (pd.DataFrame):     prediction dataframe, format: [image_id,class_id,x_min,y_min,x_max,y_max,score]
        is_normalized:              True if the predictions are normalized

    *** image_id in pred_df must be included in gt_df's image_id ***


    Example usage:

        pred_df = pd.read_csv(args.pred_csv)
        gt_df = pd.read_csv(args.gt_csv)

        metric = mAPScore(gt_df, is_normalized=True)
        for i in range(NUM_EPOCHS):
            pred_df = ...
            metric.update_pred(pred_df)
            metric.evaluate()

    """

    def __init__(self, gt_df, is_normalized=True):

        self.gt_df = gt_df
        self.gt_json = "./temp/gt.json"
        self.pred_json = "./temp/pred.json"
        self.is_normalized = is_normalized
        
        self.gt_df = self.process_df(self.gt_df)
        if self.is_normalized:
            self.tmp_df = self.gt_df[["image_id", "width", "height"]]

        # Convert csv to json
        self.make_gt_json_file()
        self.coco_gt = COCO(self.gt_json)
        self.get_image_ids()

    def process_df(self, _df):
        # Fill class 14 with [0,0,1,1]

        df = _df.fillna(0)

        df.loc[df["class_id"] == 14, "x_max"] = 1
        df.loc[df["class_id"] == 14, "y_max"] = 1
        return df

    def make_gt_json_file(self):
        database = self.gt_df

        try:
            os.makedirs(os.path.dirname(self.gt_json))
        except:
            pass

        my_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        img_count = 0
        item_count = 0
        self.image_dict = {}
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

            if image_name not in self.image_dict.keys():
                self.image_dict[image_name] = img_count
                img_count += 1
                image_id = self.image_dict[image_name]
                img_dict = {
                    "file_name": image_name + ".png",
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
                my_dict["images"].append(img_dict)

            ann_w = xmax - xmin
            ann_h = ymax - ymin
            image_id = self.image_dict[image_name]
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

        if os.path.isfile(self.gt_json):
            os.remove(self.gt_json)
        with open(self.gt_json, "w") as outfile:
            json.dump(my_dict, outfile)

    def make_pred_json_file(self):

        """
           Output .json format example: (source: https://cocodataset.org/#format-results)
            [{
                "image_id": int, 
                "category_id": int, 
                "bbox": [x,y,width,height], 
                "score": float,
            }]
        """

        zipped = zip(
            self.pred_df["image_id"],
            self.pred_df["class_id"],
            self.pred_df["x_min"],
            self.pred_df["y_min"],
            self.pred_df["x_max"],
            self.pred_df["y_max"],
            self.pred_df["score"],
        )
        annotations = [row for row in zipped]
        results = []
        for ann in tqdm(annotations):
            image_id, class_id, xmin, ymin, xmax, ymax, score = ann
            if self.is_normalized:
                image_df = self.tmp_df.loc[self.tmp_df["image_id"] == image_id]

                width = int(image_df["width"].iloc[0])
                height = int(image_df["height"].iloc[0])
            else:
                width = 1
                height = 1

            results.append(
                {
                    "image_id": int(self.image_dict[image_id]),
                    "category_id": class_id + 1,
                    "bbox": [
                        int(xmin * width),
                        int(ymin * height),
                        int((xmax - xmin) * width),
                        int((ymax - ymin) * height),
                    ],
                    "score": float(score),
                }
            )
        if os.path.isfile(self.pred_json):
            os.remove(self.pred_json)
        with open(self.pred_json, "w") as outfile:
            json.dump(results, outfile)

    def get_image_ids(self):
        self.image_ids = list(self.image_dict.values())

    def update_pred(self, pred_df):
        self.pred_df = self.process_df(pred_df)
        self.make_pred_json_file()
        

    def evaluate(self):
        # load results in COCO evaluation tool
        coco_pred = self.coco_gt.loadRes(self.pred_json)

        # run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_pred, "bbox")
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


if __name__ == "__main__":
    pred_df = pd.read_csv(args.pred_csv)
    gt_df = pd.read_csv(args.gt_csv)

    metric = mAPScore(gt_df, is_normalized=True)
    metric.update_pred(pred_df)
    metric.evaluate()
