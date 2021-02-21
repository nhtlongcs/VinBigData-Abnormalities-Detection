"""
COCO Mean Average Precision Evaluation
Example execution:

python evaluate.py --gt_json=wbf512_noratio_fold0_val.json --pred_json=bbox_results.json
"""

import os
import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser('Evaluate COCO mAP')
parser.add_argument('--gt_json', type=str, help='path to ground truth json file')
parser.add_argument('--pred_json', type=str, help='path to prediction json file')
args = parser.parse_args()

class mAPScore:
    """
    Arguments:
        gt_json (str):      path to ground truth json file (in COCO format)
        pred_json (str):    path to prediction json file
                            Format example:
                            [
                                {
                                    "image_id": 0,
                                    "category_id": 1,
                                    "score": 0.894,
                                    "bbox": [0, 0, 1, 1] #(x_topleft, y_topleft, width, height)
                                },
                                ...
                            ]

    """
    def __init__(self, gt_json, pred_json):
        self.gt_json = gt_json
        self.pred_json = pred_json
        self.coco_gt = COCO(gt_json)
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
    metric = mAPScore(args.gt_json, args.pred_json)
    metric.evaluate()
