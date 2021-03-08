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

"""

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .metrictemplate import TemplateMetric
from utils.utils import change_box_order
from utils.postprocess import postprocessing

def _eval(coco_gt, image_ids, pred_json_path, **kwargs):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.iouThrs = np.array([0.4])
    # Some params for COCO eval
    #imgIds = []
    #catIds = []
    #iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    #maxDets = [1, 10, 100]
    #areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    #areaRngLbl = ['all', 'small', 'medium', 'large']
    #useCats = 1

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return stats

class mAPScores(TemplateMetric):
    def __init__(
            self,
            dataset, 
            max_images = 10000,
            retransforms=None,
            min_conf = 0.3, 
            min_iou = 0.3, 
            tta = False,
            decimals = 4):

        self.coco_gt = COCO(dataset.ann_path)
        self.dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=16, 
                num_workers=4, 
                pin_memory = True,
                drop_last= True,
                shuffle=True, 
                collate_fn=dataset.collate_fn) # requires batch size = 1

        self.retransforms = retransforms
        self.tta = tta
        self.min_conf = min_conf
        self.min_iou = min_iou
        self.decimals = decimals
        self.max_images = max_images
        self.filepath = f'results/bbox_results.json'
        self.image_ids = []
        self.reset()

        if not os.path.exists('results'):
            os.mkdir('results')
            
    def reset(self):
        self.model = None
        self.image_ids = []

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute(self):
        results = []
        empty_imgs = 0
        with torch.no_grad():

            with tqdm(total=min(len(self.dataloader), self.max_images)) as pbar:
                for idx, batch in enumerate(self.dataloader):
                    if idx > self.max_images:
                        break
                    
                    if self.tta is not None:
                        preds = self.tta.make_tta_predictions(self.model, batch)
                    else:
                        preds = self.model.inference_step(batch, self.min_conf, self.min_iou)
                    
                    if self.retransforms is not None:
                            preds = postprocessing(preds, batch['imgs'].cpu(), self.retransforms, out_format='xywh')

                    for i in range(len(preds)):
                        image_id = batch['img_ids'][i]
                        self.image_ids.append(image_id)

                        pred = preds[i]

                        if self.retransforms is not None:
                            bbox_xywh = pred['bboxes']
                        else:
                            bbox_xywh = pred['bboxes']
                        
                        if bbox_xywh is None or len(bbox_xywh) == 0:
                            empty_imgs += 1
                        else:
                            bbox_xywh = change_box_order(bbox_xywh, order='xyxy2xywh')
                            cls_ids = pred['classes']
                            cls_conf = pred['scores']
                            for i in range(bbox_xywh.shape[0]):
                                score = float(cls_conf[i])
                                label = int(cls_ids[i])
                                box = bbox_xywh[i, :]
                                image_result = {
                                    'image_id': image_id,
                                    'category_id': label,
                                    'score': float(score),
                                    'bbox': box.tolist(),
                                }

                                results.append(image_result)
                    pbar.update(1)
                    pbar.set_description(f'Empty images: {empty_imgs}')

        if not len(results):
            return False

        # write output
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        json.dump(results, open(self.filepath, 'w'), indent=4)
        return True

    def value(self):
        result = self.compute()
        if result:
            stats = _eval(self.coco_gt, self.image_ids, self.filepath)
            return {
                "MAP" : np.round(float(stats[0]),self.decimals),
                "MAPsmall" : np.round(float(stats[3]),self.decimals),
                "MAPmedium" : np.round(float(stats[4]),self.decimals),
                "MAPlarge" : np.round(float(stats[5]),self.decimals),}
        else:
            return {
                "MAP" : 0.0,
                "MAPsmall" : 0.0,
                "MAPmedium" : 0.0,
                "MAPlarge" : 0.0,}

    def __str__(self):
        return f'Mean Average Precision: {self.value()}'

    def __len__(self):
        return len(self.dataloader)