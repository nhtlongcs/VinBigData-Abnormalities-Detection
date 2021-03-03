import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ensemble_boxes import weighted_boxes_fusion, nms

def filter_area(boxes, confidence_score, labels, min_area=10):
    """
    Boxes in xywh format
    """

    # dimension of bounding boxes
    width = boxes[:, 2]
    height = boxes[:, 3]

    # boxes areas
    areas = width * height

    picked_index = areas >= min_area

    # Picked bounding boxes
    picked_boxes = boxes[picked_index]
    picked_score = confidence_score[picked_index]
    picked_classes = labels[picked_index]

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)

def postprocessing(outs, imgs, retransforms = None, out_format='xyxy'):
    for item in outs:
        
        boxes_out = item['bboxes']
        if len(boxes_out) == 0:
            continue
        boxes_out_xywh = change_box_order(boxes_out, order = 'xyxy2xywh')
        new_boxes = retransforms(img = imgs, box=boxes_out_xywh)['box']
        if out_format == 'xyxy':
            new_boxes = change_box_order(new_boxes, order = 'xywh2xyxy')
        item['bboxes'] = new_boxes

    return outs

def box_fusion(
    bounding_boxes, 
    confidence_score, 
    labels, 
    mode='wbf', 
    image_size=None,
    weights=None, 
    iou_threshold=0.5):
    """
    bounding boxes: 
        list of boxes of same image [[box1, box2,...],[...]] if ensemble many models
        list of boxes of single image [[box1, box2,...]] if done on one model
    """

    if image_size is not None:
        boxes = [i*1.0/image_size for i in bounding_boxes]
    else:
        boxes = bounding_boxes.copy()

    if mode == 'wbf':
        picked_boxes, picked_score, picked_classes = weighted_boxes_fusion(
            boxes, 
            confidence_score, 
            labels, 
            weights=weights, 
            iou_thr=iou_threshold, 
            conf_type='avg', #[nms|avf]
            skip_box_thr=0.0001)
    elif mode == 'nms':
        picked_boxes, picked_score, picked_classes = nms(
            boxes, 
            confidence_score, 
            labels,
            weights=weights,
            iou_thr=iou_threshold)

    if image_size is not None:
        picked_boxes = picked_boxes*image_size

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)


