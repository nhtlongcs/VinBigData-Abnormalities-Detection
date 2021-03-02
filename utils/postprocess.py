import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ensemble_boxes import *

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

def box_nms(boxes, scores, threshold=0.5):
    """
    Non Maximum Suppression
    Use custom (very slow) or torchvision non-maximum supression on bounding boxes
    
    :param bboxes: (tensor) bounding boxes, size [N, 4]
    :param scores: (tensor) bbox scores, sized [N]
    :return: keep: (tensor) selected box's indices
    """

    # Torchvision NMS:
    keep = torchvision.ops.boxes.nms(boxes, scores,threshold)
    return keep

def box_nms_numpy(bounding_boxes, confidence_score, labels, threshold=0.2, box_format='xyxy'):
    """
    Non Maximum Suppression
    Use custom (very slow) or torchvision non-maximum supression on bounding boxes
    
    :param bboxes: (tensor) bounding boxes, size [N, 4]
    :param scores: (tensor) bbox scores, sized [N]
    :return: keep: (tensor) selected box's indices
    """

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    if box_format == 'xywh':
        end_x += boxes[:, 0]
        end_y += boxes[:, 1]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_classes = []
    
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_classes.append(labels[index])
        
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)

def box_wbf(bounding_boxes, confidence_score, labels, image_size=None,weights=None, iou_threshold=0.5, conf_threshold=0.01):
    """
    bounding boxes: list of boxes of same image [[box1, box2,...],[...]]
    """

    num_augs = len(bounding_boxes)

    if image_size is not None:
        boxes = [i*1.0/image_size for i in bounding_boxes]
    else:
        boxes = bounding_boxes.copy()

    picked_boxes, picked_score, picked_classes = weighted_boxes_fusion(
        boxes, 
        confidence_score, 
        labels, 
        weights=weights, 
        iou_thr=iou_threshold, 
        skip_box_thr=conf_threshold)

    if image_size is not None:
        picked_boxes = picked_boxes*image_size

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)


