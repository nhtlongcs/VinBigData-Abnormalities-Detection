from utils.getter import *
import argparse
import os
import cv2
import matplotlib.pyplot as plt 
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import box_nms_numpy, draw_boxes_v2, change_box_order
import pandas as pd

def main(args, config):
    if args.submission:
        test_df =  pd.read_csv('/content/main/datasets/test_info.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    val_transforms = get_augmentation(config.augmentations, _type = 'val')
    idx_classes = {idx:i for idx,i in enumerate(config.obj_list)}
    NUM_CLASSES = len(config.obj_list)
    net = EfficientDetBackbone(num_classes=NUM_CLASSES, compound_coef=args.c,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))

    model = Detector(
                    n_classes=NUM_CLASSES,
                    model = net,
                    criterion= FocalLoss(), 
                    optimizer= torch.optim.Adam,
                    optim_params = {'lr': 0.1},     
                    device = device)
    model.eval()
    if args.weight is not None:                
        load_checkpoint(model, args.weight)

    if args.images is not None or args.image is not None:
        if args.images is not None:
            args.images, output_dir = args.images.split(':')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            paths = os.listdir(args.images)
            paths = [os.path.join(args.images, i) for i in paths]
        elif args.image is not None:
            args.image, output_name = args.image.split(':')
            paths = [args.image]
        with tqdm(total=len(paths)) as pbar:
            batch_size = 4
            empty_imgs = 0
            results = []
            batch = []
            temp = []
            output_names = []
            output_imgs = []
            for img_idx, img_path in enumerate(paths):
                image_id = os.path.basename(img_path)[:-4]
                output_names.append(os.path.join(output_dir, image_id+'.png'))
                pil_img = cv2.imread(img_path)
                pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB).astype(np.float32)
                pil_img /= 255.0
                pil_img = cv2.resize(pil_img, tuple(config.image_size))
                temp.append(0)
                outputs = []
                output_imgs.append(pil_img)
                if ((img_idx+1) % batch_size == 0) or img_idx==len(paths)-1:
                    
                    inputs = torch.stack([val_transforms(image=img, class_labels=temp)['image'] for img in output_imgs]).to(device)
                    batch = {'imgs': inputs}
                    with torch.no_grad():
                      preds = model.inference_step(batch, args.min_conf, args.min_iou)

                    del inputs
                    torch.cuda.empty_cache()
                    batch=[]
                    for idx, outputs in enumerate(preds):
                        boxes = outputs['bboxes'] 
                        labels = outputs['classes']  
                        scores = outputs['scores']
                        if len(boxes) == 0:
                            empty_imgs += 1
                            boxes = None
                        else:
                            boxes = change_box_order(boxes, order='xyxy2xywh')
                            boxes, scores, labels = box_nms_numpy(boxes, scores, labels, threshold=0.4, box_format='xywh')
                        if boxes is not None:
                            draw_boxes_v2(output_names[idx], output_imgs[idx], boxes, labels, scores, idx_classes)


                        if args.submission:
                            image_id = os.path.basename(output_names[idx])[:-4]

                            row = test_df.loc[data["image_id"] == str(image_id)]
                            img_w, img_h = row.w, row.h

                            if boxes is not None:
                                pred_strs = []
                                for box, score, cls_id in zip(boxes, scores, labels):
                                    x,y,w,h = box
                                    x = round(float(x*1.0*img_w/config.image_size[0]))
                                    y = round(float(y*1.0*img_h/config.image_size[1]))
                                    w = round(float(w*1.0*img_w/config.image_size[0]))
                                    h = round(float(h*1.0*img_h/config.image_size[1]))
                                    pred_strs.append(f'{cls_id} {score} {x} {y} {w} {h}')
                                pred_str = ' '.join(pred_strs)
                            else:
                                pred_str = '14 1 0 0 1 1'
                            

                            results.append([image_id, pred_str])
    
                pbar.update(batch_size)
                pbar.set_description(f'Empty images: {empty_imgs}')
            
            if args.submission:
                submission_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
                submission_df.to_csv('results/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--image', type=str, default=None,  help='path to image')
    parser.add_argument('--images', type=str, default=None,  help='path to image')
    parser.add_argument('--min_conf', type=float, default= 0.3, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.4, help='minimum iou threshold for non max suppression')
    parser.add_argument('-c', type=int, default = 2, help='version of EfficentDet')
    parser.add_argument('--weight', type=str, default = 'weights/efficientdet-d2.pth',help='version of EfficentDet')
    parser.add_argument('--output_path', type=str, default = None, help='name of output to .avi file')
    parser.add_argument('--submission', action='store_true', default = False, help='output to submission file')
    parser.add_argument('--config', type=str, default = None,help='save detection at')

    args = parser.parse_args() 
    config = Config(os.path.join('configs',args.config+'.yaml'))                   
    main(args, config)
    