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

def visualize(self, img, boxes, labels, figsize=(15,15), img_name=None):
  """
  Visualize an image with its bouding boxes
  """
  fig,ax = plt.subplots(figsize=figsize)

  if self.mode == 'xyxy':
      boxes=change_box_order(boxes, 'xyxy2xywh')

  if isinstance(img, torch.Tensor):
      img = img.numpy().squeeze().transpose((1,2,0))
  # Display the image
  ax.imshow(img)

  # Create a Rectangle patch
  for box, label in zip(boxes, labels):
      color = np.random.rand(3,)
      x,y,w,h = box
      rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor = color,facecolor='none')
      plt.text(x, y-3, self.labels[label], color = color, fontsize=20)
      # Add the patch to the Axes
      ax.add_patch(rect)

  if img_name is not None:
      plt.title(img_name)
  plt.show()


def main(args, config):

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
            batch_size = 2
            empty_imgs = 0
            results = []
            batch = []
            temp = []
            for img_idx, img_path in enumerate(paths):
                image_id = os.path.basename(img_path)[:-4]
                pil_img = cv2.imread(img_path)
                pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB).astype(np.float32)
                pil_img /= 255.0
                pil_img = cv2.resize(pil_img, tuple(config.image_size))
                temp.append(0)
                outputs = []
                batch.append(pil_img)
                if ((img_idx+1) % batch_size == 0) or img_idx==len(paths)-1:
                    
                    inputs = torch.stack([val_transforms(image=img, class_labels=temp)['image'] for img in batch])
                    batch = {'imgs': inputs.to(device)}
                    outputs = model.inference_step(batch, args.min_conf, args.min_iou)
                    try:
                        boxes = np.concatenate([i['bboxes'] for i in outputs if len(i['bboxes'])>0]) 
                        labels = np.concatenate([i['classes'] for i in outputs if len(i['bboxes'])>0])    
                        scores = np.concatenate([i['scores'] for i in outputs if len(i['bboxes'])>0])
                        boxes, scores, labels = box_nms_numpy(boxes, scores, labels, threshold=0.001, box_format='xywh')
                        boxes = change_box_order
                    except:
                      boxes = None

                    if boxes is not None:
                      draw_boxes_v2(output_name, pil_img, boxes, labels, scores, idx_classes)
                    
            #     try:
            #         boxes = np.concatenate([i['bboxes'] for i in outputs if len(i['bboxes'])>0]) 
            #         labels = np.concatenate([i['classes'] for i in outputs if len(i['bboxes'])>0])    
            #         scores = np.concatenate([i['scores'] for i in outputs if len(i['bboxes'])>0])
            #         boxes, scores, labels = box_nms_numpy(boxes, scores, labels, threshold=0.001, box_format='xywh')
            #     except:
            #         boxes = None
                
            #     if boxes is None:
            #         empty_imgs += 1        
            #     else:
            #         for i in range(boxes.shape[0]):
            #             score = float(scores[i])
            #             label = int(labels[i])
            #             box = boxes[i, :]
            #             image_result = {
            #                 'image_id': image_id,
            #                 'category_id': label + 1,
            #                 'score': float(score),
            #                 'bbox': box.tolist(),
            #             }

            #             results.append(image_result)

            #         image_name = os.path.basename(img_path)
            #         if args.images is not None:
            #             draw_boxes_v2(os.path.join(output_dir, image_name), pil_img, boxes, labels, scores, idx_classes)
            #         elif args.image is not None:
            #             draw_boxes_v2(output_name, pil_img, boxes, labels, scores, idx_classes)
                 
            #     pbar.update(1)
            #     pbar.set_description(f'Empty images: {empty_imgs}')
            
            # if args.submission:
            #     filepath = os.path.join(output_dir, f'{config.project_name}_submission.json')
            #     if os.path.exists(filepath):
            #         os.remove(filepath)
            #     json.dump(results, open(filepath, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--image', type=str, default=None,  help='path to image')
    parser.add_argument('--images', type=str, default=None,  help='path to image')
    parser.add_argument('--min_conf', type=float, default= 0.15, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.3, help='minimum iou threshold for non max suppression')
    parser.add_argument('-c', type=int, default = 2, help='version of EfficentDet')
    parser.add_argument('--weight', type=str, default = 'weights/efficientdet-d2.pth',help='version of EfficentDet')
    parser.add_argument('--output_path', type=str, default = None, help='name of output to .avi file')
    parser.add_argument('--submission', action='store_true', default = False, help='output to submission file')
    parser.add_argument('--config', type=str, default = None,help='save detection at')

    args = parser.parse_args() 
    config = Config(os.path.join('configs',args.config+'.yaml'))                   
    main(args, config)
    