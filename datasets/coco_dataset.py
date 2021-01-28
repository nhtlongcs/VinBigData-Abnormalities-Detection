import sys
sys.path.append('..')

import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as patches

from utils.utils import change_box_order
from augmentations.transforms import Denormalize
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import albumentations as A
import cv2

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_path, inference=False, transforms=None):

        self.root_dir = root_dir
        self.ann_path = ann_path
        self.transforms = transforms
        self.mode = 'xyxy'
        self.inference = inference

        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img, img_name = self.load_image(idx)
        img_id = self.image_ids[idx]
        annot = self.load_annotations(idx)
        box = annot[:, :4]
        label = annot[:, -1]
        
        if self.transforms:
            item = self.transforms(image=img, bboxes=box, class_labels=label)
            img = item['image']
            box = item['bboxes']
            label = item['class_labels']
        
        box = np.array([np.asarray(i) for i in box])
        label = np.array(label)
        box = change_box_order(box, order = 'xywh2xyxy')
            
        return {
            'img': img,
            'box': box,
            'label': label,
            'img_id': img_id,
            'img_name': img_name
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])   
        img_ids = [s['img_id'] for s in batch]

        if self.inference:
             return {'imgs': imgs, 'img_ids': img_ids}

        annots = [torch.cat([s['box'] , s['label'].unsqueeze(1)], dim=1) for s in batch]
        max_num_annots = max(annot.shape[0] for annot in annots)
        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1
        return {'imgs': imgs, 'labels': annot_padded, 'img_ids': img_ids}

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
        
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        return image, image_info['file_name']

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations
        
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = random.randint(0,len(self.coco.imgs))
        item = self.__getitem__(index)
        img_name = item['img_name']
        img = item['img']
        box = item['box']
        label = item['label']
        
        normalize = False
        for x in self.transforms.transforms:
            if isinstance(x, A.Normalize):
                normalize = True
                denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            img = denormalize(img = img)

        if self.mode == 'xyxy':
            box=change_box_order(box, 'xyxy2xywh')

        self.visualize(img, box, label, figsize = figsize, img_name= img_name)

    
    def visualize(self, img, boxes, labels, figsize=(15,15), img_name=None):
        """
        Visualize an image with its bouding boxes
        """
        fig,ax = plt.subplots(figsize=figsize)

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

    def count_dict(self, types = 1):
        """
        Count class frequencies
        """
        cnt_dict = {}
        if types == 1: # Object Frequencies
            for cl in self.classes.keys():
                num_objs = sum([1 for i in self.coco.anns if self.coco.anns[i]['category_id']-1 == self.classes[cl]])
                cnt_dict[cl] = num_objs
        elif types == 2:
            widths = [i['width'] for i in self.coco.anns['images']]
            heights = [i['height'] for i in self.coco.anns['images']]
            cnt_dict['height'] = {}
            cnt_dict['width'] = {}
            for i in widths:
                if i not in cnt_dict['width'].keys():
                    cnt_dict['width'][i] = 1
                else:
                    cnt_dict['width'][i] += 1

            for i in heights:
                if i not in cnt_dict['height'].keys():
                    cnt_dict['height'][i] = 1
                else:
                    cnt_dict['height'][i] += 1
        elif types == 3:
            tmp_dict = {}
            for i in self.coco.anns:
                if i['image_id'] not in tmp_dict.keys():
                    tmp_dict[i['image_id']] = 1
                else:
                    tmp_dict[i['image_id']] += 1

            for i in tmp_dict.values():
                if i not in cnt_dict.keys():
                    cnt_dict[i] = 1
                else:
                    cnt_dict[i] += 1
        return cnt_dict

    def plot(self, figsize = (8,8), types = ["freqs"]):
        """
        Plot classes distribution
        """
        ax = plt.figure(figsize = figsize)
        num_plots = len(types)
        plot_idx = 1

        if "freqs" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 1)
            plt.title("Total objects can be seen: "+ str(sum(list(cnt_dict.values()))))
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        if "size_freqs" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 2)
            plt.title("Image sizes distribution: ")
            bar1 = plt.bar(list(cnt_dict['height'].keys()), list(cnt_dict['height'].values()), color='blue')
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        if "object_freqs" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 3)
            num_objs = sum([i*j for i,j in cnt_dict.items()])
            num_imgs = sum([i for i in cnt_dict.values()])
            mean = num_objs*1.0/num_imgs
            plt.title("Total objects can be seen: "+ str(num_objs) + '\nAverage number object per image: ' + str(np.round(mean, 3)))
          
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
            
        plt.show()

    def __str__(self):
        s = "Custom Dataset for Object Detection\n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.coco.anns)) + '\n'
        s2 = "Number of classes: " + str(len(self.labels)) + '\n'
        return s + line + s1 + s2