import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

class Denormalize(object):
    """
    Denormalize image and boxes for visualization
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], **kwargs):
        self.mean = mean
        self.std = std
        
    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        """
        :param img: (tensor) image to be denormalized
        :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
        """
        mean = np.array(self.mean)
        std = np.array(self.std)
        img_show = img.numpy().squeeze().transpose((1,2,0))
        img_show = (img_show * std+mean)
        img_show = np.clip(img_show,0,1)
        return img_show

def getResize(config):
    if not config.keep_ratio:
        return A.Resize(
            height = config.image_size[1],
            width = config.image_size[0]
        )
    else:
        return A.LongestMaxSize(max_size=max(config.image_size)), 
        A.PadIfNeeded(min_height=config.image_size[1], min_width=config.image_size[0], p=1.0, border_mode=cv2.BORDER_CONSTANT),
        

def get_augmentation(config, _type='train'):
    train_transforms = A.Compose([
        A.ToGray(),
        getResize(config),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.GaussianBlur(),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(p=0.3),            
        ], p=0.3),
        A.Normalize(mean=config.mean, std=config.std),
        ToTensor()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


    val_transforms = A.Compose([
        A.ToGray(),
        getResize(config),
        A.Normalize(mean=config.mean, std=config.std),
        ToTensor()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    

    return train_transforms if _type == 'train' else val_transforms