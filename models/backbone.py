# Author: Zylo117

import torch
import torchvision
import numpy as np
from torch import nn

from .effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from .effdet.efficientdet import HeadNet
from .frcnn import create_fasterrcnn_fpn
# from .yolov4 import Yolo_loss, Yolov4

def get_model(args, config, device):
    NUM_CLASSES = len(config.obj_list)

    if config.pretrained_backbone is None:
        load_weights = True
    else:
        load_weights = False

    net = None

    if config.model_name.startswith('efficientdet'):
        compound_coef = config.model_name.split('-')[1]
        assert compound_coef in [f'd{i}' for i in range(8)], f"efficientdet version {i} is not supported"
        net = EfficientDetBackbone(
            num_classes=NUM_CLASSES, 
            compound_coef=compound_coef, 
            load_weights=load_weights, 
            freeze_backbone = getattr(args, 'freeze_backbone', False),
            pretrained_backbone_path=config.pretrained_backbone,
            freeze_batchnorm=args.freeze_bn,
            image_size=config.image_size)
    elif config.model_name.startswith('fasterrcnn'):
        backbone_name = config.model_name.split('-')[1]
        net = FRCNNBackbone(
            backbone_name=backbone_name,
            num_classes=NUM_CLASSES, 
            pretrained=True,
            image_size=config.image_size)
    elif config.model_name.startswith('yolo'):
        backbone_name = config.model_name
        net = Yolov4Backbone(
            batch_size=config.batch_size,
            device=device,
            num_classes=NUM_CLASSES, 
            image_size=config.image_size, 
            pretrained_backbone_path=config.pretrained_backbone)

    return net

class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseBackbone, self).__init__()
        pass
    def forward(self, batch):
        pass
    def detect(self, batch):
        pass

class EfficientDetBackbone(BaseBackbone):
    def __init__(
        self, 
        num_classes=80, 
        compound_coef='d0', 
        load_weights=False, 
        image_size=[512,512], 
        pretrained_backbone_path=None, 
        freeze_backbone=False, 
        freeze_batchnorm = False,
        **kwargs):

        super(EfficientDetBackbone, self).__init__(**kwargs)
        self.name = f'efficientdet-{compound_coef}'
        config = get_efficientdet_config(f'tf_efficientdet_{compound_coef}')
        config.image_size = image_size
        config.norm_kwargs=dict(eps=.001, momentum=.01)

        net = EfficientDet(
            config, 
            pretrained_backbone=load_weights, 
            freeze_backbone=freeze_backbone,
            pretrained_backbone_path=pretrained_backbone_path)
        
        if freeze_batchnorm:
            print("freeze batchnorm")
            freeze_bn(net.backbone)
            freeze_bn(net.fpn)

        net.reset_head(num_classes=num_classes)
        net.class_net = HeadNet(config, num_outputs=config.num_classes)

        self.model = DetBenchTrain(net, config)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        targets = batch['targets']
        targets_res = {}

        inputs = inputs.to(device)
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]

        targets_res['boxes'] = boxes
        targets_res['labels'] = labels

        return self.model(inputs, targets_res)

    def detect(self, batch, device):
        inputs = batch["imgs"].to(device)
        img_sizes = batch['img_sizes'].to(device)
        img_scales = batch['img_scales'].to(device)

        outputs = self.model(inputs, inference=True, img_sizes=img_sizes, img_scales=img_scales)
        outputs = outputs.cpu().numpy()
        out = []
        for i, output in enumerate(outputs):
            boxes = output[:, :4]
            labels = output[:, -1]
            scores = output[:,-2]

            if len(boxes) > 0:
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out
    
class FRCNNBackbone(BaseBackbone):
    def __init__(
        self, 
        backbone_name,
        num_classes=80, 
        image_size=[512,512], 
        pretrained=False,
        **kwargs):

        super(FRCNNBackbone, self).__init__(**kwargs)
        self.name = f'fasterrcnn-{backbone_name}'
        self.model = create_fasterrcnn_fpn(backbone_name,pretrained=pretrained, num_classes=num_classes, min_size = image_size[0], max_size = image_size[1])
        self.num_classes = num_classes

    def forward(self, batch, device):
        self.model.train()

        # Tensor of images
        inputs = batch["imgs"]

        # Unwrap tensor into list of images
        inputs = [i for i in inputs]
        
        targets = batch['targets']

        inputs = list(image.to(device) for image in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(inputs, targets)
        loss_dict = {k:v.mean() for k,v in loss_dict.items()}
        loss = sum(loss for loss in loss_dict.values())
        ret_loss_dict = {
            'T': loss,
            'C': loss_dict['loss_classifier'],
            'B': loss_dict['loss_box_reg'],
            'O': loss_dict['loss_objectness'],
            'RPN': loss_dict['loss_rpn_box_reg'],     
        }
        return ret_loss_dict

    def detect(self, batch, device):
        inputs = batch["imgs"]
        inputs = list(image.to(device) for image in inputs)
        outputs =  self.model(inputs)
        out = []
        for i, output in enumerate(outputs):
            boxes = output['boxes'].data.cpu().numpy()
            labels = output['labels'].data.cpu().numpy()
            scores = output['scores'].data.cpu().numpy()
            if len(boxes) > 0:
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out

class Yolov4Backbone(BaseBackbone):
    def __init__(
        self, 
        batch_size,
        device,
        num_classes=80, 
        image_size=[512,512], 
        pretrained_backbone_path=None, 
        **kwargs):

        super(Yolov4Backbone, self).__init__(**kwargs)
        self.name = 'yolov4'
        self.model = Yolov4(
            yolov4conv137weight=pretrained_backbone_path,
            n_classes=num_classes
        )

        self.model = nn.DataParallel(self.model)
        self.model.module.switch_mode('train')

        self.loss_fn = Yolo_loss(
            device=device, 
            batch_size=batch_size, 
            n_classes=num_classes, 
            image_size=image_size[0])

        self.num_classes = num_classes

    def forward(self, batch, device):
        self.model.module.switch_mode('train')

        inputs = batch["imgs"]
        targets = batch['targets']

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = self.model(inputs)
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = self.loss_fn(outputs, targets)
        ret_loss_dict = {
            'T': loss,
            'OBJ': loss_obj,
            'CLS': loss_cls,
            'L2': loss_l2 
        }
        return ret_loss_dict

    def detect(self, batch, device):
        self.model.module.switch_mode('inference')

        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs =  self.model(inputs)
        out = []
        for i, (outboxes, outconfs) in enumerate(zip(outputs[0], outputs[1])):
            # print(outboxes)
            # print(outconfs)

            img_height, img_width = inputs.shape[-2:]
            boxes = outboxes.squeeze(2).cpu().detach().numpy()
            # boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[:,0] = boxes[:,0]*img_width
            boxes[:,1] = boxes[:,1]*img_height
            boxes[:,2] = boxes[:,2]*img_width
            boxes[:,3] = boxes[:,3]*img_height

            confs = outconfs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            scores = np.max(confs, axis=1).flatten()

            if len(boxes) > 0:
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out

def freeze_bn(model):
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if "BatchNorm2d" in classname:
            m.affine = False
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            m.eval() 
    model.apply(set_bn_eval)




    
