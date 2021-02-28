# Author: Zylo117

import torch
import torchvision
import numpy as np
from torch import nn

from .effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from .effdet.efficientdet import HeadNet


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, image_size=[512,512],**kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.name = f'EfficientDet-D{compound_coef}'
        config = get_efficientdet_config(f'tf_efficientdet_d{compound_coef}')
        config.image_size = image_size
        config.norm_kwargs=dict(eps=.001, momentum=.01)

        net = EfficientDet(config, pretrained_backbone=load_weights)
        net.reset_head(num_classes=num_classes)
        net.class_net = HeadNet(config, num_outputs=config.num_classes)

        self.model = DetBenchTrain(net, config)


    def forward(self, inputs, targets):
        return self.model(inputs, targets)

    def detect(self, inputs, img_sizes, img_scales, conf_threshold=0.2):
        outputs = self.model.detect(inputs, img_sizes, img_scales)
        outputs = outputs.cpu().numpy()
        out = []
        for i, output in enumerate(outputs):
            boxes = output[:, :4]
            labels = output[:, -1]
            scores = output[:,-2]

            if len(boxes) > 0:
                selected = scores >= conf_threshold
                boxes = boxes[selected].astype(np.int32)
                scores = scores[selected]
                labels = labels[selected]
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