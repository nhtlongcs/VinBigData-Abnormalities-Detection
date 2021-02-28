from metrics import *
from datasets import *
from models import *
from trainer import *
from augmentations import *
from loggers import *
from configs import *


import torch
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils.utils import init_weights, one_cycle
import torchvision.models as models
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from .random_seed import seed_everything


def get_instance(config, **kwargs):
    # Inherited from https://github.com/vltanh/pytorch-template
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)

def get_lr_policy(opt_config):
    optimizer_params = {}
    if opt_config["name"] == 'sgd':
        optimizer = SGD
        optimizer_params = {
            'lr': opt_config.lr, 
            'weight_decay': opt_config.weight_decay,
            'momentum': opt_config.momentum,
            'nesterov': True}
    elif opt_config["name"] == 'adam':
        optimizer = AdamW
        optimizer_params = {
            'lr': opt_config.lr, 
            'weight_decay': opt_config.weight_decay,
            'betas': (opt_config.momentum, 0.999)}
    
    return optimizer, optimizer_params