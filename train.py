from utils.getter import *
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.utils import init_weights, one_cycle

import torchvision

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

def train(args, config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    train_transforms = get_augmentation(config, _type = 'train')
    val_transforms = get_augmentation(config, _type = 'val')
    

    #input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    trainset = CocoDataset(
        config = config,
        root_dir = os.path.join('datasets', config.project_name, config.train_imgs),
        ann_path = os.path.join('datasets', config.project_name, config.train_anns),
        train=True,
        transforms=train_transforms)
    
    valset = CocoDataset(
        config = config,
        root_dir=os.path.join('datasets', config.project_name, config.val_imgs), 
        ann_path = os.path.join('datasets', config.project_name, config.val_anns),
        train=False,
        transforms=val_transforms)
    
    

    testset = CocoDataset(
        config = config,
        root_dir=os.path.join('datasets', config.project_name, config.val_imgs), 
        ann_path = os.path.join('datasets', config.project_name, config.val_anns),
        inference = True,
        train = False,
        transforms=val_transforms)

    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle = True, 
        drop_last= True, 
        collate_fn=trainset.collate_fn, 
        num_workers= args.num_workers, 
        pin_memory=True)

    valloader = DataLoader(
        valset, 
        batch_size=args.batch_size, 
        shuffle = False, 
        drop_last= True, 
        collate_fn=valset.collate_fn, 
        num_workers= args.num_workers, 
        pin_memory=True)

    NUM_CLASSES = len(config.obj_list)

    net = EfficientDetBackbone(
        num_classes=NUM_CLASSES, 
        compound_coef=args.compound_coef, 
        load_weights=True, 
        image_size=config.image_size)

    if args.saved_path is not None:
        args.saved_path = os.path.join(args.saved_path, args.config)

    if args.log_path is not None:
        args.log_path = os.path.join(args.log_path, args.config)

    metric = mAPScores(
        dataset=testset,
        min_conf = 0.01,
        min_iou = 0.15,
        retransforms = None)

    model = Detector(
            n_classes=NUM_CLASSES,
            model = net,
            metrics=metric,
            optimizer= torch.optim.AdamW,
            optim_params = {'lr': args.lr, 'weight_decay':0.0005},     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter = get_epoch_iters(args.resume)
    else:
        print('[Info] initialize weights')
        init_weights(model.model)
        start_epoch, start_iter = 0, 0

    # One cycle lr-scheduler. Source: https://github.com/ultralytics/yolov5
    lf = one_cycle(1, 0.158, args.num_epochs)  # cosine 1->hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=lf)

    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     logger = Logger(log_dir=args.log_path),
                     scheduler = scheduler,
                     evaluate_per_epoch = args.val_interval,
                     visualize_when_val = args.no_visualization)

    print("---------TRAINSET INFO----------------")
    print(trainset)
    print("---------VALSET INFO----------------")
    print(valset)
    print(trainer)
    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=args.num_epochs, print_per_iter=300)

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--config' , type=str, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=str, default='0', help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', action='store_true', default = False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
    parser.add_argument('--log_path', type=str, default='loggers/runs')
    parser.add_argument('--resume', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize')
    parser.add_argument('--saved_path', type=str, default='./weights')
    parser.add_argument('--no_visualization', action='store_false', help='whether to visualize box to ./sample when validating (for debug), default=on')

    args = parser.parse_args()
    config = Config(os.path.join('configs',args.config+'.yaml'))


    train(args, config)
    


