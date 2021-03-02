from utils.getter import *
import argparse
import os


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_transforms = get_augmentation(config, _type = 'train')
    val_transforms = get_augmentation(config, _type = 'val')
    
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
        batch_size=config.batch_size, 
        shuffle = True, 
        collate_fn=trainset.collate_fn, 
        num_workers= config.num_workers, 
        pin_memory=True)

    valloader = DataLoader(
        valset, 
        batch_size=config.batch_size, 
        shuffle = False,
        collate_fn=valset.collate_fn, 
        num_workers= config.num_workers, 
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

    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    model = Detector(
            n_classes=NUM_CLASSES,
            model = net,
            metrics=metric,
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Initialize weights')
        init_weights(model.model)
        start_epoch, start_iter, best_value = 0, 0, 0.0

    # One cycle lr-scheduler. Source: https://github.com/ultralytics/yolov5
    lf = one_cycle(1, 0.158, config.num_epochs)  # cosine 1->hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=lf)

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.log_path),
                     scheduler = scheduler,
                     evaluate_per_epoch = args.val_interval,
                     visualize_when_val = args.no_visualization)
    
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s)')
    print(f"Start training at [{start_epoch}|{start_iter}]")
    print(f"Current best MAP: {best_value}")
    
    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--config' , type=str, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=str, default='0', help='coefficients of efficientdet')
    parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
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
    


