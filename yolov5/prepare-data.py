import yaml
from os import listdir
from os.path import isfile, join
import os
import argparse

parser = argparse.ArgumentParser(description='prepare yolo data')
parser.add_argument('--fold', default=0,
                    help='fold id')
parser.add_argument('--output_dir', default='/content/src',
                    help='folder write yaml cfg, train.txt and val.txt')
parser.add_argument('--imgs_dir', default='/content/inputs/data/images/',
                    help='folder contain train/*.png val/*.png')
parser.add_argument('--lbls_dir', default='/content/inputs/data/labels/',
                    help='folder contains train/*.txt val/*.txt')

args = parser.parse_args()

FOLD            = int(args.fold)
WORKSPACE_DIR   = args.output_dir
IMGS_DIR        = args.imgs_dir
label_dir       = args.lbls_dir

os.chdir(WORKSPACE_DIR)

cwd = WORKSPACE_DIR

label_train_dir = join(label_dir, "train")
label_val_dir = join(label_dir, "val")

with open(join(cwd, "train.txt"), "w") as f:
    for img_id in listdir(label_train_dir):
        img_id = img_id.split(".")[0]
        path = join(IMGS_DIR,'train' ,f"{img_id}.png")
        f.write(path + "\n")
    f.close()


with open(join(cwd, "val.txt"), "w") as f:
    for img_id in listdir(label_val_dir):
        img_id = img_id.split(".")[0]
        path = join(IMGS_DIR, 'val' ,f"{img_id}.png")
        f.write(path + "\n")
    f.close()

classes = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    # "No finding"
]

data = dict(
    train=join(cwd, "train.txt"), val=join(cwd, "val.txt"), nc=14, names=classes
)

with open(join(cwd, "vinbigdata.yaml"), "w") as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(join(cwd, "vinbigdata.yaml"), "r")
print("\nyaml:")
print(f.read())
