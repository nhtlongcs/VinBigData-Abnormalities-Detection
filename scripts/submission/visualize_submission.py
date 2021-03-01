import os
import csv
import math
import argparse

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
]
class_names = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
]

# Arguments parser
parser = argparse.ArgumentParser('a')
parser.add_argument('--csv', '-c', type=str,
                    help='path to submission csv')
parser.add_argument('--img_dir', '-d', type=str,
                    help='path to test image directory')
parser.add_argument('--meta', '-m', type=str,
                    help='path to metadata csv (format: image_id, width, height)')
parser.add_argument('--out_dir', '-o', type=str, default=None,
                    help='output directory (automatically created if not existed)')
parser.add_argument('--thres', '-t', type=float, default=0.4,
                    help='confidence threshold to visualize')
parser.add_argument('--class_filter', '-cc', type=int, default=-1,
                    help='only class to show, -1 for all (default: -1)')
parser.add_argument('--show', '-s', action='store_true',
                    help='only plot instead of saving to output directory')
args = parser.parse_args()

f = csv.reader(open(args.csv))
headers = next(f)
assert headers == ['image_id', 'PredictionString']
lines = list(f)

f = csv.reader(open(args.meta))
headers = next(f)
metadata = {image_id: (int(w), int(h)) for image_id, w, h, *_ in f}

if args.out_dir is not None:
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

for image_id, preds in tqdm(lines):
    preds = preds.split(' ')
    assert len(preds) % 6 == 0
    preds = [preds[i:i+6] for i in range(0, len(preds), 6)]

    im = cv2.imread(args.img_dir + '/' + image_id + '.png')
    nw, nh, _ = im.shape
    width, height = metadata[image_id]
    for class_id, conf, *pos in preds:
        class_id = int(class_id)
        if class_id == 14:
            continue
        if args.class_filter != -1 and class_id != args.class_filter:
            continue

        conf = float(conf)
        if conf <= args.thres:
            continue

        xmin, ymin, xmax, ymax = map(float, pos)

        xmin = int(xmin / width * nw)
        xmax = int(xmax / width * nw) + 1
        ymin = int(ymin / height * nh)
        ymax = int(ymax / height * nh) + 1

        thick = int(conf*10)

        cv2.rectangle(im,
                      (xmin, ymin), (xmax, ymax),
                      colors[class_id], thick)
        cv2.putText(im,
                    class_names[class_id],
                    (xmin, ymin - 12),
                    0, 1e-3 * nh,
                    colors[class_id], 1)

    if args.show:
        plt.imshow(im)
        plt.show()
    else:
        cv2.imwrite(args.out_dir + '/' + image_id + '.png', im)
