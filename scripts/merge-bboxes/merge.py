import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion, nms




# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--iou_threshold', default=0.5, type=float,
                        help='iou threshold to remove overlappings')
parser.add_argument('--skip_threshold', default=0.0001 , type=float,
                        help='skip box whose confidence lower than threshold (for WBF)')

parser.add_argument('--csv_in', type=str, help='path to input csv file')
parser.add_argument('--csv_out', default=None, type=str, help='path to output csv file')
parser.add_argument('--type', default='wbf', type=str, help='box fusion method')
parser.add_argument('--class_mapping', default='./class_mapping.csv', type=str, help='class name csv file')
parser.add_argument('--ignored_classes', default='[14]', type=str, help='string of list of ignored classes')

args = parser.parse_args()

def main(args):
    if args.csv_out is None:
        args.csv_out = args.csv_in[:-4] + '_' + args.type + '.csv'
    
    # Ignore classes
    IGNORED_CLASSES = eval(args.ignored_classes)

    #Read class mapping indexes
    class_mapping_df = pd.read_csv(args.class_mapping, header=None)
    classes_to_idx = {row[1]:row[0] for idx, row in class_mapping_df.iterrows()}
    
    # Read csv
    df = pd.read_csv(args.csv_in)

    # Ignore box with class id
    for ignore_class in IGNORED_CLASSES:
        df = df[df["class_id"] != ignore_class]


    # Start fusing boxes
    results = []
    image_ids = df["image_id"].unique()

    for image_id in tqdm(image_ids, total=len(image_ids)):

        # All annotations for the current image.
        data = df[df["image_id"] == image_id]
        data = data.reset_index(drop=True)

        width = data['width'][0]
        height = data['height'][0]
        

        annotations = {}
        weights = []

        # WBF expects the coordinates in 0-1 range.
        max_value = data.iloc[:, 4:].values.max()
        data.loc[:, ["x_min", "y_min", "x_max", "y_max"]] = data.iloc[:, 4:] / max_value

        # Loop through all of the annotations
        for idx, row in data.iterrows():

            rad_id = row["rad_id"]

            if rad_id not in annotations:
                annotations[rad_id] = {
                    "boxes_list": [],
                    "scores_list": [],
                    "labels_list": [],
                }

                # We consider all of the radiologists as equal.
                weights.append(1.0) 

            annotations[rad_id]["boxes_list"].append(
                [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
            )
            annotations[rad_id]["scores_list"].append(1.0)
            annotations[rad_id]["labels_list"].append(row["class_id"])

        boxes_list = []
        scores_list = []
        labels_list = []

        for annotator in annotations.keys():
            boxes_list.append(annotations[annotator]["boxes_list"])
            scores_list.append(annotations[annotator]["scores_list"])
            labels_list.append(annotations[annotator]["labels_list"])

        # Calculate WBF

        if args.type == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=args.iou_threshold,
                skip_box_thr=args.skip_threshold,
            )
        elif args.type == 'nms':
            boxes, scores, labels = nms(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=args.iou_threshold,
            )

        for idx, box in enumerate(boxes):
            results.append(
                {
                    "image_id": image_id,
                    "class_name": classes_to_idx[int(labels[idx])],
                    "class_id": int(labels[idx]),
                    "rad_id": 'wbf',
                    "x_min": int(box[0]*max_value),
                    "y_min": int(box[1]*max_value),
                    "x_max": int(box[2]*max_value),
                    "y_max": int(box[3]*max_value),
                    "width": width,
                    "height": height,
                }
            )

    results = pd.DataFrame(results)
    results.to_csv(args.csv_out, index = False)
    
    print(f"Number of original boxes : {len(df)}")
    print(f"Number of boxes (after removing overlappings): {len(results)}")
if __name__ =='__main__':
    main(args)