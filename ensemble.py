import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.postprocess import box_fusion

parser = argparse.ArgumentParser(description='Ensemble submission')
parser.add_argument('--csv_folder', type=str, default = None,help='all csvs prediction')
parser.add_argument('--weight', type=str, default = '[1.0, 1.0, 1.0, 1.0, 1.0]',help='weight for each prediction, in string format')
parser.add_argument('--mode', type=str, default = 'wbf',help='box fusion method')
parser.add_argument('--min_iou', type=float, default = 0.5,help='min iou threshold')
parser.add_argument('--min_conf', type=float, default = 0.001,help='min conf threshold')

args = parser.parse_args() 

#'image_id', 'class_id', 'score', 'x_min', 'y_min' , 'x_max', 'y_max'

"""
x_min, y_min, x_max, y_max are all normalized
"""

def ensemble(args):

    args.weight = eval(args.weight)

    df_list = []

    csv_paths = os.listdir(args.csv_folder)
    for csv_name in csv_paths:
        csv_path = os.path.join(args.csv_folder,  csv_name)
        df = pd.read_csv(csv_path)
        df_list.append(df.copy())
        
    # Get image ids, expects all submissions have same image ids
    image_ids = [
        row
        for row in df_list[0]['image_id'].unique()
    ]

    final_outputs = []

    for image_id in tqdm(image_ids):
        results_one_img = {
            'boxes': [],
            'scores': [],
            'class_ids': [],
        }
        for df in df_list:
            data = df[df["image_id"] == image_id]
            data = data.reset_index(drop=True)
            annotations = [
                row
                for row in zip(
                    data["class_id"], 
                    data["score"], 
                    data["x_min"], 
                    data["y_min"], 
                    data["x_max"],
                    data["y_max"])
            ]
 
            result_boxes = []
            result_scores = []
            result_classes = []
            for ann in annotations:
                class_id, score, x_min, y_min, x_max, y_max = ann
                box = [x_min, y_min, x_max, y_max]
                result_boxes.append(box)
                result_scores.append(float(score))
                result_classes.append(class_id)

            results_one_img['boxes'].append(result_boxes)
            results_one_img['scores'].append(result_scores)
            results_one_img['class_ids'].append(result_classes)

        final_boxes, final_scores, final_classes = box_fusion(
                results_one_img['boxes'],
                results_one_img['scores'],
                results_one_img['class_ids'],
                mode=args.mode, 
                iou_threshold=args.min_iou,
                weights = args.weight
            )

        indexes = np.where(final_scores >= args.min_conf)[0]
        final_boxes = final_boxes[indexes]
        final_scores = final_scores[indexes]
        final_classes = final_classes[indexes]

        for i in range(len(final_boxes)):
            x_min = final_boxes[i][0]
            y_min = final_boxes[i][1]
            x_max = final_boxes[i][2]
            y_max = final_boxes[i][3]
            final_outputs.append([
                image_id, 
                final_classes[i], 
                final_scores[i], 
                x_min, y_min, x_max, y_max])

    submission_df = pd.DataFrame(final_outputs, columns=['image_id', 'class_id', 'score', 'x_min', 'y_min' , 'x_max', 'y_max'])
    submission_df.to_csv('results/ensemble_output.csv', index=False)

if __name__ == '__main__':                 
    ensemble(args)
