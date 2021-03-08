import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from commons import *

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_in",
    default="../data/folds",
    type=str,
    help="input .csv file or folder to k-folds .csv files",
)
parser.add_argument(
    "--format", default="coco", type=str, help="convert to coco/yolo format"
)
parser.add_argument(
    "--resize",
    default=512,
    type=int,
    help="resize boxes to fit image size (for COCO only)",
)
parser.add_argument(
    "--keep_ratio",
    action="store_true",
    help="whether to keep the original aspect ratio (for COCO only)",
)
parser.add_argument(
    "--output_path",
    default="../data/annotations",
    type=str,
    help="output path to store annotations",
)
parser.add_argument(
    "--ignored", default="[14]", type=str, help="list of ignored indexes"
)
args = parser.parse_args()


def main(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if os.path.isfile(args.csv_in):
        # If is a .csv file
        df = pd.read_csv(args.csv_in)

        if args.format == "coco":
            dataset = CocoDataset(
                df.copy(), img_size=args.resize, keep_ratio=args.keep_ratio
            )
            dataset.ignore_classes = eval(args.ignored)
            filename = Path(args.csv_in)
            filename = os.path.basename(filename)
            args.output_path = os.path.join(
                args.output_path, filename[:-4] + "_coco.json"
            )
            dataset.write_all_annotations(save_filename=args.output_path)
        elif args.format == "yolo":
            dataset = YoloDataset(df.copy())
            dataset.ignore_classes = eval(args.ignored)
            dataset.write_all_annotations(save_dir=args.output_path)
    else:
        # If is a folder
        num_folds = int(len(os.listdir(args.csv_in)) / 2)  # number of file divides two

        for fold in range(num_folds):
            print(f"Converting fold {fold}:")
            for i in ["train", "val"]:
                csv_name = os.path.join(args.csv_in, f"{fold}_{i}.csv")
                df = pd.read_csv(csv_name)

                if args.format == "yolo":
                    dataset = YoloDataset(df.copy())
                    dataset.ignore_classes = eval(args.ignored)

                    dataset.write_all_annotations(
                        save_dir=os.path.join(args.output_path, f"yolo/{fold}/{i}"),
                    )

                elif args.format == "coco":
                    dataset = CocoDataset(
                        df.copy(), img_size=args.resize, keep_ratio=args.keep_ratio
                    )
                    dataset.ignore_classes = eval(args.ignored)
                    dataset.write_all_annotations(
                        save_filename=os.path.join(
                            args.output_path, f"coco/{fold}_{i}.json"
                        ),
                    )


if __name__ == "__main__":
    main(args)
