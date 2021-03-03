import argparse
import pandas as pd
import json
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm


def convert_coco_json_to_csv(filename, output_path, meta_path=None, normalize=True):
    if normalize:
        assert meta_path != None, "missing meta data path <image_id,width,height>"
        metadata = pd.read_csv(meta_path)
        metadata = metadata.set_index("image_id").T.to_dict()
    os.makedirs(output_path, exist_ok=True)
    filename = Path(filename)
    s = json.load(open(filename, "r"))
    out_file = os.path.join(
        output_path, os.path.basename(filename).split(".")[0] + ".csv"
    )
    out = open(out_file, "w")
    out.write("image_id,class_id,x_min,y_min,x_max,y_max,score\n")
    pbar = tqdm(s)
    for ann in pbar:
        pbar.set_description(f"exporting {os.path.basename(filename)}")
        image_id = ann["image_id"]
        x1 = ann["bbox"][0]
        x2 = ann["bbox"][0] + ann["bbox"][2]
        y1 = ann["bbox"][1]
        y2 = ann["bbox"][1] + ann["bbox"][3]
        if normalize:
            assert (
                image_id in metadata.keys()
            ), "mismatch key json and metadata, please check again"
            x1 /= metadata[image_id]["width"]
            x2 /= metadata[image_id]["width"]
            y1 /= metadata[image_id]["height"]
            y2 /= metadata[image_id]["height"]
        label = ann["category_id"]
        score = ann["score"]
        out.write(
            "{},{},{},{},{},{},{}\n".format(image_id, label, x1, y1, x2, y2, score)
        )
    out.close()

    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.sort_values("image_id", inplace=True)
    s1.to_csv(out_file, index=False)


# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--json",
    default="data/preds/",
    type=str,
    help="input .json file or folder to k-folds .json files",
)
parser.add_argument(
    "--output-path",
    default="data/preds/",
    type=str,
    help="output path to store annotations",
)
parser.add_argument(
    "--meta-path",
    default="data/train_info.csv",
    type=str,
    help="meta data width height path",
)
parser.add_argument(
    "--normalize", action="store_true", default=False, help="normalize bboxes flag",
)
args = parser.parse_args()

if os.path.isfile(args.json):
    convert_coco_json_to_csv(
        filename=args.json, output_path=args.output_path, meta_path=args.meta_path
    )
else:
    json_ls = glob(f"{args.json}/*.json")
    for json_file in json_ls:
        convert_coco_json_to_csv(
            filename=json_file, output_path=args.output_path, meta_path=args.meta_path
        )

