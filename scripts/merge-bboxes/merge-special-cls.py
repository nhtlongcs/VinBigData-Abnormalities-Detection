import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv-in",
    default="../../data/raw/train_filled.csv",
    type=str,
    help="path to raw filled csv",
)
parser.add_argument(
    "--output-path",
    default="../../data/raw/train_filled_removed_dup.csv",
    type=str,
    help="path output csv",
)


def calc_conf(row, count_df, data_df, classes=[0, 3, 14], nor=3):

    """
    classes : processing classes
    nor : number of rad_ids
    """

    ann = row[["x_min", "y_min", "x_max", "y_max", "score"]]
    if row.class_id in classes:
        key = (row.image_id, row.class_id)
        score = (
            count_df[(row.image_id, row.class_id)] / nor
            if key in count_df.keys()
            else 0
        )
        ann = data_df.get_group(key).mean()
        ann = *ann[["x_min", "y_min", "x_max", "y_max"]], round(score, 2)

    row[["x_min", "y_min", "x_max", "y_max", "score"]] = ann
    return row


def clean(csv: pd.DataFrame) -> tuple:

    process_csv = csv[(csv.class_id == 0) | (csv.class_id == 3) | (csv.class_id == 14)]
    ignore_csv = csv[(csv.class_id != 0) & (csv.class_id != 3) & (csv.class_id != 14)]
    process_csv = process_csv.drop_duplicates(
        subset=["image_id", "class_id"], keep="last"
    )

    rad_count_df = csv.groupby(["image_id", "class_id"]).rad_id.count()
    data_df = csv.groupby(["image_id", "class_id"])

    process_csv = process_csv.progress_apply(
        lambda row: calc_conf(row, rad_count_df, data_df), axis=1
    )
    for df in [process_csv, ignore_csv]:
        df[["x_min", "y_min", "x_max", "y_max"]] = csv[
            ["x_min", "y_min", "x_max", "y_max"]
        ].astype(int)

    import pdb

    pdb.set_trace()

    return process_csv, ignore_csv


if __name__ == "__main__":
    tqdm.pandas()

    args = parser.parse_args()
    out_path = Path(args.output_path)
    csv_path = Path(args.csv_in)
    csv = pd.read_csv(csv_path)

    csv["score"] = 1

    final_csv = pd.concat([*clean(csv)])

    print(f"droped {len(csv) - len(final_csv)} rows")

    out_path = (
        out_path / "train_removed_dup.csv" if os.path.isdir(out_path) else out_path
    )
    final_csv.to_csv(out_path, index=False)
