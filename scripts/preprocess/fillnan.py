import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="data/raw/train.csv", type=str, help="path to csv file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/train_filled.csv",
        help="output filename and directory",
    )
    return parser.parse_args()


cfg = parse_args()

if __name__ == "__main__":
    df = pd.read_csv(cfg.csv)
    df["x_min"] = df.apply(lambda row: 0 if row.class_id == 14 else row.x_min, axis=1)
    df["x_max"] = df.apply(lambda row: 1 if row.class_id == 14 else row.x_max, axis=1)
    df["y_min"] = df.apply(lambda row: 0 if row.class_id == 14 else row.y_min, axis=1)
    df["y_max"] = df.apply(lambda row: 1 if row.class_id == 14 else row.y_max, axis=1)
    df.to_csv(cfg.out, index=False)
