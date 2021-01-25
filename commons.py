import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

IGNORE_CLASSES = [14]


class YoloDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = normalize(df)
        self.ignore_classes = IGNORE_CLASSES

    def get_order_classes(self) -> list:
        df = self.df
        ignore_classes = self.ignore_classes
        for ignore_c in ignore_classes:
            df = df.loc[df.class_id != ignore_c]
        class_ids, class_names = list(zip(*set(zip(df.class_id, df.class_name))))
        classes = list(np.array(class_names)[np.argsort(class_ids)])
        classes = list(map(lambda x: str(x), classes))  # sorted by idxes
        return classes

    def write_one_annotation(self, image_id: str, save_dir: str) -> None:
        from os.path import join

        database = self.df
        ignore_classes = self.ignore_classes

        df = database.loc[database.image_id == image_id]

        annotations = [
            row
            for row in zip(df["class_id"], df["x_mid"], df["y_mid"], df["w"], df["h"])
        ]
        with open(join(save_dir, image_id + ".txt"), "w") as f:
            for annotation in annotations:
                id, x, y, w, h = annotation
                if int(id) not in ignore_classes:
                    f.write(f"{id} {x} {y} {w} {h}\n")
            f.close()

    def write_all_annotations(self, save_dir: str = ".cache") -> None:
        database = self.df

        try:
            os.makedirs(save_dir)
        except:
            pass
        img_ids = set(list(database.image_id.values))

        for img_id in tqdm(img_ids):
            self.write_one_annotation(
                image_id=img_id, save_dir=save_dir,
            )


def get_XY(
    df: pd.DataFrame,
    feature_cols: list = [
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "x_mid",
        "y_mid",
        "w",
        "h",
        "area",
    ],
    label_col: list = ["class_id"],
) -> tuple:

    return df[feature_cols], df[label_col]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # preprocess and normalize inp
    df["x_min"] = df.apply(lambda row: (row.x_min) / row.width, axis=1)
    df["y_min"] = df.apply(lambda row: (row.y_min) / row.height, axis=1)

    df["x_max"] = df.apply(lambda row: (row.x_max) / row.width, axis=1)
    df["y_max"] = df.apply(lambda row: (row.y_max) / row.height, axis=1)

    df["x_mid"] = df.apply(lambda row: (row.x_max + row.x_min) / 2, axis=1)
    df["y_mid"] = df.apply(lambda row: (row.y_max + row.y_min) / 2, axis=1)

    df["w"] = df.apply(lambda row: (row.x_max - row.x_min), axis=1)
    df["h"] = df.apply(lambda row: (row.y_max - row.y_min), axis=1)

    df["area"] = df["w"] * df["h"]
    return df


def resized_ratio(raw_shape: tuple, target_size: tuple, keep_ratio: bool = True) -> tuple:
    if keep_ratio:
        w_ratio = target_size[0] / raw_shape[0]
        h_ratio = target_size[1] / raw_shape[1]
        img_ratio = min(w_ratio, h_ratio), min(w_ratio, h_ratio)
    else:
        img_ratio = target_size[0] / raw_shape[0], target_size[1] / raw_shape[1]
    return img_ratio


def get_bboxes_xyxy(
    img_id: str, database: pd.DataFrame, ignore_classes: list = IGNORE_CLASSES
) -> list:
    df = database.loc[database.image_id == img_id]
    for c in ignore_classes:
        df = df.loc[database.class_id != c]

    annotations = [
        row
        for row in zip(
            df["class_id"], df["x_min"], df["y_min"], df["x_max"], df["y_max"]
        )
    ]
    return annotations


def get_bboxes_xyxy_resized(
    size: int,
    img_id: str,
    database: pd.DataFrame,
    ignore_classes: list = IGNORE_CLASSES,
) -> list:
    bboxes = get_bboxes_xyxy(
        img_id=img_id, database=database, ignore_classes=ignore_classes
    )
    database = database.loc[database.image_id == img_id].head(1)
    base_size = int(database.width), int(database.height)
    target_size = size, size
    ratio = resized_ratio(raw_shape=base_size, target_size=target_size, keep_ratio=True)
    annotations = [
        (id, xmin * ratio[0], ymin * ratio[1], xmax * ratio[0], ymax * ratio[1])
        for id, xmin, ymin, xmax, ymax in bboxes
    ]

    return annotations


def plt_bboxes(img_path: str, bboxes: list) -> None:
    import cv2

    img = cv2.imread(img_path)
    for id, xmin, ymin, xmax, ymax in bboxes:
        id, xmin, ymin, xmax, ymax = id, int(xmin), int(ymin), int(xmax), int(ymax)
        tmp = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()

