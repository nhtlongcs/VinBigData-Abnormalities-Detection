import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

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

class CocoDataset:
    def __init__(self, df: pd.DataFrame, img_size: int = 512, keep_ratio: bool = False):
        self.df = df
        self.img_size = img_size
        self.keep_ratio = keep_ratio
        self.ignore_classes = IGNORE_CLASSES
        self.preprocess()

    def preprocess(self):
        self.df['x_min'] = self.df.apply(lambda row: 0 if row.class_id==14 else row.x_min, axis=1)
        self.df['x_max'] = self.df.apply(lambda row: 1 if row.class_id==14 else row.x_max, axis=1)
        self.df['y_min'] = self.df.apply(lambda row: 0 if row.class_id==14 else row.y_min, axis=1)
        self.df['y_max'] = self.df.apply(lambda row: 1 if row.class_id==14 else row.y_max, axis=1)

    def get_order_classes(self) -> list:
        df = self.df
        ignore_classes = self.ignore_classes
        for ignore_c in ignore_classes:
            df = df.loc[df.class_id != ignore_c]
        class_ids, class_names = list(zip(*set(zip(df.class_id, df.class_name))))
        classes = list(np.array(class_names)[np.argsort(class_ids)])
        classes = list(map(lambda x: str(x), classes))  # sorted by idxes
        return classes

    @staticmethod
    def train_test_split(in_filename: str, ratio: float =0.8):
        import funcy
        from sklearn.model_selection import train_test_split

        TRAIN_PATH = in_filename[:-5]+'_train.json'
        VAL_PATH = in_filename[:-5]+'_val.json'

        def save_coco(file, images, annotations, categories):
            with open(file, 'wt', encoding='UTF-8') as coco:
                json.dump({ 'images': images, 
                    'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

        def filter_annotations(annotations, images):
            image_ids = funcy.lmap(lambda i: int(i['id']), images)
            return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

        with open(in_filename, 'rt', encoding='UTF-8') as annotations:
            coco = json.load(annotations)
        
            images = coco['images'] 
            annotations = coco['annotations']
            categories = coco['categories']

            x, y = train_test_split(images, train_size=ratio)

            save_coco(TRAIN_PATH,  x, filter_annotations(annotations, x), categories)
            save_coco(VAL_PATH, y, filter_annotations(annotations, y), categories)

            print('Split completed!')

    def write_all_annotations(self, save_filename: str = ".cache/coco.json") -> None:
        database = self.df
        ignore_classes = self.ignore_classes

        try:
            os.makedirs(os.path.dirname(save_filename))
        except:
            pass
        
        my_dict = {
            'images': [],
            'annotations': [],
            'categories': [],
        }

        img_count = 0
        item_count = 0
        image_dict = {}
        labels = self.get_order_classes()

        for label_idx, label in enumerate(labels):
            if int(label_idx) not in ignore_classes:
                class_dict = {
                    'supercategory': None,
                    'id': label_idx+1,  #Coco starts from 1
                    'name': label
                }
                my_dict['categories'].append(class_dict)

        annotations = [
            row
            for row in zip(
                database["image_id"], 
                database["class_id"], 
                database["rad_id"], 
                database["x_min"], 
                database["y_min"], 
                database["x_max"], 
                database['y_max'],
                database['width'],
                database['height'])
        ]

        for row in tqdm(annotations):
            image_name, class_id, rad_id, xmin, ymin, xmax, ymax, width, height = row
            if int(class_id) not in ignore_classes:
                if image_name not in image_dict.keys():
                    image_dict[image_name] = img_count
                    img_count += 1
                    image_id = image_dict[image_name]
                    img_dict = {
                        'file_name': image_name+".png",
                        'height': height,
                        'width':width,
                        'id':image_id,
                    }
                    my_dict['images'].append(img_dict)

                ratio = resized_ratio(
                    raw_shape = (width, height),
                    target_size = (self.img_size, self.img_size),
                    keep_ratio=self.keep_ratio,
                )

                # Resize bbox
                xmax = xmax*ratio[0]
                xmin = xmin*ratio[0]
                ymax = ymax*ratio[1]
                ymin = ymin*ratio[1]
                ann_w = xmax-xmin
                ann_h = ymax-ymin
                image_id = image_dict[image_name]
                ann_dict = {
                    'id': item_count,
                    'image_id': image_id,
                    'bbox': [xmin, ymin, ann_w, ann_h],
                    'area': ann_w * ann_h,
                    'category_id': int(class_id)+1, #Coco starts from 1
                    'rad_id': str(rad_id),
                    'iscrowd': 0 
                }
                item_count += 1
                my_dict['annotations'].append(ann_dict)

        with open(save_filename, "w") as outfile:  
            json.dump(my_dict, outfile) 

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
            df["class_id"], df["x_min"], df["y_min"], df["x_max"], df["y_max"], df["rad_id"]
        )
    ]
    return annotations

def get_image_info(
    img_id: str,
    database: pd.DataFrame,
) -> list:
    database = database.loc[database.image_id == img_id].head(1)
    base_size = int(database.width), int(database.height)
    return base_size

def get_bboxes_xyxy_resized(
    size: int,
    img_id: str,
    database: pd.DataFrame,
    ignore_classes: list = IGNORE_CLASSES,
    keep_ratio: bool = True
) -> list:
    bboxes = get_bboxes_xyxy(
        img_id=img_id, database=database, ignore_classes=ignore_classes
    )
    database = database.loc[database.image_id == img_id].head(1)
    base_size = int(database.width), int(database.height)
    target_size = size, size
    ratio = resized_ratio(raw_shape=base_size, target_size=target_size, keep_ratio=keep_ratio)
    annotations = [
        (id, xmin * ratio[0], ymin * ratio[1], xmax * ratio[0], ymax * ratio[1], rad_id)
        for id, xmin, ymin, xmax, ymax, rad_id in bboxes
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

