import pycocotools.coco as coco
from os.path import join
import cv2
from matplotlib import pyplot as plt

annot_path = "../../inputs/nms512_ratio_fold0_train.json"
img_dir = "../../inputs/512/train/"


def display_annotationbyid(json_path, imageIds, img_dir):
    coco_json = coco.COCO(json_path)
    image_name = coco_json.loadImgs(imageIds)[0]["file_name"]
    image_path = join(img_dir, image_name)
    img = cv2.imread(image_path)
    annIds = coco_json.getAnnIds(imgIds=[imageIds], catIds=[])
    anns = coco_json.loadAnns(annIds)
    for i in anns:
        [x, y, w, h] = i["bbox"]
        img = cv2.rectangle(
            img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
        )
    return img, image_name


def display_annotationbyname(json_path, name, img_dir):
    coco_json = coco.COCO(json_path)
    img_ids = coco_json.getImgIds()
    for id in img_ids:
        image_name = coco_json.loadImgs(id)[0]["file_name"]
        if image_name.split(".png")[0] == name:
            img, image_name = display_annotationbyid(
                json_path=annot_path, imageIds=id, img_dir=img_dir
            )
            return img, image_name

    assert False, f"Not found {name} in json"


img, image_name = display_annotationbyname(
    json_path=annot_path, name="0a0ac65c40a9ac441651e4bfbde03c4e", img_dir=img_dir
)

# img, image_name = display_annotation(json_path=annot_path, imageIds=0, img_dir=img_dir)

cv2.imwrite(image_name, img)
