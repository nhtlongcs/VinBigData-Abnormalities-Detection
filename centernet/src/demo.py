from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
from tqdm import tqdm
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ["jpg", "jpeg", "png", "webp"]
video_ext = ["mp4", "mov", "avi", "mkv"]
time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]
normalize = True
conf_thresh = 0.01
class_name = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
save_dir = "/content/cp/"


def to_float(x):
    return float("{:.2f}".format(x))


dictionary = dict(zip(class_ids, class_name))


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == "webcam" or opt.demo[opt.demo.rfind(".") + 1 :].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == "webcam" else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow("input", img)
            ret = detector.run(img)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind(".") + 1 :].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]
        txt_path = os.path.join(save_dir, "preds.csv")
        os.makedirs("/".join(map(str, txt_path.split("/")[:-1])), exist_ok=True)
        f = open(txt_path, "w")
        f.write("image_id,class_name,class_id,score,x_min,y_min,x_max,y_max\n")
        for image_name in tqdm(image_names):
            if normalize:
                h, w, _ = cv2.imread(image_name).shape
            ret = detector.run(image_name)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            # print(time_str)
            dets = ret["results"]
            img_id = image_name.split("/")[-1].split(".")[0]
            # print(txt_path)
            for key in dets.keys():
                if key in dictionary.keys():
                    for i in range(len(dets[key])):
                        if dets[key][i, 4] > conf_thresh:
                            if normalize:
                                f.write(
                                    "{}, {}, {}, {}, {}, {}, {}, {}\n".format(
                                        img_id,
                                        dictionary[key],
                                        int(key),
                                        float(dets[key][i, 4]),
                                        float(dets[key][i, 0] / w),
                                        float(dets[key][i, 1] / h),
                                        float(dets[key][i, 2] / w),
                                        float(dets[key][i, 3] / h),
                                    )
                                )
                            else:
                                f.write(
                                    "{}, {}, {}, {}, {}, {}, {}, {}\n".format(
                                        img_id,
                                        dictionary[key],
                                        int(key),
                                        float(dets[key][i, 4]),
                                        int(dets[key][i, 0]),
                                        int(dets[key][i, 1]),
                                        int(dets[key][i, 2]),
                                        int(dets[key][i, 3]),
                                    )
                                )
        f.close()


if __name__ == "__main__":
    opt = opts().init()
    demo(opt)

