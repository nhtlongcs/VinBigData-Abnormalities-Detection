import pandas as pd
import optuna
import argparse
import sys

sys.path.insert(0, "scripts/evaluate")
from evaluate_map import mAPScore

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-trials", default=10, type=int, help="the number of experiments",
)
args = parser.parse_args()


def filter(
    low: int, high: int, det: pd.DataFrame, clf: dict, meta: dict, im_ids: list
) -> pd.DataFrame:
    """
    if prob < low_thr, pred= '14 1 0 0 1 1'
    if low_thr<=prob<high_thr, pred +=f' 14 {prob} 0 0 1 1'
    if high_thr<=prob, do nothing

    (prob là xs tìm được bệnh)
    """

    def addcls14(df: pd.DataFrame, prob: float):
        df = df.append(
            {
                "image_id": image_id,
                "class_id": 14,
                "x_min": 0,
                "y_min": 0,
                "x_max": 1 / meta[image_id]["width"],  # normalize pos class 14
                "y_max": 1 / meta[image_id]["height"],  # normalize pos class 14
                "score": 1,
            },
            ignore_index=True,
        )
        return df

    low = 0
    high = 1
    df = det.copy()
    df14 = pd.DataFrame(
        columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max", "score"]
    )

    for image_id in im_ids:
        prob = clf[image_id]["class14prob"]
        assert (
            image_id in meta.keys()
        ), "mismatch key json and metadata, please check again"

        if prob < low:
            ## Less chance of having any disease -> remove all preds
            df = df[df.image_id != image_id]  # remove all img == id
            df14 = addcls14(df14, 1)
        elif low <= prob < high:
            ## More change of having any dieseasedf = df[df.image_id != image_id]  # remove all img == id
            df14 = addcls14(df14, prob)
        elif high <= prob:
            ## Good chance of having any disease so believe in object detection model, do nothing
            pass
        else:
            raise ValueError("Prediction must be from [0-1]")
    df = df.reset_index(drop=True)
    df = pd.concat([df, df14])
    return df


class Reward(object):
    def __init__(self, gt_df: pd.DataFrame, normalize=True):
        self.metric = mAPScore(gt_df=gt_df, is_normalized=normalize)

    def eval(self, pred_df):
        self.metric.update_pred(pred_df)
        return self.metric.evaluate()[0]


class Objective(object):
    def __init__(self, preds_clf, preds_det, gt_dets, meta_data):
        self.preds = (
            preds_clf.copy()
        )  # predicted classification csv   (binary classify 14)
        self.preds = self.preds.set_index("image_id").T.to_dict()

        self.dets = (
            preds_det.copy()
        )  # predicted detection csv        (without class 14)
        self.targets = (
            gt_dets.copy()
        )  # target detection csv           (include class 14)
        self.img_ids = set(self.targets["image_id"])
        self.metadata = meta_data.copy()  # meta width heigh csv
        self.metadata = self.metadata.set_index("image_id").T.to_dict()
        self.metric = Reward(self.targets)

    def __call__(self, trial):
        low_thrs = trial.suggest_float("low-threshold", low=0.0001, high=0.99)
        # high_thrs = trial.suggest_float("high-threshold", low=0.0001, high=0.99)
        high_thrs = 0.95
        df = filter(
            low=low_thrs,
            high=high_thrs,
            det=self.dets,
            clf=self.preds,
            meta=self.metadata,
            im_ids=self.img_ids,
        )
        score = self.metric.eval(df)
        sys.stdout.flush()
        return score


if __name__ == "__main__":
    models_df = []
    folds_df = []
    num_folds = 5
    trials = []
    fold_id = 0

    binary_csv_path = "data/meta/class14-binary.csv"
    meta_data_csv_path = "data/meta/train_info.csv"
    det_pred_csv_path = f"data/preds/{fold_id}.csv"
    det_gt_csv_path = f"data/raw/folds/{fold_id}/{fold_id}_val.csv"

    class14_preds = pd.read_csv(binary_csv_path)
    detection_preds = pd.read_csv(det_pred_csv_path)
    detection_gt = pd.read_csv(det_gt_csv_path)
    metadata = pd.read_csv(meta_data_csv_path)
    for i in range(num_folds):
        # Create a study
        study = optuna.create_study(direction="maximize", study_name=f"tuning_fold{i}")
        study.optimize(
            Objective(
                preds_clf=class14_preds,
                preds_det=detection_preds,
                gt_dets=detection_gt,
                meta_data=metadata,
            ),
            n_trials=int(args.num_trials),
        )
        trial = study.best_trial
        trials.append(trial)

    for fold_id, trial in enumerate(trials):
        print(fold_id)
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
