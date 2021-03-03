import pandas as pd
import optuna
import argparse


# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-trials", default=1, type=int, help="the number of experiments",
)
args = parser.parse_args()


def filter(low: int, high: int, df: pd.DataFrame) -> pd.DataFrame:
    return df


class Objective(object):
    def __init__(self, preds_clf, preds_det, gt_dets):
        self.preds = (
            preds_clf.copy()
        )  # predicted classification csv   (binary classify 14)
        self.dets = (
            preds_det.copy()
        )  # predicted detection csv        (without class 14)
        self.targets = (
            gt_dets.copy()
        )  # target detection csv           (include class 14)

    def __call__(self, trial):
        low_thrs = trial.suggest_float("low-threshold", low=0.0001, high=0.99)
        high_thrs = trial.suggest_float("high-threshold", low=0.0001, high=0.99)
        import pdb

        pdb.set_trace()
        df = filter(low=low_thrs, high=high_thrs, df=self.preds)
        score = eval_map(df, self.targets)
        sys.stdout.flush()
        return score


if __name__ == "__main__":
    models_df = []
    folds_df = []
    num_folds = 5
    thresholds = []

    binary_csv_path = "data/class14-binary.csv"
    fold_id = 0
    det_pred_csv_path = f"data/preds/{fold_id}.json"
    det_gt_csv_path = f"data/folds/{fold_id}/{fold_id}_val.csv"

    class14_preds = pd.read_csv(binary_csv_path)
    # detection_preds = pd.read_csv(det_pred_csv_path)
    detection_gt = pd.read_csv(det_gt_csv_path)
    import pdb

    pdb.set_trace()
    for i in range(num_folds):
        # Create a study
        study = optuna.create_study(direction="maximize", study_name=f"tuning_fold{i}")
        study.optimize(
            Objective(
                preds_clf=binary_csv_path,
                preds_det=det_pred_csv_path,
                gt_dets=det_gt_csv_path,
            ),
            n_trials=int(args.n_trials),
        )
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        thresholds.append(thrs)
