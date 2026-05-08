import argparse
import json
from functools import cache
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
from lifelines.exceptions import ConvergenceError

from ..experiments.models.xg_boost import GBMWeibull

DS_NAME = ""
METRIC = "c_index"
N_JOBS = 8
NUM_BOOST_ROUND = 2000

def delete_const_cats(x_cat, cat_chosen_mask = None):
    return_two = cat_chosen_mask is None
    if cat_chosen_mask is None:
        empty_idx = np.min(x_cat, 0) == np.max(x_cat, 0)
        cat_chosen_mask = ~empty_idx
    x_cat = x_cat[:, cat_chosen_mask].copy()
    if return_two:
        return x_cat, cat_chosen_mask
    else:
        return x_cat


@cache
def get_ds():
    data = np.load(f"data/{DS_NAME}.npz")
    x_num_train = data["X_num_train"]
    x_num_val = data["X_num_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    if "X_cat_train" in data.files and data["X_cat_train"].shape[1] > 0:
        x_cat_train, cat_mask = delete_const_cats(data["X_cat_train"])
        x_cat_val = delete_const_cats(data["X_cat_val"], cat_mask)
    else:
        x_cat_train = None
        x_cat_val = None

    return x_num_train, x_cat_train, y_train, x_num_val, x_cat_val, y_val


def objective(trial: optuna.Trial):
    x_num_train, x_cat_train, y_train, x_num_val, x_cat_val, y_val = get_ds()

    early_stopping_rounds = trial.suggest_categorical("early_stopping_rounds", [10, 25, 50, 100])
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True)
    max_depth = trial.suggest_categorical("max_depth", [2, 3, 4, 5, 6, 8])
    min_child_weight = trial.suggest_float("min_child_weight", 1e-1, 20.0, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    l2_reg = trial.suggest_float("l2_reg", 1e-4, 10.0, log=True)
    l1_reg = trial.suggest_float("l1_reg", 1e-4, 10.0, log=True)
    aft_loss_distribution = trial.suggest_categorical(
        "aft_loss_distribution",
        ["normal", "logistic", "extreme"],
    )
    aft_loss_distribution_scale = trial.suggest_float("aft_loss_distribution_scale", 0.1, 10.0, log=True)
    wb_pen_mode = trial.suggest_categorical('wb_pen_mode', [False, True])
    if wb_pen_mode:
        wb_penalty = trial.suggest_loguniform('wb_penalty', 1e-2, 10)
    else:
        wb_penalty = 0.0
    wb_l1_ratio = trial.suggest_uniform('wb_l1_ratio', 0, 1)
    model = GBMWeibull(
        num_boost_round=NUM_BOOST_ROUND,
        time_ruler=np.unique(y_train["time"]),
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        l2_reg=l2_reg,
        l1_reg=l1_reg,
        tree_method="hist",
        aft_loss_distribution=aft_loss_distribution,
        aft_loss_distribution_scale=aft_loss_distribution_scale,
        wb_penalty=wb_penalty,
        wb_l1_ratio=wb_l1_ratio,
        n_jobs=N_JOBS,
        random_state=None,
    )

    with mlflow.start_run(nested=True, run_name=str(trial.number + 1)):
        mlflow.log_params(trial.params)
        model.fit(
            x_num_train,
            x_cat_train,
            y_train,
            (x_num_val, x_cat_val, y_val),
        )
        c_index, ibs, auc = model.score(x_num_val, x_cat_val, y_val)
        mlflow.log_metric("c_index", c_index)
        mlflow.log_metric("ibs", ibs)
        mlflow.log_metric("auc", auc)
        scores = {
            "c_index": c_index,
            "ibs": ibs,
            "auc": auc,
        }
        return scores[METRIC]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("-trials", type=int, required=False, default=200)
    parser.add_argument("-threads", type=int, required=False, default=8)
    parser.add_argument("-tracking-uri", type=str, required=False, default="http://127.0.0.1:5000")
    parser.add_argument(
        "-metric",
        type=str,
        required=False,
        default="c_index",
        choices=["c_index", "ibs", "auc"],
    )
    args = parser.parse_args()

    DS_NAME = args.data
    METRIC = args.metric
    N_JOBS = args.threads
    torch.set_num_threads(args.threads)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(f"optuna_gbm_wb_{DS_NAME}")

    with mlflow.start_run(run_name=f"gbm_wb_{METRIC}"):
        study_direction = "minimize" if METRIC == "ibs" else "maximize"
        study = optuna.create_study(direction=study_direction)
        study.optimize(objective, n_trials=args.trials, catch=[ConvergenceError])

        best_params = dict(study.best_trial.params)
        best_params.pop('wb_pen_mode')
        best_params["num_boost_round"] = NUM_BOOST_ROUND
        # best_params["val_metric"] = METRIC
        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_{METRIC}", study.best_value)

        print("Best trial:")
        print(best_params)
        print(study.best_trial.value)

        out_dir = Path(f"pkg/params_selection/params/{DS_NAME}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "gbm_wb.json"
        with out_file.open("w", encoding="utf-8") as file:
            json.dump(best_params, file, indent=2)
        print("Saved best params to", out_file)
