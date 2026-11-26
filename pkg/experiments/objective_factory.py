from __future__ import annotations

import gc
from functools import partial
from typing import Callable, Literal, get_args

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .exp_util import get_model


Model = Literal[
    "mlp",
    "realmlp",
    "tabm",
    "weibull_params",
    "weibull_proba",
    "deepsurv",
    "deephit",
    "survtrace",
    "coxph",
    "rsf",
    "gbm_km",
    "gbm_wb",
]
Metric = Literal["c_index", "ibs", "auc"]

NUM_BOOST_ROUND = 2000


def get_models_list() -> list[str]:
    return list(get_args(Model))


def get_metrics_list() -> list[str]:
    return list(get_args(Metric))


def k_fold_wrapper(objective: Callable, k_stop: int, k_all: int) -> Callable:
    def wrapper(trial, X_num, X_cat, y, device, metric, seed):
        splitter = StratifiedKFold(n_splits=k_all, shuffle=True, random_state=seed)
        scores = []
        folds = splitter.split(X_num, y["censor"])
        for _, (train_idx, test_idx) in zip(range(k_stop), folds):
            data = {
                "X_num_train": X_num[train_idx],
                "X_cat_train": X_cat[train_idx],
                "X_num_test": X_num[test_idx],
                "X_cat_test": X_cat[test_idx],
                "y_train": y[train_idx],
                "y_test": y[test_idx],
            }
            scores.append(objective(trial=trial, data=data, device=device, metric=metric, seed=seed))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return float(np.mean(scores))

    return wrapper


def _sample_piecewise_params(trial) -> dict:
    return {
        "rtdl_n_bins": trial.suggest_categorical("rtdl_n_bins", [32, 48, 64]),
        "emb_dim_rtdl": trial.suggest_categorical("emb_dim_rtdl", [8, 12, 16]),
        "rtdl_activation": trial.suggest_categorical("rtdl_activation", [True, False]),
    }


def _sample_tabm_common(
    trial, d_blocks: list[int], activations: list[str], llh_radii: list[int]
) -> dict:
    params = {
        "n_blocks": trial.suggest_int("n_blocks", 1, 3),
        "d_block": trial.suggest_categorical("d_block", d_blocks),
        "k": trial.suggest_categorical("k", [8, 16, 32]),
        "activation": trial.suggest_categorical("activation", activations),
        "llh_radius": trial.suggest_categorical("llh_radius", llh_radii),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
    }
    params.update(_sample_piecewise_params(trial))
    return params


def sample_mlp_params(trial) -> dict:
    params = {
        "layer_n": trial.suggest_int("layer_n", 1, 4),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "activation": trial.suggest_categorical("activation", ["ReLU", "SiLU", "SELU"]),
        "llh_radius": trial.suggest_categorical("llh_radius", [1, 2, 3, 4, 5]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
        "normalization": trial.suggest_categorical("normalization", [False, True]),
    }
    params.update(_sample_piecewise_params(trial))
    return params


def sample_realmlp_params(trial) -> dict:
    return {
        "n_hidden_layers": trial.suggest_int("n_hidden_layers", 1, 4),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "activation": trial.suggest_categorical("activation", ["Mish", "SELU"]),
        "llh_radius": trial.suggest_categorical("llh_radius", [1, 2, 3, 4, 5]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
        "n_frequencies": trial.suggest_categorical("n_frequencies", [32, 48, 64]),
        "emb_dim_rtdl": trial.suggest_categorical("emb_dim_rtdl", [8, 12, 16]),
        "frequency_scale": trial.suggest_float("frequency_scale", 1e-2, 1.0, log=True),
    }


def sample_tabm_params(trial) -> dict:
    params = _sample_tabm_common(
        trial, [128, 256, 512], ["ReLU", "SiLU", "SELU"], [1, 2, 3, 4, 5]
    )
    params["dropout"] = trial.suggest_float("dropout", 1e-2, 1e-1, log=True)
    return params


def sample_weibull_params(trial, model_name: str) -> dict:
    d_blocks = [64, 128, 256] if model_name == "weibull_params" else [128, 256, 512]
    return _sample_tabm_common(trial, d_blocks, ["ReLU", "SELU"], [1, 3, 5])


def sample_deepsurv_params(trial) -> dict:
    return {
        "n_hid_layers": trial.suggest_int("n_hid_layers", 1, 4),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "activation": trial.suggest_categorical("activation", ["ReLU", "Tanh"]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 1e-2, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "norm": trial.suggest_categorical("norm", [False, True]),
    }


def sample_deephit_params(trial) -> dict:
    params = sample_deepsurv_params(trial)
    params.update(
        alpha=trial.suggest_float("alpha", 0.0, 1.0),
        sigma=trial.suggest_categorical("sigma", [1e-3, 1e-2, 1e-1, 1.0]),
    )
    return params


def sample_survtrace_params(trial) -> dict:
    return {
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 4),
        "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
        "intermediate_size": trial.suggest_categorical("intermediate_size", [128, 256, 512]),
        "num_attention_heads": trial.suggest_categorical("num_attention_heads", [2, 4, 8]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 1e-2, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
    }


def sample_rsf_params(trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [20, 50, 100, 200]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, None]),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 0.01, 0.05]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def sample_coxph_params(trial) -> dict:
    return {
        # CoxPH uses an L2 penalty. A log domain covers an effectively
        # unregularized fit as well as strongly regularized, ill-conditioned data.
        "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True),
        "ties": trial.suggest_categorical("ties", ["breslow", "efron"]),
    }


def sample_gbm_params(trial) -> dict:
    return {
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [10, 25, 50, 100]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4, 5, 6, 8]),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "l2_reg": trial.suggest_float("l2_reg", 1e-4, 10.0, log=True),
        "l1_reg": trial.suggest_float("l1_reg", 1e-4, 10.0, log=True),
        "aft_loss_distribution": trial.suggest_categorical(
            "aft_loss_distribution", ["normal", "logistic", "extreme"]
        ),
        "aft_loss_distribution_scale": trial.suggest_float(
            "aft_loss_distribution_scale", 0.1, 10.0, log=True
        ),
    }


def sample_gbm_km_params(trial) -> dict:
    params = sample_gbm_params(trial)
    params["n_neighbours"] = trial.suggest_int("n_neighbours", 10, 100, step=10)
    return params


def sample_gbm_wb_params(trial) -> dict:
    params = sample_gbm_params(trial)
    params["wb_pen_mode"] = trial.suggest_categorical("wb_pen_mode", [False, True])
    params["wb_penalty"] = (
        trial.suggest_float("wb_penalty", 1e-2, 10.0, log=True)
        if params["wb_pen_mode"]
        else 0.0
    )
    params["wb_l1_ratio"] = trial.suggest_float("wb_l1_ratio", 0.0, 1.0)
    return params


def get_trial_params(model_name: Model, trial) -> dict:
    match model_name:
        case "mlp":
            return sample_mlp_params(trial)
        case "realmlp":
            return sample_realmlp_params(trial)
        case "tabm":
            return sample_tabm_params(trial)
        case "weibull_params" | "weibull_proba":
            return sample_weibull_params(trial, model_name)
        case "deepsurv":
            return sample_deepsurv_params(trial)
        case "deephit":
            return sample_deephit_params(trial)
        case "survtrace":
            return sample_survtrace_params(trial)
        case "coxph":
            return sample_coxph_params(trial)
        case "rsf":
            return sample_rsf_params(trial)
        case "gbm_km":
            return sample_gbm_km_params(trial)
        case "gbm_wb":
            return sample_gbm_wb_params(trial)
        case _:
            raise NotImplementedError(f"Unknown model: {model_name}")


def resolve_params(model_name: Model, params: dict) -> dict:
    params = params.copy()
    if model_name == "gbm_wb":
        params.pop("wb_pen_mode", None)
        params["num_boost_round"] = NUM_BOOST_ROUND
    elif model_name == "gbm_km":
        params["n_neighbours"] = int(params["n_neighbours"])
        params["num_boost_round"] = NUM_BOOST_ROUND
    return params


def evaluate_model_metrics(
    model_name: Model,
    params: dict,
    data: dict,
    device: str,
    seed: int,
) -> dict[str, float]:
    x_num_train = data["X_num_train"]
    x_cat_train = data["X_cat_train"]
    y_train = data["y_train"]

    model = get_model(
        model_name=model_name,
        config=params,
        X_num=x_num_train,
        X_cat=x_cat_train,
        y_train=y_train,
        time_ruler=np.unique(y_train["time"]),
        device=device,
        seed=seed,
    )
    train_idx, val_idx = train_test_split(
        np.arange(y_train.shape[0]),
        test_size=0.25,
        stratify=y_train["censor"],
        random_state=seed,
    )
    model.fit(
        x_num_train[train_idx],
        x_cat_train[train_idx],
        y_train[train_idx],
        (x_num_train[val_idx], x_cat_train[val_idx], y_train[val_idx]),
    )
    c_index, ibs, auc = model.score(data["X_num_test"], data["X_cat_test"], data["y_test"])
    return {
        "c_index": float(c_index),
        "ibs": float(ibs),
        "auc": float(auc),
    }


def evaluate_model(model_name: Model, trial, data, device: str, metric: Metric, seed: int) -> float:
    params = resolve_params(model_name, get_trial_params(model_name, trial))
    return evaluate_model_metrics(model_name, params, data, device, seed)[metric]


def get_objective(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.recarray,
    k_all: int,
    k_stop: int,
    model_name: Model,
    device: str,
    metric: Metric,
    seed: int,
) -> Callable:
    if not 1 <= k_stop <= k_all:
        raise ValueError(f"k_stop must be in range [1, {k_all}]")
    objective = partial(evaluate_model, model_name=model_name)
    return partial(
        k_fold_wrapper(objective, k_stop, k_all),
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        metric=metric,
        device=device,
        seed=seed,
    )
