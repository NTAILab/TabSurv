from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
from lifelines.exceptions import ConvergenceError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import load_data
from .exp_util import get_model
from .objective_factory import (
    Model,
    get_models_list,
    get_trial_params,
    resolve_params,
)


SEED_MAX = 2**20
TEST_FOLDS = 4
OPTUNA_FOLDS = 4
DEFAULT_TABLE = Path("pkg/experiments/results/comparison_table.csv")


@dataclass(frozen=True)
class Result:
    model_name: str
    c_index: float
    ibs: float
    auc: float
    time: float
    repeat_i: int
    repeat_seed: int
    fold_i: int
    fold_seed: int
    params: str
    ds_name: str
    outer_seed: int

    @classmethod
    def header(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def get_logger(name: str, filename: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if filename is not None:
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_repeats_spec(spec: list[str] | None, repeats: int) -> set[int] | None:
    if spec is None:
        return None
    selected = {int(value) for value in spec}
    if any(value <= 0 or value > repeats for value in selected):
        raise ValueError(f"repeats_spec must contain integers in range [1, {repeats}]")
    return selected


def initialize_table(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=Result.header(), delimiter=";").writeheader()
    return path


def append_result(path: str | Path, result: Result) -> None:
    with Path(path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Result.header(), delimiter=";")
        writer.writerow(asdict(result))


def preprocess_fold(train, *held_out):
    """Fit on raw training covariates and transform each (X_num, X_cat, y) split."""
    x_num, x_cat, _ = train
    num_columns = ~np.isnan(x_num).all(axis=0)
    numeric = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    if num_columns.any():
        numeric.fit(x_num[:, num_columns])

    def category_labels(values):
        # Raw NPZs can use NaN for numeric codes or an empty string for text.
        return np.where(pd.isna(values), "", values.astype(object)).astype(str)

    labels = category_labels(x_cat)
    cat_columns = (labels != "").any(axis=0)
    if cat_columns.any():
        labels = labels[:, cat_columns]
        # Reserve a missing indicator even if only held-out rows have missing values.
        categories = [np.union1d(column, [""]) for column in labels.T]
        encoder = OneHotEncoder(categories=categories, handle_unknown="ignore",
                                sparse_output=False, dtype=np.int64)
        encoder.fit(labels)

    def transform(split):
        num, cat, y = split
        num = num[:, num_columns]
        if num.shape[1]:
            num = numeric.transform(num)
        cat = cat[:, cat_columns]
        if cat.shape[1]:
            cat = encoder.transform(category_labels(cat))
        return num.astype(np.float32), cat.astype(np.int64, copy=False), y

    return tuple(transform(split) for split in (train, *held_out))


def split_training(X_num, X_cat, y, seed):
    train, val = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y["censor"], random_state=seed,
    )
    return (X_num[train], X_cat[train], y[train]), (X_num[val], X_cat[val], y[val])


def train_model(model_name, params, train, val, device, seed):
    X_num, X_cat, y = train
    model = get_model(
        model_name=model_name, config=params, X_num=X_num, X_cat=X_cat,
        y_train=y, time_ruler=np.unique(y["time"]), device=device, seed=seed,
    )
    started = time.perf_counter()
    model.fit(*train, val)
    return model, time.perf_counter() - started


def do_optuna_param_search(model_name, X_num, X_cat, y, device, trials, seed) -> dict:
    train, val = preprocess_fold(*split_training(X_num, X_cat, y, seed))

    def objective(trial):
        params = resolve_params(model_name, get_trial_params(model_name, trial))
        monitor = None if model_name in ("coxph", "rsf") else val
        model, _ = train_model(model_name, params, train, monitor, device, seed)
        return float(model.score(*val)[0])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        objective, n_trials=trials, show_progress_bar=True, gc_after_trial=True,
        catch=(ConvergenceError, ValueError, TypeError),
    )
    return resolve_params(model_name, study.best_trial.params)


def fit_and_score(model_name, device, X_num_train, X_cat_train, y_train,
                  X_num_test, X_cat_test, y_test, params, fold_seed):
    train = (X_num_train, X_cat_train, y_train)
    test = (X_num_test, X_cat_test, y_test)
    val = None
    if model_name in ("coxph", "rsf"):
        train, test = preprocess_fold(train, test)
    else:
        train, val = split_training(*train, seed=fold_seed)
        train, val, test = preprocess_fold(train, val, test)
    model, wall_time = train_model(model_name, params, train, val, device, fold_seed)
    c_index, ibs, auc = model.score(*test)
    return float(c_index), float(ibs), float(auc), wall_time


def do_test(
    model_name: Model,
    device: str,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.recarray,
    repeats: int,
    trials: int,
    table_filename: str | Path,
    seed: int,
    ds_name: str,
    logger_filename: str | None = None,
    repeats_spec: set[int] | None = None,
) -> None:
    if repeats <= 0 or trials <= 0:
        raise ValueError("repeats and trials must be positive")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    logger = get_logger(f"Comparison {ds_name}:{model_name}", logger_filename)
    repeat_seeds = rng.integers(1, SEED_MAX, repeats).tolist()
    logger.info("Repeat seeds: %s", repeat_seeds)
    if repeats_spec is not None:
        logger.info("Selected repeats (1-based): %s", sorted(repeats_spec))

    for repeat_i, repeat_seed in enumerate(repeat_seeds):
        repeat_ord = repeat_i + 1
        fold_seeds = rng.integers(1, SEED_MAX, TEST_FOLDS).tolist()
        if repeats_spec is not None and repeat_ord not in repeats_spec:
            logger.info("Skipping repeat %d/%d", repeat_ord, repeats)
            continue

        logger.info("Starting repeat %d/%d; fold seeds: %s", repeat_ord, repeats, fold_seeds)
        splitter = StratifiedKFold(
            n_splits=TEST_FOLDS,
            shuffle=True,
            random_state=repeat_seed,
        )
        scores = {"c_index": [], "ibs": [], "auc": []}
        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X_num, y["censor"])):
            x_num_train, x_cat_train, y_train = X_num[train_idx], X_cat[train_idx], y[train_idx]
            params = do_optuna_param_search(
                model_name=model_name,
                X_num=x_num_train,
                X_cat=x_cat_train,
                y=y_train,
                device=device,
                trials=trials,
                seed=int(np.random.SeedSequence(fold_seeds[fold_i]).generate_state(1)[0]),
            )
            fit_kwargs = dict(
                model_name=model_name,
                device=device,
                X_num_train=x_num_train,
                X_cat_train=x_cat_train,
                y_train=y_train,
                X_num_test=X_num[test_idx],
                X_cat_test=X_cat[test_idx],
                y_test=y[test_idx],
                params=params,
                fold_seed=fold_seeds[fold_i],
            )
            c_index, ibs, auc, wall_time = fit_and_score(**fit_kwargs)

            scores["c_index"].append(c_index)
            scores["ibs"].append(ibs)
            scores["auc"].append(auc)
            append_result(
                table_filename,
                Result(
                    model_name=model_name,
                    c_index=c_index,
                    ibs=ibs,
                    auc=auc,
                    time=wall_time,
                    repeat_i=repeat_i,
                    repeat_seed=repeat_seed,
                    fold_i=fold_i,
                    fold_seed=fold_seeds[fold_i],
                    params=json.dumps(params, sort_keys=True),
                    ds_name=ds_name,
                    outer_seed=seed,
                ),
            )
            logger.info(
                "Fold %d/%d | C-index %.3f | IBS %.3f | AUC %.3f",
                fold_i + 1,
                TEST_FOLDS,
                c_index,
                ibs,
                auc,
            )

        logger.info(
            "Repeat %d results | C-index %.3f | IBS %.3f | AUC %.3f",
            repeat_ord,
            np.mean(scores["c_index"]),
            np.mean(scores["ibs"]),
            np.mean(scores["auc"]),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested-CV survival model comparison")
    parser.add_argument("-data", required=True)
    parser.add_argument("-data_dir", type=Path, default=Path("data/raw"),
                        help="Directory containing raw, unprocessed unified NPZs")
    parser.add_argument("-model", required=True, choices=get_models_list())
    parser.add_argument("-trials", type=int, default=100)
    parser.add_argument("-repeats", type=int, default=10)
    parser.add_argument("-k_optuna", type=int, choices=[1], default=1)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-device", default="cpu")
    parser.add_argument("-threads", type=int, default=8)
    parser.add_argument("-repeats_spec", nargs="+", default=None)
    parser.add_argument("-val_metric", default="c_index", choices=["c_index"])
    parser.add_argument("-table_filename", default=str(DEFAULT_TABLE))
    parser.add_argument("-logger_filename", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repeats_spec = parse_repeats_spec(args.repeats_spec, args.repeats)
    torch.set_num_threads(args.threads)
    data = load_data(args.data, args.data_dir)
    table_path = initialize_table(args.table_filename)
    do_test(
        model_name=args.model,
        device=args.device,
        X_num=data["X_num"],
        X_cat=data["X_cat"],
        y=data["y"],
        repeats=args.repeats,
        trials=args.trials,
        table_filename=table_path,
        seed=args.seed,
        ds_name=args.data,
        logger_filename=args.logger_filename,
        repeats_spec=repeats_spec,
    )


if __name__ == "__main__":
    main()
