from __future__ import annotations

import argparse
import gc
from functools import partial

import mlflow
import numpy as np
import optuna
import torch
from lifelines.exceptions import ConvergenceError
from sklearn.model_selection import StratifiedKFold

from .comparison import OPTUNA_FOLDS, set_seed
from .data import load_data
from .objective_factory import (
    Metric,
    evaluate_model_metrics,
    get_metrics_list,
    resolve_params,
    sample_mlp_params,
)


LOSSES = ("c-index", "likelihood", "soft-llh")
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def sample_mlp_loss_params(trial: optuna.Trial, loss: str) -> dict:
    if loss not in LOSSES:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {LOSSES}")
    params = sample_mlp_params(trial)
    # Loss allocation is controlled by the outer loop, not by Optuna. This gives
    # every loss exactly the same trial budget and an independent TPE history.
    params["loss"] = loss
    return resolve_params("mlp", params)


def _loss_key(loss: str) -> str:
    return loss.replace("-", "_")


def _loss_sampler_seeds(seed: int) -> dict[str, int]:
    sequences = np.random.SeedSequence(seed).spawn(len(LOSSES))
    return {
        loss: int(sequence.generate_state(1, dtype=np.uint32)[0])
        for loss, sequence in zip(LOSSES, sequences)
    }


def evaluate_loss_trial(
    trial: optuna.Trial,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.recarray,
    device: str,
    metric: Metric,
    k_stop: int,
    seed: int,
    loss: str,
) -> float:
    if not 1 <= k_stop <= OPTUNA_FOLDS:
        raise ValueError(f"k_optuna must be in range [1, {OPTUNA_FOLDS}]")

    with mlflow.start_run(
        nested=True,
        run_name=f"{_loss_key(loss)}_trial_{trial.number}",
    ):
        params = sample_mlp_loss_params(trial, loss)
        mlflow.log_params(params)
        mlflow.set_tags({
            "trial_number": trial.number,
            "model": "mlp",
            "loss": loss,
        })

        splitter = StratifiedKFold(
            n_splits=OPTUNA_FOLDS,
            shuffle=True,
            random_state=seed,
        )
        scores = {name: [] for name in get_metrics_list()}
        folds = splitter.split(X_num, y["censor"])
        for fold_i, (train_idx, val_idx) in zip(range(k_stop), folds):
            data = {
                "X_num_train": X_num[train_idx],
                "X_cat_train": X_cat[train_idx],
                "y_train": y[train_idx],
                # The objective fold is validation data. There is no test stage.
                "X_num_test": X_num[val_idx],
                "X_cat_test": X_cat[val_idx],
                "y_test": y[val_idx],
            }
            fold_scores = evaluate_model_metrics(
                model_name="mlp",
                params=params,
                data=data,
                device=device,
                seed=seed,
            )
            for name, value in fold_scores.items():
                scores[name].append(value)
                mlflow.log_metric(f"fold_{name}", value, step=fold_i)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_scores = {
            f"val_{name}": float(np.mean(values))
            for name, values in scores.items()
        }
        mlflow.log_metrics(mean_scores)
        objective_value = mean_scores[f"val_{metric}"]
        mlflow.log_metric("objective", objective_value)
        return objective_value


def run_search(
    dataset: str,
    device: str,
    trials_per_loss: int,
    k_optuna: int,
    seed: int,
    val_metric: Metric,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> dict[str, optuna.Study]:
    if trials_per_loss <= 0:
        raise ValueError("trials_per_loss must be positive")
    if not 1 <= k_optuna <= OPTUNA_FOLDS:
        raise ValueError(f"k_optuna must be in range [1, {OPTUNA_FOLDS}]")

    data = load_data(dataset)
    set_seed(seed)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name or f"optuna_mlp_loss_{dataset}")

    direction = "minimize" if val_metric == "ibs" else "maximize"
    sampler_seeds = _loss_sampler_seeds(seed)
    studies: dict[str, optuna.Study] = {}

    with mlflow.start_run(run_name=run_name or f"mlp_loss_{val_metric}"):
        mlflow.log_params({
            "dataset": dataset,
            "device": device,
            "trials_per_loss": trials_per_loss,
            "total_trials": trials_per_loss * len(LOSSES),
            "k_optuna": k_optuna,
            "seed": seed,
            "val_metric": val_metric,
        })
        for loss in LOSSES:
            loss_key = _loss_key(loss)
            sampler_seed = sampler_seeds[loss]
            sampler = optuna.samplers.TPESampler(seed=sampler_seed)
            study = optuna.create_study(direction=direction, sampler=sampler)
            studies[loss] = study
            objective = partial(
                evaluate_loss_trial,
                X_num=data["X_num"],
                X_cat=data["X_cat"],
                y=data["y"],
                device=device,
                metric=val_metric,
                k_stop=k_optuna,
                seed=seed,
                loss=loss,
            )
            mlflow.log_param(f"{loss_key}_sampler_seed", sampler_seed)
            study.optimize(
                objective,
                n_trials=trials_per_loss,
                show_progress_bar=True,
                catch=(ConvergenceError, ValueError, TypeError),
            )

            best_params = {
                f"best_{loss_key}_{name}": value
                for name, value in study.best_trial.params.items()
            }
            best_params[f"best_{loss_key}_loss"] = loss
            mlflow.log_params(best_params)
            mlflow.log_metric(
                f"best_val_{val_metric}_{loss_key}",
                study.best_value,
            )
            mlflow.set_tag(
                f"best_trial_number_{loss_key}",
                study.best_trial.number,
            )

    for loss, study in studies.items():
        print(f"Best trial for {loss}:")
        print({**study.best_trial.params, "loss": loss})
        print(study.best_value)
    return studies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna MLP loss search with MLflow validation logging"
    )
    parser.add_argument("-data", required=True)
    parser.add_argument("-device", default="cpu")
    parser.add_argument(
        "-trials_per_loss",
        "--trials-per-loss",
        "-trials",
        dest="trials_per_loss",
        type=int,
        default=50,
        help="number of trials in each independent loss-specific study",
    )
    parser.add_argument("-k_optuna", type=int, default=1)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-threads", type=int, default=8)
    parser.add_argument(
        "-val_metric",
        default="c_index",
        choices=get_metrics_list(),
    )
    parser.add_argument(
        "-tracking_uri",
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
    )
    parser.add_argument("-experiment_name", "--experiment-name", default=None)
    parser.add_argument("-run_name", "--run-name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(args.threads)
    run_search(
        dataset=args.data,
        device=args.device,
        trials_per_loss=args.trials_per_loss,
        k_optuna=args.k_optuna,
        seed=args.seed,
        val_metric=args.val_metric,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
