from ..model.tabm import TabMBaseline
from ..model.loss import CIndexLoss, LikelihoodLoss, SoftLlhLoss
from ..model.model import SurvMLP
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from sklearn.model_selection import train_test_split

import optuna
import mlflow
from functools import cache
import numpy as np
from pathlib import Path
import torch
import json
import argparse

DEVICE = 'cpu'
DS_NAME = None

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
    data = np.load(f'data/{DS_NAME}.npz')
    X_num_train = data['X_num_train']
    X_num_val = data['X_num_val']
    X_cat_train, cat_mask = delete_const_cats(data['X_cat_train'])
    X_cat_val = delete_const_cats(data['X_cat_val'], cat_mask)
    y_train = data['y_train']
    y_val = data['y_val']
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val

def objective_non_rtdl(trial):
        
    X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val = get_ds()
    time_ruler = np.unique(y_train['time'])

    n_blocks = trial.suggest_int("n_blocks", 1, 3)
    d_block = trial.suggest_categorical("d_block", [128, 256, 512])
    k = trial.suggest_categorical("k", [1])
    activation = trial.suggest_categorical("activation", ["ReLU", "SiLU", "SELU"])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
    dropout = trial.suggest_loguniform("dropout", 1e-2, 1e-1)

    rtdl_embeddings, rtdl_dim = None, None
    loss_fn = LikelihoodLoss(DEVICE, time_ruler)
    
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)

        mlp = TabMBaseline(
            n_num_features=X_num_train.shape[1],
            cat_cardinalities=[2 for _ in range(X_cat_train.shape[1])],
            output_dim=time_ruler.shape[0],
            n_blocks=n_blocks,
            d_block=d_block,
            k=k,
            dropout=dropout,
            activation=activation,
            num_embeddings=rtdl_embeddings
        )
        
        model = SurvMLP(
            device=DEVICE,
            baseline=mlp,
            time_ruler=time_ruler,
            loss=loss_fn,
            epochs_n=500,
            val_period=1,
            batch_size=batch_size,
            optimizer='AdamW',
            optimizer_params={'lr': lr},
            verbose=0,
            patience=20
        )

        X_num_ptrain, X_num_pval, X_cat_ptrain, X_cat_pval, y_ptrain, y_pval = train_test_split(X_num_train, X_cat_train, y_train,
                                                        test_size=0.25, stratify=y_train['censor'])
        val_set = (X_num_pval, X_cat_pval, y_pval)
        # val_set = None
        model.fit(X_num_ptrain, X_cat_ptrain, y_ptrain, val_set)

        c_index, ibs, auc = model.score(X_num_val, X_cat_val, y_val)
        mlflow.log_metric('ibs', ibs)
        mlflow.log_metric('auc', auc)
        mlflow.log_metric("c_index", c_index)

        return c_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-device', type=str, required=False, default='cpu')
    args = parser.parse_args()
    DS_NAME = args.data
    DEVICE = args.device
    torch.set_num_threads(8)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.create_experiment("optuna_mlp_veterans")
    mlflow.set_experiment(f"optuna_ablation_{DS_NAME}")
        
    with mlflow.start_run(run_name="ablation_non_rtdl"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_non_rtdl, n_trials=80)


        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_c_index", study.best_value)

        print("Best trial:")
        print(study.best_trial.params)
        print(study.best_trial.value)

        path = Path(f'pkg/params_selection/params/ablation/{DS_NAME}')
        filename = path / 'tabm_empty.json'
        with open(filename, 'w') as file:
            json.dump(study.best_trial.params, file)
        print('Saved best params to', filename)
