from ..experiments.models.py_cox import DeepHit
from sklearn.model_selection import train_test_split

import optuna
import mlflow
from functools import cache
import numpy as np
from torch import set_num_threads
import argparse
import json

DS_NAME = None

@cache
def get_ds():
    data = np.load(f'data/{DS_NAME}.npz')
    X_num_train = data['X_num_train']
    X_num_val = data['X_num_val']
    X_cat_train = data['X_cat_train']
    X_cat_val = data['X_cat_val']
    y_train = data['y_train']
    y_val = data['y_val']
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val

def objective(trial):
    X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val = get_ds()
    time_ruler = np.unique(y_train['time'])

    n_hid_layers = trial.suggest_int("n_hid_layers", 1, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    activation = trial.suggest_categorical("activation", ["ReLU", "Tanh"])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    dropout = trial.suggest_loguniform("dropout", 1e-2, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    norm = trial.suggest_categorical("norm", [False, True])
    alpha = trial.suggest_float("alpha", 0, 1)
    sigma = trial.suggest_categorical("sigma", [1e-3, 1e-2, 1e-1, 1])
    

    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        model = DeepHit(
            device='cpu',
            x_num_dim=X_num_train.shape[1],
            x_cat_dim=X_cat_train.shape[1],
            time_ruler=time_ruler,
            n_hid_layers=n_hid_layers,
            hidden_dim=hidden_dim,
            norm=norm,
            activation=activation,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=200,
            alpha=alpha,
            sigma=sigma
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
    args = parser.parse_args()
    DS_NAME = args.data
    set_num_threads(8)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.create_experiment("optuna_mlp_veterans")
    mlflow.set_experiment(f"optuna_deephit_{DS_NAME}")
    
    with mlflow.start_run(run_name="deephit"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=150)

        
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_c_index", study.best_value)

        print("Best trial:")
        print(study.best_trial.params)
        print(study.best_trial.value)

        filename = f'pkg/params_selection/params/{DS_NAME}/deephit.json' 
        with open(filename, 'w') as file:
            json.dump(study.best_trial.params, file)
        print('Saved best params to', filename)
