from ..experiments.models.transformer import SurvTRACE
from sklearn.model_selection import train_test_split

import optuna
import mlflow
from functools import cache
import numpy as np
from torch import set_num_threads, device
import json
import argparse

DEVICE = 'cpu'
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

    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    intermediate_size = trial.suggest_categorical("intermediate_size", [128, 256, 512])
    num_attention_heads = trial.suggest_categorical("num_attention_heads", [2, 4, 8])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    hidden_dropout_prob = trial.suggest_loguniform("hidden_dropout_prob", 1e-2, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
    

    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        if DEVICE == 'cpu':
            device_id = None
        else:
            device_id = device(DEVICE).index
            device_id = 0 if device_id is None else device_id
        model = SurvTRACE(
            device_id=device_id,
            time_ruler=time_ruler,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            early_stop=10,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            epochs=200,
            verbose=False
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
    set_num_threads(8)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.create_experiment("optuna_mlp_veterans")
    mlflow.set_experiment(f"optuna_survtrace_{DS_NAME}")
    
    with mlflow.start_run(run_name="survtrace"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_c_index", study.best_value)

        print("Best trial:")
        print(study.best_trial.params)
        print(study.best_trial.value)

        filename = f'pkg/params_selection/params/{DS_NAME}/survtrace.json' 
        with open(filename, 'w') as file:
            json.dump(study.best_trial.params, file)
        print('Saved best params to', filename)
