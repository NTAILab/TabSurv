from ..experiments.models.sk_surv import RSF

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

    n_estimators = trial.suggest_categorical("n_estimators", [20, 50, 100, 200])
    max_depth = trial.suggest_categorical("max_depth", [3, 5, None])
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 0.01, 0.05])
    max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2'])
    
    
    with mlflow.start_run(nested=True):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("max_features", max_features)
        model = RSF(
            time_ruler=np.unique(y_train['time']),
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features
        )

        # X_num_ptrain, X_num_pval, X_cat_ptrain, X_cat_pval, y_ptrain, y_pval = train_test_split(X_num_train, X_cat_train, y_train,
        #                                                 test_size=0.25, stratify=y_train['censor'])
        # val_set = None
        model.fit(X_num_train, X_cat_train, y_train)
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
    mlflow.set_experiment(f"optuna_rsf_{DS_NAME}")
    
    with mlflow.start_run(run_name="rsf"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_c_index", study.best_value)

        print("Best trial:")
        print(study.best_trial.params)
        print(study.best_trial.value)
        
        filename = f'pkg/params_selection/params/{DS_NAME}/rsf.json' 
        with open(filename, 'w') as file:
            json.dump(study.best_trial.params, file)
        print('Saved best params to', filename)
