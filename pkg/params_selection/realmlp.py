from ..model.realmlp import RealMLP, PBLDEmbeddings
from ..model.loss import CIndexLoss, LikelihoodLoss, SoftLlhLoss
from ..model.model import SurvMLP
from sklearn.model_selection import train_test_split

import optuna
import mlflow
from functools import cache
import numpy as np
import json
from torch import set_num_threads
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

def objective(trial):
    
    def make_rtdl(X_num, y_train):
        n_frequencies = trial.suggest_categorical("n_frequencies", [32, 48, 64])
        emb_dim_rtdl = trial.suggest_categorical("emb_dim_rtdl", [8, 12, 16])
        frequency_scale = trial.suggest_loguniform("frequency_scale", 1e-2, 1)
        num_embeddings = PBLDEmbeddings(X_num.shape[1], n_frequencies, frequency_scale, emb_dim_rtdl)
        return num_embeddings, emb_dim_rtdl
    
    def make_loss(loss_name, time_ruler, device):
        match loss_name:
            case 'c-index':
                return CIndexLoss(device, time_ruler, sigma_temp=1)
            case 'likelihood':
                return LikelihoodLoss(device, time_ruler)
            case 'soft-llh':
                return SoftLlhLoss(device, time_ruler, radius=3)
            case _:
                raise NotImplementedError()
        
    X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val = get_ds()
    time_ruler = np.unique(y_train['time'])

    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    activation = trial.suggest_categorical("activation", ["Mish", "SELU"])
    llh_radius = trial.suggest_categorical("llh_radius", [1, 2, 3, 4, 5])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])

    rtdl_embeddings, rtdl_dim = make_rtdl(X_num_train, y_train)
    loss_fn = SoftLlhLoss(DEVICE, time_ruler, radius=llh_radius)
    
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)

        mlp = RealMLP(
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            hidden_dim=hidden_dim,
            output_dim=time_ruler.shape[0],
            rtdl_embeddings=rtdl_embeddings,
            rtdl_dim=rtdl_dim,
            num_dim=X_num_train.shape[1],
            cat_dim=X_cat_train.shape[1]
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
    set_num_threads(8)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.create_experiment("optuna_mlp_veterans")
    mlflow.set_experiment(f"optuna_realmlp_{DS_NAME}")
    
    with mlflow.start_run(run_name="realmlp_rtdl"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200)

        
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_c_index", study.best_value)

        print("Best trial:")
        print(study.best_trial.params)
        print(study.best_trial.value)

        filename = f'pkg/params_selection/params/{DS_NAME}/realmlp.json' 
        with open(filename, 'w') as file:
            json.dump(study.best_trial.params, file)
        print('Saved best params to', filename)