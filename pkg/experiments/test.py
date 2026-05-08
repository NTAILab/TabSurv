from models.py_cox import DeepHit, DeepSurv
from models.sk_surv import CoxNet, RSF
from models.transformer import SurvTRACE
from models.base import CompetitiveModelBase
import numpy as np
from functools import cache

@cache
def load_ds():
    data = np.load('data/METABRICK.npz')
    X_num_train = data['X_num_train']
    X_num_val = data['X_num_val']
    X_cat_train = data['X_cat_train']
    X_cat_val = data['X_cat_val']
    y_train = data['y_train']
    y_val = data['y_val']
    X_num_test = data['X_num_test']
    X_cat_test = data['X_cat_test']
    y_test = data['y_test']
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val, X_num_test, X_cat_test, y_test

def test_model(model_cls: type[CompetitiveModelBase], **params):
    model = model_cls(**params)
    X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val, X_num_test, X_cat_test, y_test = load_ds()
    model.fit(X_num_train, X_cat_train, y_train, (X_num_val, X_cat_val, y_val))
    c_ind, ibs, auc = model.score(X_num_test, X_cat_test, y_test)
    print('C-index:', c_ind)
    print('IBS:', ibs)
    print('AUC:', auc)
    
def main():
    X_num_train, X_cat_train, y_train, *_ = load_ds()
    time_ruler = np.unique(y_train['time'])
    models = {
        CoxNet: {'alpha': 0.01, 'l1_ratio': 0.5, 'normalize': True},
        RSF: {'n_estimators': 10, 'max_depth': 4, 
                 'min_samples_split': 2, 'max_features': 'sqrt'},
        DeepHit: {'device': 'cpu', 'x_num_dim': X_num_train.shape[1],
                 'x_cat_dim': X_cat_train.shape[1], 'time_ruler': time_ruler,
                 'n_hid_layers': 3, 'hidden_dim': 32,
                 'norm': True, 'activation': 'ReLU', 'dropout': 0.0,
                 'lr': 1e-3, 'batch_size': 128, 'epochs': 500,
                 'verbose': False},
        DeepSurv: {'device': 'cpu', 'x_num_dim': X_num_train.shape[1],
                 'x_cat_dim': X_cat_train.shape[1],
                 'n_hid_layers': 3, 'hidden_dim': 32,
                 'norm': True, 'activation': 'ReLU', 'dropout': 0.0,
                 'lr': 1e-3, 'batch_size': 128, 'epochs': 500,
                 'verbose': False},
        SurvTRACE: {'device_id': None, 'time_ruler': time_ruler, 'hidden_size': 64,
                 'intermediate_size': 128, 'num_hidden_layers': 2, 'num_attention_heads': 4,
                 'hidden_dropout_prob': 0.1, 'early_stop': 10, 'batch_size': 64, 'epochs': 500,
                 'lr': 1e-3, 'weight_decay': 1e-3, 'verbose': True}
        
    }
    for model_cls, params in models.items():
        print(f'--- Testing {model_cls.__name__} ---')
        test_model(model_cls, **params)
    
if __name__=='__main__':
    main()
