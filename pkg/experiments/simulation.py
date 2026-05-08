import numpy as np
from sklearn.cluster import KMeans
from .exp_util import get_ds, gen_gompetz, gen_cens_time
from .exp_util import do_sim_experiment
import json
import argparse
from pathlib import Path
from torch import set_num_threads
import matplotlib.pyplot as plt
from ..util import get_str_array

def gen_rotterdam_ds():
    seed = 42
    gen = np.random.default_rng(seed)
    data = get_ds('rotterdam')
    feat_num_n = data['X_num_train'].shape[1]
    feat_cat_n = data['X_cat_train'].shape[1]
    train_n = data['X_num_train'].shape[0]
    test_n = data['X_num_test'].shape[0]
    val_n = data['X_num_val'].shape[0]
    
    X_train = np.concat((data['X_num_train'], data['X_cat_train']), axis=-1)
    X_test = np.concat((data['X_num_test'], data['X_cat_test']), axis=-1)
    X_val = np.concat((data['X_num_val'], data['X_cat_val']), axis=-1)
    X = np.concat((X_train, X_val, X_test), axis=0)
    
    labels = gen.binomial(1, 0.5, X.shape[0])
    mask_1 = labels == 0
    mask_2 = labels == 1
    print('Partitions per clusters:')
    print(f'\tOverall: {mask_1.mean()}/{mask_2.mean()}')
    print(f'\tTrain: {mask_1[:train_n].mean()}/{mask_2[:train_n].mean()}')
    print(f'\tVal: {mask_1[train_n:train_n + val_n].mean()}/{mask_2[train_n:train_n + val_n].mean()}')
    print(f'\tTest: {mask_1[train_n + val_n:].mean()}/{mask_2[train_n + val_n:].mean()}')
    
    
    X_1 = X[mask_1]
    X_2 = X[mask_2]
    
    b_1 = gen.uniform(0, 1, X.shape[1])
    lmd_1 = 1e-5
    nu_1 = 4
    b_2 = gen.uniform(-1, 1, X.shape[1])
    lmd_2 = 1e-10
    nu_2 = 6
    T_1 = gen_gompetz(X_1, lmd_1, b_1, nu_1, seed=seed)
    T_2 = gen_gompetz(X_2, lmd_2, b_2, nu_2, seed=seed)
    
    cens_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    simul_data = {
        'cens_list': np.asarray(cens_list),
        'X_num_train': data['X_num_train'],
        'X_cat_train': data['X_cat_train'],
        'X_num_val': data['X_num_val'],
        'X_cat_val': data['X_cat_val'],
        'X_num_test': data['X_num_test'],
        'X_cat_test': data['X_cat_test'],
    }
    
    T = np.empty(train_n + val_n + test_n)
    T[np.where(mask_1)] = T_1
    T[np.where(mask_2)] = T_2
    
    t_train = T[:train_n]
    t_val = T[train_n:train_n + val_n]
    t_test = T[train_n + val_n:]
    assert t_test.shape[0] == test_n
    c_train = np.zeros((train_n, len(cens_list)))
    c_val = np.zeros((val_n, len(cens_list)))
    c_test = np.zeros((test_n, len(cens_list)))
    for i, c in enumerate(cens_list):
        C_1 = gen_cens_time(T_1, c, seed)
        C_2 = gen_cens_time(T_2, c, seed)
        C = np.zeros(train_n + val_n + test_n)
        C[np.where(mask_1)] = C_1
        C[np.where(mask_2)] = C_2
        # C = gen_cens_time(T, cens_part=c)
        c_train[:, i] = C[:train_n]
        c_val[:, i] = C[train_n:train_n + val_n]
        c_test[:, i] = C[train_n + val_n:]
        print(f'For requested cens part {c}:')
        print(f'\tTrain part: {(c_train[:, i] < t_train).mean()}')
        print(f'\tVal part: {(c_val[:, i] < t_val).mean()}')
        print(f'\tTest part: {(c_test[:, i] < t_test).mean()}')
    simul_data['t_train'] = t_train
    simul_data['t_val'] = t_val
    simul_data['t_test'] = t_test
    simul_data['c_train'] = c_train
    simul_data['c_val'] = c_val
    simul_data['c_test'] = c_test
    simul_data['idx_0'] = np.argwhere(mask_1).ravel()
    simul_data['idx_1'] = np.argwhere(mask_2).ravel()
    plt.hist(T_1, 100)
    plt.hist(T_2, 100)
    plt.show()
    np.savez('data/sim_rotterdam_full.npz', **simul_data)
    def get_y_time(t, c):
        y = np.stack((t, c), -1).min(-1)
        c = t < c
        return get_str_array(y, c)
    simul_data_slice = {
        'X_num_train': data['X_num_train'],
        'X_cat_train': data['X_cat_train'],
        'X_num_val': data['X_num_val'],
        'X_cat_val': data['X_cat_val'],
        'X_num_test': data['X_num_test'],
        'X_cat_test': data['X_cat_test'],
        'y_train': get_y_time(t_train, c_train[:, 3]),
        'y_val': get_y_time(t_val, c_val[:, 3]),
        'y_test': get_y_time(t_test, c_test[:, 3])
    }
    np.savez('data/sim_rotterdam.npz', **simul_data_slice)

def do_experiment():
    set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, nargs='+', required=False, default=None)
    parser.add_argument('-iters', type=int, required=False, default=20)
    parser.add_argument('-n_workers', type=int, required=False, default=1)
    args = parser.parse_args()
    device = args.device
    iters = args.iters
    n_workers = args.n_workers
    if device is not None and len(device) > n_workers:
        n_workers = len(device)
    ds_name = 'sim_rotterdam'
    path = Path(f'pkg/experiments/results/{ds_name}')
    path.mkdir(exist_ok=True, parents=True)
    results = do_sim_experiment(iters, device, n_workers)
    with open(path / f'{ds_name}_results.json', 'w') as file:
        json.dump(results, file)

if __name__=='__main__':
    do_experiment()
    # gen_rotterdam_ds()
