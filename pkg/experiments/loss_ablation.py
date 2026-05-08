from .exp_util import get_configs, get_ds, get_model
from typing import List
import matplotlib.pyplot as plt
from .paper_format import ds_rename
import numpy as np
import json
from tqdm import tqdm

def do_sigma_experiment(ds_name: str, model_name: str,
                        steps_part_list: List[float],
                        sigma_list: List[int],
                        repeat_n: int,
                        device: str):
    data = get_ds(ds_name)
    X_num_train = data['X_num_train']
    X_cat_train = data['X_cat_train']
    y_train = data['y_train']
    X_num_val = data['X_num_val']
    X_cat_val = data['X_cat_val']
    y_val = data['y_val']
    X_num_test = data['X_num_test']
    X_cat_test = data['X_cat_test']
    y_test = data['y_test']
    config = get_configs(ds_name, [model_name])[model_name]
    results = {
        'sigma_list': sigma_list,
        'steps_part_list': steps_part_list,
        'c_index': [],
        'ibs': [],
        'auc': []
    }
    time_ruler_all = np.unique(y_train['time'])
    for i, part in enumerate(steps_part_list):
        n = int(time_ruler_all.shape[0] * part)
        if part == 1:
            time_ruler = time_ruler_all.copy()
        else:
            quantiles = np.linspace(0, 1.0, num=n)
            time_ruler = np.unique(np.quantile(time_ruler_all, quantiles))
        part_c_ind_list = []
        part_ibs_list = []
        part_auc_list = []
        bar = tqdm(sigma_list, desc=f"Part {i + 1}/{len(steps_part_list)} | ")
        for sigma in bar:
            config['llh_radius'] = sigma
            cur_c_index_list = []
            cur_ibs_list = []
            cur_auc_list = []
            for _ in range(repeat_n):
                model = get_model(
                    model_name=model_name,
                    config=config,
                    X_num=X_num_train,
                    X_cat=X_cat_train,
                    y_train=y_train,
                    time_ruler=time_ruler,
                    device=device
                )
                model.fit(X_num_train, X_cat_train, y_train, (X_num_val, X_cat_val, y_val))
                c_index, ibs, auc = model.score(X_num_test, X_cat_test, y_test)
                cur_c_index_list.append(c_index)
                cur_ibs_list.append(ibs)
                cur_auc_list.append(auc)
            part_c_ind_list.append(cur_c_index_list)
            part_ibs_list.append(cur_ibs_list)
            part_auc_list.append(cur_auc_list)
        results['c_index'].append(part_c_ind_list)
        results['ibs'].append(part_ibs_list)
        results['auc'].append(part_auc_list)
    with open(f'pkg/experiments/results/{ds_name}_loss.json', 'w') as file:
         json.dump(results, file)
    return results

def draw_figure(exp_results, ticks, title):
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    x = exp_results['sigma_list']
    part_list = exp_results['steps_part_list']
    for i, p in enumerate(part_list):
        y = np.mean(exp_results['c_index'][i], axis=-1)
        ax.plot(x, y, label=f"{int(p * 100)}%")
    ax.set_xlabel('r')
    ax.set_ylabel('C-index')
    ax.set_title(title)
    ax.set_xticks(ticks)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig, ax
