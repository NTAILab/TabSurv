from .exp_util import do_experiment, dict_to_df
import pandas as pd
import json
import argparse
from pathlib import Path
from torch import set_num_threads

if __name__=='__main__':
    set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-device', type=str, required=False, default='cpu')
    parser.add_argument('-iters', type=int, required=False, default=20)
    args = parser.parse_args()
    ds_name = args.data
    device = args.device
    iters = args.iters
    path = Path(f'pkg/experiments/results/{ds_name}')
    path.mkdir(exist_ok=True, parents=True)
    results = do_experiment(ds_name, iters, device=device)
    with open(path / f'{ds_name}_results.json', 'w') as file:
        json.dump(results, file)
    df_c_ind = dict_to_df(results['c-index'], 'C-index')
    print(df_c_ind)
    df_ibs = dict_to_df(results['ibs'], 'IBS', bigger_is_better=False)
    print(df_ibs)
    df_auc = dict_to_df(results['auc'], 'AUC')
    print(df_auc)
    with pd.ExcelWriter(str(path / f'{ds_name}_results.xlsx')) as writer:
        df_c_ind.to_excel(writer, sheet_name="C-index")
        df_ibs.to_excel(writer, sheet_name="IBS")
        df_auc.to_excel(writer, sheet_name="AUC")
