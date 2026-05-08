from .exp_util import do_time_experiment, dict_to_df
import pandas as pd
import json
import argparse
from pathlib import Path
from torch import set_num_threads

if __name__=='__main__':
    set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-device', type=str, required=True)
    parser.add_argument('-iters', type=int, required=False, default=20)
    args = parser.parse_args()
    ds_name = args.data
    device = args.device
    iters = args.iters
    path = Path(f'pkg/experiments/results/{ds_name}')
    path.mkdir(exist_ok=True, parents=True)
    results = do_time_experiment(ds_name, iters, device=device)
    with open(path / f'{ds_name}_time.json', 'w') as file:
        json.dump(results, file)
    df_wt = dict_to_df(results['walltime'], 'Walltime', False)
    print(df_wt)
    df_cpu = dict_to_df(results['cpu-time'], 'CPU', False)
    print(df_cpu)
    df_gpu = dict_to_df(results['gpu-time'], 'GPU', False)
    print(df_gpu)
