import numpy as np
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from ..model.realmlp import PBLDEmbeddings, RealMLP
from ..model.mlp import MLPBaseline
from ..model.tabm import TabMBaseline
from ..model.loss import CIndexLoss, LikelihoodLoss, SoftLlhLoss
from ..model.model import SurvMLP, SurvMLPWeibull
from .models.py_cox import DeepHit, DeepSurv
from .models.sk_surv import CoxPH, RSF
from .models.xg_boost import AFTModel, GBMKM, GBMWeibull
from .models.transformer import SurvTRACE
from typing import Dict, Iterable, List, Literal, Sequence, Tuple
from torch import from_numpy
from torch.nn import Module
from functools import cache
import pandas as pd
from tqdm import tqdm
import time
import json
from collections import defaultdict
from scipy.stats import rankdata
from scipy.optimize import bisect
from ..util import get_str_array
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import traceback
import random
from scipy.stats import wasserstein_distance, ks_2samp, cramervonmises_2samp
import torch

def get_num_emb(model_name: Literal['mlp', 'tabm', 'realmlp', 'weibull_params', 'weibull_proba'],
                model_config: dict, X_num: np.ndarray, y_train: np.recarray) -> Tuple[Module, int]:
    if 'n_frequencies' not in model_config and 'rtdl_n_bins' not in model_config:
        return None, None
    if model_name in ['mlp', 'tabm', 'weibull_params', 'weibull_proba']:
        n_bins = model_config['rtdl_n_bins']
        bins = compute_bins(from_numpy(X_num), n_bins,
                            y=from_numpy(y_train['time'].copy()),
                            regression=True, tree_kwargs={})
        emb_dim_rtdl = model_config["emb_dim_rtdl"]
        rtdl_activation = model_config["rtdl_activation"]
        num_embeddings = PiecewiseLinearEmbeddings(bins, emb_dim_rtdl, activation=rtdl_activation, version='B')
        return num_embeddings, emb_dim_rtdl
    else:
        n_frequencies = model_config["n_frequencies"]
        emb_dim_rtdl = model_config["emb_dim_rtdl"]
        frequency_scale = model_config["frequency_scale"]
        num_embeddings = PBLDEmbeddings(X_num.shape[1], n_frequencies, frequency_scale, emb_dim_rtdl)
        return num_embeddings, emb_dim_rtdl

def make_loss(loss_name: str, llh_rad: int,
              time_ruler: np.ndarray, device: str):
    match loss_name:
        case 'c-index':
            return CIndexLoss(device, time_ruler, sigma_temp=1)
        case 'likelihood':
            return LikelihoodLoss(device, time_ruler)
        case 'soft-llh':
            return SoftLlhLoss(device, time_ruler, radius=llh_rad)
        case _:
            raise NotImplementedError()

def get_model(model_name: str, config: dict,
              X_num: np.ndarray, X_cat: np.ndarray,
              y_train: np.recarray, time_ruler: np.ndarray,
              device: str | None = None,
              seed: int | None = None):
    if device is None:
        device = 'cpu'
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if model_name.startswith('tabm'): # ablation
        model_name = 'tabm'
    match model_name:
        case 'rsf':
            return RSF(time_ruler=time_ruler, random_state=seed, **config)
        case 'coxph':
            return CoxPH(time_ruler=time_ruler, **config)
        case 'aft':
            return AFTModel(
                num_boost_round=config["num_boost_round"],
                time_ruler=time_ruler,
                early_stopping_rounds=config["early_stopping_rounds"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                min_child_weight=config.get("min_child_weight", 1.0),
                subsample=config.get("subsample", 1.0),
                colsample_bytree=config.get("colsample_bytree", 1.0),
                gamma=config.get("gamma", 0.0),
                l2_reg=config["l2_reg"],
                l1_reg=config["l1_reg"],
                tree_method=config.get("tree_method", "hist"),
                max_bin=config.get("max_bin", 256),
                aft_loss_distribution=config["aft_loss_distribution"],
                aft_loss_distribution_scale=config["aft_loss_distribution_scale"],
                n_jobs=8,
                random_state=seed,
                verbose=False
            )
        case 'gbm_km':
            return GBMKM(
                num_boost_round=config["num_boost_round"],
                time_ruler=time_ruler,
                early_stopping_rounds=config["early_stopping_rounds"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                min_child_weight=config.get("min_child_weight", 1.0),
                subsample=config.get("subsample", 1.0),
                colsample_bytree=config.get("colsample_bytree", 1.0),
                gamma=config.get("gamma", 0.0),
                l2_reg=config["l2_reg"],
                l1_reg=config["l1_reg"],
                tree_method=config.get("tree_method", "hist"),
                max_bin=config.get("max_bin", 256),
                aft_loss_distribution=config["aft_loss_distribution"],
                aft_loss_distribution_scale=config["aft_loss_distribution_scale"],
                n_neighbors=config["n_neighbours"],
                n_jobs=8,
                random_state=seed,
                verbose=False
            )
        case 'gbm_wb':
            return GBMWeibull(
                num_boost_round=config["num_boost_round"],
                time_ruler=time_ruler,
                early_stopping_rounds=config["early_stopping_rounds"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                min_child_weight=config.get("min_child_weight", 1.0),
                subsample=config.get("subsample", 1.0),
                colsample_bytree=config.get("colsample_bytree", 1.0),
                gamma=config.get("gamma", 0.0),
                l2_reg=config["l2_reg"],
                l1_reg=config["l1_reg"],
                tree_method=config.get("tree_method", "hist"),
                max_bin=config.get("max_bin", 256),
                aft_loss_distribution=config["aft_loss_distribution"],
                aft_loss_distribution_scale=config["aft_loss_distribution_scale"],
                wb_penalty=config.get("wb_penalty", 0.0),
                wb_l1_ratio=config["wb_l1_ratio"],
                n_jobs=8,
                random_state=seed,
                verbose=False
            )
        case 'deepsurv':
            return DeepSurv(device=device, x_num_dim=X_num.shape[1],
                            x_cat_dim=X_cat.shape[1], verbose=False,
                            epochs=200, **config)
        case 'deephit':
            return DeepHit(device=device, x_num_dim=X_num.shape[1],
                           x_cat_dim=X_cat.shape[1], time_ruler=time_ruler,
                           verbose=False, epochs=200, **config)
        case 'survtrace':
            if device == 'cpu':
                device_id = None
            else:
                device_id = 0 if device == 'cuda' else int(device.split(':')[-1])
            return SurvTRACE(device_id=device_id, time_ruler=time_ruler, epochs=200,
                             early_stop=10, verbose=False, seed=seed, **config)
        case 'mlp' | 'realmlp' | 'tabm' | 'weibull_params' | 'weibull_proba':
            rtdl_embeddings, rtdl_dim = get_num_emb(model_name, config, X_num, y_train)
            f_wbl_model = model_name.startswith('weibull')
            if not f_wbl_model:
                loss_fn = make_loss(config.get('loss', 'soft-llh'),
                                    config.get('llh_radius', 3),
                                    time_ruler, device)
            else:
                loss_fn = None
            if model_name == 'mlp':
                baseline = MLPBaseline(
                    layer_n=config['layer_n'],
                    num_dim=X_num.shape[1],
                    cat_dim=X_cat.shape[1],
                    rtdl_dim=rtdl_dim,
                    hidden_dim=config['hidden_dim'],
                    output_dim=time_ruler.shape[0],
                    activation=config['activation'],
                    normalization=config['normalization'],
                    num_embeddings=rtdl_embeddings
                )
            elif model_name == 'tabm' or f_wbl_model:
                output_dim = 2 if f_wbl_model else time_ruler.shape[0]
                baseline = TabMBaseline(
                    n_num_features=X_num.shape[1],
                    cat_cardinalities=[2 for _ in range(X_cat.shape[1])],
                    output_dim=output_dim,
                    n_blocks=config['n_blocks'],
                    d_block=config['d_block'],
                    k=config['k'],
                    dropout=config.get('dropout', 0),
                    activation=config['activation'],
                    num_embeddings=rtdl_embeddings
                )
            else:
                baseline = RealMLP(
                    activation=config['activation'],
                    n_hidden_layers=config['n_hidden_layers'],
                    hidden_dim=config['hidden_dim'],
                    output_dim=time_ruler.shape[0],
                    rtdl_embeddings=rtdl_embeddings,
                    rtdl_dim=rtdl_dim,
                    cat_dim=X_cat.shape[1],
                    num_dim=X_num.shape[1]
                )
            if not f_wbl_model:
                model = SurvMLP(
                    device=device,
                    baseline=baseline,
                    time_ruler=time_ruler,
                    loss=loss_fn,
                    epochs_n=500,
                    val_period=1,
                    batch_size=config['batch_size'],
                    optimizer='AdamW',
                    optimizer_params={'lr': config['lr']},
                    verbose=0, patience=20
                )
            else:
                mode = 'wbl_params_mean' if model_name == 'weibull_params' else 'wbl_proba_mean'
                model = SurvMLPWeibull(
                    device=device,
                    baseline=baseline,
                    time_ruler=time_ruler,
                    epochs_n=500,
                    val_period=1,
                    batch_size=config['batch_size'],
                    optimizer='AdamW',
                    optimizer_params={'lr': config['lr']},
                    verbose=0, patience=20,
                    mode=mode,
                    llh_radius=config.get('llh_radius', 3)
                )
            return model
        case _:
            raise NotImplementedError()

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
def get_ds(name: str):
    data = np.load(f'data/{name}.npz')
    return dict(data)

def get_configs(ds_name: str, model_list: List[str]) -> Dict:
    configs = {}
    path = f'pkg/params_selection/params/{ds_name}/'
    for model in model_list:
        file = open(path + f'{model}.json', 'r')
        configs[model] = json.load(file)
        file.close()
    return configs

models_rename = {
    'mlp': r'$\text{TabSurv}_\text{LS(MLP)}$',
    'realmlp': r'$\text{TabSurv}_\text{LS(RealMLP)}$',
    'tabm': r'$\text{TabSurv}_\text{LAS}$',
    'weibull_params': r'$\text{TabSurv}_\text{WAS}$',
    'weibull_proba': r'$\text{TabSurv}_\text{WSA}$',
    'deepsurv': 'DeepSurv',
    'deephit': 'DeepHit',
    'survtrace': 'SurvTRACE',
    'coxph': 'CoxPH',
    'rsf': 'RSF',
    # 'aft': 'AFT',
    'gbm_km': "GBM-KM",
    'gbm_wb': "GBM-Weibull",
    # 'tabm_non_ens': r'$\text{TabSurv}_\text{LAS}\setminus\text{Ens.}$',
    # 'tabm_non_loss': r'$\text{TabSurv}_\text{LAS}\setminus\text{Loss}$',
    # 'tabm_non_rtdl': r'$\text{TabSurv}_\text{LAS}\setminus\text{RTDL}$'
}

gpu_models = {
    'tabm',
    'mlp',
    'realmlp',
    'survtrace',
    'weibull_params',
    'weibull_proba',
}

datasets_rename = {
    'flchain': 'FLC',
    'gbsg2': 'GBSG2',
    'METABRICK': 'METABRICK',
    'SEER': 'SEER',
    'SUPPORT': 'SUPPORT',
    'rotterdam': 'Rotterdam',
    'TCGA_GBM': 'TCGA-GBM',
    'whas500': 'WHAS500',
    'lung': 'Lung',
    'PBC': 'PBC',
    'sim_rotterdam': 'Simulated Rotterdam',
}


def delete_duplicates(
    df: pd.DataFrame,
    dedup_key: tuple[str, ...] = ('model_name', 'ds_name', 'fold_seed', 'repeat_seed'),
) -> pd.DataFrame:
    return df.drop_duplicates(subset=dedup_key, keep='last', inplace=False)


def summarize_experiments(
    df: pd.DataFrame,
    dedup_key: tuple[str, ...] = ('model_name', 'ds_name', 'fold_seed', 'repeat_seed'),
    metric_cols: tuple[str, ...] = ('c_index', 'ibs', 'auc'),
    time_col: str = 'time',
    repeat_col: str = 'repeat_i',
    extra_group_cols: tuple[str, ...] = tuple(),
) -> pd.DataFrame:
    """Average raw fold results for each model, dataset, and repeat."""
    work = delete_duplicates(df, dedup_key)
    group_cols = list(dict.fromkeys((*extra_group_cols, 'model_name', 'ds_name', repeat_col)))
    agg_dict = {column: 'mean' for column in metric_cols}
    agg_dict[time_col] = 'mean'
    return (
        work.groupby(group_cols, as_index=False)
        .agg(agg_dict)
        .rename(columns={time_col: f'mean_{time_col}'})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def ranks_from_results(
    df: pd.DataFrame,
    models: Sequence[str],
    metric_cols: Sequence[str] = ('c_index', 'ibs', 'auc'),
    bigger_is_better: dict[str, bool] | None = None,
    group_cols: tuple[str, ...] = ('ds_name', 'fold_seed'),
    model_col: str = 'model_name',
    dedup_key: tuple[str, ...] = ('model_name', 'ds_name', 'fold_seed', 'repeat_seed'),
) -> pd.DataFrame:
    """Replace raw metrics with per-split model ranks, retaining missing rows."""
    required_cols = {model_col, *group_cols, *metric_cols}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'Dataframe is missing columns: {sorted(missing_cols)}')

    if bigger_is_better is None:
        bigger_is_better = {'c_index': True, 'ibs': False, 'auc': True}

    rank_cols = [f'{metric}_rank' for metric in metric_cols]
    work = delete_duplicates(df, dedup_key)
    group_df = work[list(group_cols)].drop_duplicates()
    work = work[work[model_col].isin(models)].copy()
    index_cols = [*group_cols, model_col]
    rank_df = work[index_cols].drop_duplicates().set_index(index_cols)

    for metric in metric_cols:
        metric_ranks = work.groupby(list(group_cols))[metric].rank(
            method='average',
            ascending=not bigger_is_better.get(metric, True),
            na_option='keep',
        )
        metric_rank_df = work[index_cols].copy()
        metric_rank_df[f'{metric}_rank'] = metric_ranks
        metric_rank_df = (
            metric_rank_df.groupby(index_cols, as_index=True)[f'{metric}_rank']
            .last()
            .to_frame()
        )
        rank_df = rank_df.join(metric_rank_df, how='left')

    full_index_df = group_df.merge(pd.DataFrame({model_col: list(models)}), how='cross')
    full_index = pd.MultiIndex.from_frame(full_index_df[index_cols])
    result = (
        rank_df.reindex(full_index)
        .reset_index()
        .sort_values(index_cols)
        .reset_index(drop=True)
    )
    return result[[*index_cols, *rank_cols]]


def _format_int_values(values: Iterable[int]) -> str:
    sorted_values = sorted(values)
    return ', '.join(str(value) for value in sorted_values) if sorted_values else 'none'


def check_results(
    df: pd.DataFrame,
    folds: int,
    repeats: int,
    models: Sequence[str] | None = None,
    outer_seed: int = 42,
    model_col: str = 'model_name',
    dataset_col: str = 'ds_name',
    repeat_col: str = 'repeat_i',
    fold_col: str = 'fold_i',
    outer_seed_col: str = 'outer_seed',
) -> bool:
    """Check that every selected model/dataset has all folds and repeats."""
    required_cols = {model_col, dataset_col, repeat_col, fold_col, outer_seed_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'Dataframe is missing columns: {sorted(missing_cols)}')
    if folds <= 0:
        raise ValueError('folds must be positive')
    if repeats <= 0:
        raise ValueError('repeats must be positive')

    expected_repeats = set(range(repeats))
    expected_folds = set(range(folds))
    work = df[df[outer_seed_col] == outer_seed].copy()
    models_to_check = (
        sorted(df[model_col].dropna().unique().tolist())
        if models is None
        else list(models)
    )
    datasets_to_check = sorted(df[dataset_col].dropna().unique().tolist())
    is_ok = True

    for model in models_to_check:
        for dataset in datasets_to_check:
            pair_df = work[
                (work[model_col] == model) & (work[dataset_col] == dataset)
            ]
            present_repeats = set(pair_df[repeat_col].dropna().astype(int).unique())
            missing_repeats = expected_repeats - present_repeats
            extra_repeats = present_repeats - expected_repeats
            incomplete_folds = {}

            for repeat_i in sorted(expected_repeats & present_repeats):
                repeat_df = pair_df[pair_df[repeat_col] == repeat_i]
                present_folds = set(repeat_df[fold_col].dropna().astype(int).unique())
                missing_folds = expected_folds - present_folds
                extra_folds = present_folds - expected_folds
                if missing_folds or extra_folds:
                    incomplete_folds[repeat_i] = (
                        present_folds,
                        missing_folds,
                        extra_folds,
                    )

            if missing_repeats or extra_repeats or incomplete_folds:
                is_ok = False
                print(
                    f'[ERROR] Model `{model}`, dataset `{dataset}` has only repeats '
                    f'{_format_int_values(present_repeats)}'
                )
                if missing_repeats:
                    print(f'        Missing repeats: {_format_int_values(missing_repeats)}')
                if extra_repeats:
                    print(f'        Unexpected repeats: {_format_int_values(extra_repeats)}')
                for repeat_i, values in incomplete_folds.items():
                    present_folds, missing_folds, extra_folds = values
                    print(
                        f'        Repeat {repeat_i} has only folds '
                        f'{_format_int_values(present_folds)}'
                    )
                    if missing_folds:
                        print(f'        Missing folds: {_format_int_values(missing_folds)}')
                    if extra_folds:
                        print(f'        Unexpected folds: {_format_int_values(extra_folds)}')
            else:
                print(f'[OK] Model `{model}`, dataset `{dataset}` is ok')
    return is_ok


def metric_matrix_from_melt(
    df: pd.DataFrame,
    metric: str,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    model_col: str = 'model_name',
    dataset_col: str = 'ds_name',
    repeat_col: str = 'repeat_i',
) -> pd.DataFrame:
    """Build a model-by-dataset matrix averaged over experiment repeats."""
    required_cols = {model_col, dataset_col, repeat_col, metric}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'Dataframe is missing columns: {sorted(missing_cols)}')

    work = df.copy()
    if models is not None:
        work = work[work[model_col].isin(models)]
    if datasets is not None:
        work = work[work[dataset_col].isin(datasets)]

    averages = (
        work.groupby([model_col, dataset_col], as_index=False)[metric]
        .mean()
    )
    result = averages.pivot(index=model_col, columns=dataset_col, values=metric)
    if models is not None:
        result = result.reindex(models)
    if datasets is not None:
        result = result.reindex(columns=datasets)
    return result


def metric_latex_table_from_melt(
    df: pd.DataFrame,
    metric: Literal['c_index', 'ibs', 'auc'],
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    model_col: str = 'model_name',
    dataset_col: str = 'ds_name',
    repeat_col: str = 'repeat_i',
    precision: int = 3,
    na_rep: str = '',
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Render mean ± standard deviation by model and dataset as LaTeX."""
    required_cols = {model_col, dataset_col, repeat_col, metric}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'Dataframe is missing columns: {sorted(missing_cols)}')

    bigger_is_better = {'c_index': True, 'auc': True, 'ibs': False}
    if metric not in bigger_is_better:
        raise ValueError(f'Unknown optimization direction for metric: {metric}')
    work = df.copy()
    if models is not None:
        work = work[work[model_col].isin(models)]
    if datasets is not None:
        work = work[work[dataset_col].isin(datasets)]

    stats_df = (
        work.groupby([model_col, dataset_col], as_index=False)
        .agg(mean=(metric, 'mean'), std=(metric, 'std'))
    )
    best_agg = 'max' if bigger_is_better[metric] else 'min'
    best_means = stats_df.groupby(dataset_col)['mean'].transform(best_agg)
    stats_df['is_best'] = stats_df['mean'].eq(best_means) & stats_df['mean'].notna()

    def format_cell(row: pd.Series) -> str:
        mean, std = row['mean'], row['std']
        if pd.isna(mean):
            return na_rep
        value = f'{mean:.{precision}f}'
        cell = value if pd.isna(std) else f'{value} \\pm {std:.{precision}f}'
        return f'$\\mathbf{{{cell}}}$' if row['is_best'] else f'${cell}$'

    stats_df['value'] = stats_df.apply(format_cell, axis=1)
    result = stats_df.pivot(index=model_col, columns=dataset_col, values='value')
    if models is not None:
        result = result.reindex(models)
    if datasets is not None:
        result = result.reindex(columns=datasets)

    result = result.rename(
        index={model: models_rename.get(model, model) for model in result.index},
        columns={dataset: datasets_rename.get(dataset, dataset) for dataset in result.columns},
    )
    result.index.name = None
    result.columns.name = None
    latex_df = result.T
    column_format = '|' + '|'.join(['c'] * (len(latex_df.columns) + 1)) + '|'
    latex = latex_df.to_latex(
        escape=False,
        na_rep=na_rep,
        caption=caption,
        label=label,
        column_format=column_format,
    )
    return (
        latex.replace(r'\toprule', r'\hline')
        .replace(r'\midrule', r'\hline')
        .replace(r'\bottomrule', r'\hline')
    )


def _result_value_mean(value):
    if isinstance(value, (int, float)):
        return float(value)
    return float(np.mean(value))


def merge_results_mean_by_model(first_results: Dict, second_results: Dict) -> Dict[str, Dict[str, float]]:
    """
    Merge two experiment-result dicts by averaging model means.

    Expected input shape matches saved result JSONs:
    {
        "c-index": {"model_a": [...], "model_b": [...]},
        "ibs": {"model_a": [...], "model_b": [...]},
        "auc": {"model_a": [...], "model_b": [...]}
    }

    For overlapping metric/model pairs, the function averages the means from both
    inputs. Models that exist in only one input are copied over with their mean.
    """
    merged_results = {}
    all_metrics = set(first_results) | set(second_results)

    for metric_name in all_metrics:
        merged_results[metric_name] = {}
        first_metric = first_results.get(metric_name, {})
        second_metric = second_results.get(metric_name, {})
        all_models = set(first_metric) | set(second_metric)

        for model_name in all_models:
            first_value = first_metric.get(model_name)
            second_value = second_metric.get(model_name)

            if first_value is not None and second_value is not None:
                merged_results[metric_name][model_name] = np.mean([
                    _result_value_mean(first_value),
                    _result_value_mean(second_value)
                ]).item()
            elif first_value is not None:
                merged_results[metric_name][model_name] = _result_value_mean(first_value)
            else:
                merged_results[metric_name][model_name] = _result_value_mean(second_value)

    return merged_results

def do_hist_experiment(device):
    
    def prepare(y_t, y_c, i):
        time = np.stack((y_t, y_c[:, i]), axis=-1).min(axis=-1)
        censor = y_t < y_c[:, i]
        return get_str_array(time, censor)
    
    i = 0
    data = np.load('data/sim_rotterdam_full.npz')
    X_num_train = data['X_num_train']
    X_cat_train = data['X_cat_train']
    y_train = prepare(data['t_train'], data['c_train'], i)
    X_num_val = data['X_num_val']
    X_cat_val = data['X_cat_val']
    y_val = prepare(data['t_val'], data['c_val'], i)
    X_num_test = data['X_num_test']
    X_cat_test = data['X_cat_test']
    # y_test = prepare(data['t_test'], data['c_test'], i)
    
    time_ruler = np.unique(y_train['time'])
    models = ['tabm', 'rsf', 'weibull_proba', 'weibull_params']
    res_proba = {
        'time_ruler': time_ruler,
        't_test': data['t_test'],
        'models': models
    }
    configs = get_configs('sim_rotterdam', models)
    for model_name in models:
        model = get_model(
            model_name=model_name,
            config=configs[model_name],
            X_num=X_num_train,
            X_cat=X_cat_train,
            y_train=y_train,
            time_ruler=time_ruler,
            device=device
        )
        model.fit(X_num_train, X_cat_train, y_train, (X_num_val, X_cat_val, y_val))
        res_proba[model_name] = model.predict_proba(X_num_test, X_cat_test)
    return res_proba

def exp_cycle(data, models, configs, iters, device):
    time_ruler = np.unique(data['y_train']['time'])
    results = {
        'c-index': defaultdict(list),
        'ibs': defaultdict(list),
        'auc': defaultdict(list)
    }
    for i in range(iters):
        bar = tqdm(models, desc=f'Iter {i + 1}', unit='model')
        for model_name in bar:
            model = get_model(
                model_name=model_name,
                config=configs[model_name],
                X_num=data['X_num_train'],
                X_cat=data['X_cat_train'],
                y_train=data['y_train'],
                time_ruler=time_ruler,
                device=device
            )
            bar.set_postfix_str(f"Training {model_name}")
            model.fit(X_num=data['X_num_train'],
                      X_cat=data['X_cat_train'],
                      y=data['y_train'],
                      val_set=(
                          data['X_num_val'],
                          data['X_cat_val'],
                          data['y_val']
                      ))
            bar.set_postfix_str(f"Testing {model_name}")
            c_index, ibs, auc = model.score(
                X_num=data['X_num_test'],
                X_cat=data['X_cat_test'],
                y=data['y_test']
            )
            results['c-index'][model_name].append(c_index)
            results['ibs'][model_name].append(ibs)
            results['auc'][model_name].append(auc)
    return results

def do_parts_ablation_exp(ds_name: str, iters: int, device: str):
    data = get_ds(ds_name)
    models = ['tabm_non_ens', 'tabm_non_loss', 'tabm_non_rtdl']
    configs = {}
    path = f'pkg/params_selection/params/ablation/{ds_name}/'
    for model in models:
        with open(path + f'{model}.json', 'r') as file:
            configs[model] = json.load(file)
    return exp_cycle(data, models, configs, iters, device)

def do_experiment(ds_name: str, iters: int, device: str) -> dict:
    data = get_ds(ds_name)
    models = list(models_rename.keys())
    configs = get_configs(ds_name, models)
    return exp_cycle(data, models, configs, iters, device)

def do_time_experiment(ds_name: str, iters: int, device: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("It was planned to use CUDA during this experiment")
    data = get_ds(ds_name)
    models = list(models_rename.keys())
    configs = get_configs(ds_name, models)
    time_ruler = np.unique(data['y_train']['time'])
    results = {
        'walltime': defaultdict(list),
        'cpu-time': defaultdict(list),
        'gpu-time': defaultdict(list)
    }
    for i in range(iters):
        bar = tqdm(models, desc=f'Iter {i + 1}', unit='model')
        for model_name in bar:
            model = get_model(
                model_name=model_name,
                config=configs[model_name],
                X_num=data['X_num_train'],
                X_cat=data['X_cat_train'],
                y_train=data['y_train'],
                time_ruler=time_ruler,
                device=device
            )
            bar.set_postfix_str(f"Training {model_name}")
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_t0 = time.time()
            cpu_t0 = time.process_time()
            start_event.record()
            model.fit(
                X_num=data['X_num_train'],
                X_cat=data['X_cat_train'],
                y=data['y_train'],
                val_set=(
                    data['X_num_val'],
                    data['X_cat_val'],
                    data['y_val']
                )
            )
            end_event.record()
            torch.cuda.synchronize()
            wall = time.time() - wall_t0
            cpu = time.process_time() - cpu_t0
            gpu = start_event.elapsed_time(end_event) / 1000.0
            results['walltime'][model_name].append(wall)
            results['cpu-time'][model_name].append(cpu)
            results['gpu-time'][model_name].append(gpu)
    return results

def get_ranks(values: dict, bigger_is_better: bool):
    array = np.asarray([v for v in values.values()])
    coef = -1 if bigger_is_better else 1
    ranks = np.apply_along_axis(
            lambda x: rankdata(coef * x, method="average"),
            axis=0,
            arr=array
        )
    res_mean = {}
    res_std = {}
    for i, k in enumerate(values.keys()):
        res_mean[k] = np.mean(ranks[i])
        res_std[k] = np.std(ranks[i])
    return res_mean, res_std

def dict_to_df(results: Dict[str, List[float]], metric_name: str,
               bigger_is_better: bool=True) -> pd.DataFrame:
    res_mean = {models_rename[key]: np.mean(val) for key, val in results.items()}
    res_std = {models_rename[key]: np.std(val) for key, val in results.items()}
    rank_mean, rank_std = get_ranks({models_rename[k]: v for k, v in results.items()}, bigger_is_better)
    return pd.DataFrame.from_records([res_mean, res_std, rank_mean, rank_std],
                              index=[
                                  metric_name, f'{metric_name} std',
                                  'Rank', 'Rank std'
                                  ]
                              ).T

def gen_gompetz(X: np.ndarray, lmd: float, b: np.ndarray,
                        nu: int, seed: int | None = None) -> np.ndarray:
    assert b.ndim == 1 and X.ndim == 2
    gen = np.random.default_rng(seed)
    u = gen.uniform(0, 1, X.shape[0])
    eta = np.sum(X * b[None, :], axis=-1)
    return np.power(-np.log(u) / (lmd * np.exp(eta)), 1 / nu)

def gen_exponential(n: int, lmd: float, seed: int | None = None) -> np.ndarray:
    gen = np.random.default_rng(seed)
    u = gen.uniform(0, 1, n)
    return - 1 / lmd * np.log(u)

def gen_cens_time(T: np.ndarray, cens_part: float, seed: int | None = None) -> np.ndarray:
    assert T.ndim == 1
    # def f(lmd):
    #     return np.mean(np.exp(-lmd * T)) - 1 + cens_part
    # lmd_left = 0 # f = cens_part
    # lmd_right = 1
    # while f(lmd_right) > 0:
    #     lmd_right *= 2
    #     if lmd_right > 1e6:
    #         raise RuntimeError("Unable to find proper parameter of the exponential distribution")
    # lmd_root = bisect(f, lmd_left, lmd_right, maxiter=1000)
    # C = gen_exponential(T.shape[0], lmd_root, seed)
    gen = np.random.default_rng(seed=seed)
    cens_mark = gen.binomial(1, cens_part, size=T.shape[0])
    C = np.zeros_like(T)
    C[cens_mark == 1] = 0.95 * T[cens_mark == 1]
    C[cens_mark == 0] = 1.05 * T[cens_mark == 0]
    return C

def run_model_job(model_name, config, gpu_queue, data, y_train, y_val, y_test, time_ruler, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if model_name in gpu_models and gpu_queue is not None:
        device = gpu_queue.get()
    else:
        device = 'cpu'
    
    print(f"[{model_name}] Running on device {device}")

    model = get_model(
        model_name=model_name,
        config=config,
        X_num=data['X_num_train'],
        X_cat=data['X_cat_train'],
        y_train=y_train,
        time_ruler=time_ruler,
        device=device
    )
    
    model.fit(
        X_num=data['X_num_train'],
        X_cat=data['X_cat_train'],
        y=y_train,
        val_set=(data['X_num_val'], data['X_cat_val'], y_val)
    )

    c_index, ibs, auc = model.score(
        X_num=data['X_num_test'],
        X_cat=data['X_cat_test'],
        y=y_test
    )
    proba = model.predict_proba(X_num=data['X_num_test'], 
                                X_cat=data['X_cat_test'])
    mean_proba = np.mean(proba, axis=0)
    values = np.repeat(time_ruler, (mean_proba * 10000).astype(int))
    ks = ks_2samp(values, data['t_test'])
    vm = cramervonmises_2samp(values, data['t_test'])
    test_hist, _ = np.histogram(data['t_test'], bins=np.concatenate((time_ruler, [1e8])))
    test_proba = test_hist / np.sum(test_hist)
    emd = wasserstein_distance(time_ruler, time_ruler, mean_proba, test_proba)
    print(f"[{model_name}] Job done")
    
    del model
    
    if device != 'cpu':
        print(f"[{model_name}] Release GPU {device}")
        gpu_queue.put(device)
        
    result = {
        'c_index': c_index,
        'ibs': ibs,
        'auc': auc,
        'emd': emd,
        'ks': ks.statistic,
        'vm': vm.statistic,
    }

    return result


def do_sim_experiment(iters: int, gpus: List[str] | None, n_workers: int):
    if n_workers > 0:
        print('[WARNING] Read https://github.com/havakv/pycox/issues/149 | PyCox is an awesome library!')
    data = get_ds('sim_rotterdam_full')
    models = ['mlp', 'tabm', 'weibull_params', 'weibull_proba', 'rsf', 'deepsurv'] # list(models_rename.keys())
    configs = get_configs('sim_rotterdam', models)
    results_list = []
    if gpus is not None:
        manager = Manager()
        gpu_queue = manager.Queue()
        for g in gpus:
            gpu_queue.put(g)
    else:
        gpu_queue = None

    def prepare(y_t, y_c, i):
        time = np.stack((y_t, y_c[:, i]), axis=-1).min(axis=-1)
        censor = y_t < y_c[:, i]
        return get_str_array(time, censor)
    
    try:
        for i_c, c in enumerate(data['cens_list']):
            print(f'=== Processing c = {c} ===')

            y_train = prepare(data['t_train'], data['c_train'], i_c)
            y_val = prepare(data['t_val'], data['c_val'], i_c)
            y_test = prepare(data['t_test'], data['c_test'], i_c)

            time_ruler = np.unique(y_train['time'])

            results = {
                'c-index': defaultdict(list),
                'ibs': defaultdict(list),
                'auc': defaultdict(list),
                'emd': defaultdict(list),
                'ks': defaultdict(list),
                'vm': defaultdict(list),
            }

            for i in range(iters):
                print(f"=== Iteration {i + 1}/{iters} (outer iter {i_c + 1}/{len(data['cens_list'])}) ===")
                seed = np.random.randint(1, 2 ** 31)

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(
                            run_model_job,
                            model_name,
                            configs[model_name],
                            gpu_queue,
                            data,
                            y_train,
                            y_val,
                            y_test,
                            time_ruler,
                            seed
                        ): model_name
                        for model_name in models
                    }

                    for fut in as_completed(futures):
                        model_name = futures[fut]
                        run_res = fut.result()
                        results['c-index'][model_name].append(run_res['c_index'])
                        results['ibs'][model_name].append(run_res['ibs'])
                        results['auc'][model_name].append(run_res['auc'])
                        results['emd'][model_name].append(run_res['emd'])
                        results['ks'][model_name].append(run_res['ks'])
                        results['vm'][model_name].append(run_res['vm'])
                        print(f"[OK] {model_name}")

            results_list.append(results)
    except Exception:
        traceback.print_exc()
    finally:
        return results_list
