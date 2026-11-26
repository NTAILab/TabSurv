from .exp_util import models_rename, do_hist_experiment
import numpy as np
from pathlib import Path
import json
import seaborn as sns
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from functools import cache
import matplotlib as mpl
from scipy.special import kl_div
from scipy.stats import ks_2samp, cramervonmises_2samp

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "STIX Two Text", "DejaVu Serif"],
    "mathtext.default": "regular",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlepad": 10,
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 300
})

ds_rename = {
    'flchain': 'FLC',
    # 'vlbw': 'VLBW',
    'gbsg2': 'GBSG2',
    'METABRICK': 'METABRICK',
    'SEER': 'SEER',
    'SUPPORT': 'SUPPORT',
    'rotterdam': 'Rotterdam',
    'TCGA_GBM': 'TCGA-GBM',
    'whas500': 'WHAS500',
    'lung': 'Lung',
    'PBC': 'PBC'
}

metrics_rename = {
    'c_index': 'C-index',
    'ibs': 'IBS',
    'auc': 'AUC'
}

DEFAULT_RESULTS_TABLE = Path('pkg/experiments/results/results_final.csv')
EXPECTED_FOLDS = 4
EXPECTED_REPEATS = 10


@cache
def get_results(ds_name: str) -> dict:
    path = Path(f'pkg/experiments/results/{ds_name}')
    file = open(path / f'{ds_name}_all_results.json', 'r')
    results = json.load(file)
    file.close()
    return results

@cache
def _read_cv_results(results_path: str, outer_seed: int) -> pd.DataFrame:
    df = pd.read_csv(results_path, sep=';')
    required = {
        'model_name',
        'ds_name',
        'repeat_i',
        'fold_i',
        'outer_seed',
        *metrics_rename,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Results table is missing columns: {sorted(missing)}')
    df = df[df['outer_seed'] == outer_seed].copy()
    if df.empty:
        raise ValueError(f'No results found for outer_seed={outer_seed}')
    return df


def _get_repeat_results(
    results_path: str | Path = DEFAULT_RESULTS_TABLE,
    outer_seed: int = 42,
    models: tuple[str, ...] | list[str] | None = None,
    datasets: tuple[str, ...] | list[str] | None = None,
    expected_folds: int = EXPECTED_FOLDS,
    expected_repeats: int = EXPECTED_REPEATS,
) -> pd.DataFrame:
    models = list(models_rename) if models is None else list(models)
    datasets = list(ds_rename) if datasets is None else list(datasets)
    raw = _read_cv_results(str(results_path), outer_seed).copy()
    raw = raw[
        raw['model_name'].isin(models) & raw['ds_name'].isin(datasets)
    ]
    raw = raw.drop_duplicates(
        subset=['model_name', 'ds_name', 'repeat_i', 'fold_i'],
        keep='last',
    )

    expected_repeats_set = set(range(expected_repeats))
    expected_folds_set = set(range(expected_folds))
    errors = []
    for model in models:
        for dataset in datasets:
            pair = raw[
                (raw['model_name'] == model) & (raw['ds_name'] == dataset)
            ]
            repeats = set(pair['repeat_i'].astype(int))
            if repeats != expected_repeats_set:
                errors.append(
                    f'{model}/{dataset}: repeats={sorted(repeats)}, '
                    f'expected={sorted(expected_repeats_set)}'
                )
                continue
            for repeat_i in sorted(expected_repeats_set):
                folds = set(
                    pair.loc[pair['repeat_i'] == repeat_i, 'fold_i'].astype(int)
                )
                if folds != expected_folds_set:
                    errors.append(
                        f'{model}/{dataset}/repeat={repeat_i}: '
                        f'folds={sorted(folds)}, expected={sorted(expected_folds_set)}'
                    )
    if errors:
        details = '\n'.join(errors[:10])
        suffix = '' if len(errors) <= 10 else f'\n... and {len(errors) - 10} more'
        raise ValueError(f'Incomplete cross-validation results:\n{details}{suffix}')

    repeat_results = (
        raw.groupby(['model_name', 'ds_name', 'repeat_i'], as_index=False)[
            list(metrics_rename)
        ]
        .mean()
    )
    for metric in metrics_rename:
        repeat_results[f'{metric}_rank'] = repeat_results.groupby(
            ['ds_name', 'repeat_i']
        )[metric].rank(
            method='average',
            ascending=metric == 'ibs',
        )
    return repeat_results


def _get_dataset_rank_points(repeat_results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank models once per dataset after averaging repeat-level metrics (1b)."""
    rank_col = f'{metric}_rank'
    dataset_results = (
        repeat_results.groupby(['ds_name', 'model_name'], as_index=False)[metric]
        .mean()
    )
    dataset_results[rank_col] = dataset_results.groupby('ds_name')[metric].rank(
        method='average',
        ascending=metric == 'ibs',
    )
    return (
        dataset_results.set_index(['ds_name', 'model_name'])[rank_col]
        .unstack('model_name')
        .reindex(index=list(ds_rename), columns=list(models_rename))
    )


def _get_model_palette(model_names) -> dict:
    colors = sns.hls_palette(len(model_names), s=0.8, l=0.65)
    return dict(zip(model_names, colors))

def _get_highlight_palette(model_names) -> dict:
    colors = sns.hls_palette(len(model_names), s=0.8, l=0.65)
    return dict(zip(model_names, colors))

def get_metric_figure(
    results_path: str | Path = DEFAULT_RESULTS_TABLE,
    output_dir: str | Path = '.',
    outer_seed: int = 42,
):
    repeat_results = _get_repeat_results(results_path, outer_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, metric_name in metrics_rename.items():
        rank_points = _get_dataset_rank_points(repeat_results, metric)
        rank_points = rank_points.rename(columns=models_rename)
        mean_ranks = rank_points.mean(axis=0)
        sorted_mean_ranks = mean_ranks.sort_values(ascending=False)
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        model_palette = _get_model_palette(rank_points.columns)
        means_palette = _get_highlight_palette(rank_points.columns)
        sns.boxplot(
            data=rank_points,
            orient='h',
            palette=model_palette,
            width=0.5,
            showfliers=False,
            order=sorted_mean_ranks.index,
            ax=ax,
        )
        sns.scatterplot(
            x=sorted_mean_ranks.values,
            y=sorted_mean_ranks.index,
            palette=means_palette,
            hue=sorted_mean_ranks.index,
            s=80,
            marker="d",
            zorder=10,
            linewidth=1,
            edgecolors='black',
            ax=ax,
        )
        ax.set_xlabel(r"Model rank $(\!\downarrow\!)$")
        ax.set_ylabel(None)
        ax.set_title(metric_name)
        ax.grid(which='major', axis='x', ls=':')
        ax.set_xticks([1, 3, 5, 6.5, 8, 10, 12],
                      labels=['1', '3', '5', '6.5', '8', '10', '12'])
        ax.legend([], [], frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f'ranks_{metric}.png', dpi=350)
        plt.close(fig)


def _format_mean_std(
    mean: float,
    std: float,
    precision: int,
    is_best: bool,
) -> str:
    if np.isnan(mean):
        return '--'
    mean_str = f'{mean:.{precision}f}'
    if is_best:
        mean_str = rf'\mathbf{{{mean_str}}}'
    std_str = '--' if np.isnan(std) else f'{std:.{precision}f}'
    return rf'${mean_str}\,{{\scriptstyle \pm {std_str}}}$'


def get_metric_table_small(
    ds: str,
    results_path: str | Path = DEFAULT_RESULTS_TABLE,
    outer_seed: int = 42,
) -> str:
    if ds not in ds_rename:
        raise ValueError(f'Unknown dataset: {ds}')
    metrics_list = list(metrics_rename.keys())
    repeat_results = _get_repeat_results(
        results_path,
        outer_seed,
        datasets=[ds],
    )
    dataset_results = repeat_results[repeat_results['ds_name'] == ds]
    result_strings = [
        r'\begin{table*}[h!]',
        r'\centering',
        # r'\normalsize',
        rf'\caption{{Comparison of model performance on the {ds_rename[ds]} '
        r'dataset across C-index, Integrated Brier Score (IBS), and '
        r'time-dependent AUC}',
        # r'\footnotesize',
        r'\begin{tabular}{l*{3}{cc}}',
        r'\toprule',
        r'& ' + ' & '.join(
            rf'\multicolumn{{2}}{{c}}{{{name}}}'
            for name in metrics_rename.values()
        ) + r' \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
    ]
    header = ['Model']
    for metric in metrics_list:
        arrow = r'\downarrow' if metric == 'ibs' else r'\uparrow'
        header.extend([f'Mean (${arrow}$)', r'Rank ($\downarrow$)'])
    result_strings.extend([' & '.join(header) + r' \\', r'\midrule'])

    stats = {}
    for metric in metrics_list:
        rank_col = f'{metric}_rank'
        metric_stats = dataset_results.groupby('model_name').agg(
            value_mean=(metric, 'mean'),
            value_std=(metric, 'std'),
            rank_std=(rank_col, 'std'),
        )
        metric_stats['rank_mean'] = metric_stats['value_mean'].rank(
            method='average',
            ascending=metric == 'ibs',
        )
        stats[metric] = metric_stats

    for model, model_name in models_rename.items():
        cells = [model_name]
        for metric in metrics_list:
            metric_stats = stats[metric]
            row = metric_stats.loc[model]
            if metric == 'ibs':
                best_value = metric_stats['value_mean'].min()
            else:
                best_value = metric_stats['value_mean'].max()
            best_rank = metric_stats['rank_mean'].min()
            cells.append(_format_mean_std(
                row['value_mean'],
                row['value_std'],
                3,
                np.isclose(row['value_mean'], best_value),
            ))
            cells.append(_format_mean_std(
                row['rank_mean'],
                row['rank_std'],
                1,
                np.isclose(row['rank_mean'], best_rank),
            ))
        result_strings.append(' & '.join(cells) + r' \\')
    result_strings.extend([r'\bottomrule', r'\end{tabular}'])
    result_strings.append(r'\label{tab:res_' + ds + '}')
    result_strings.append(r'\end{table*}')
    return '\n'.join(result_strings)


def get_metric_table_big(
    metric: str,
    results_path: str | Path = DEFAULT_RESULTS_TABLE,
    outer_seed: int = 42,
    precision: int = 3,
) -> str:
    """Render one metric for every configured model and dataset as LaTeX."""
    if metric not in metrics_rename:
        raise ValueError(
            f'Unknown metric: {metric}. Expected one of {list(metrics_rename)}'
        )

    repeat_results = _get_repeat_results(results_path, outer_seed)
    metric_stats = repeat_results.groupby(['model_name', 'ds_name']).agg(
        value_mean=(metric, 'mean'),
        value_std=(metric, 'std'),
    )
    mean_matrix = metric_stats['value_mean'].unstack('ds_name').reindex(
        index=list(models_rename),
        columns=list(ds_rename),
    )
    std_matrix = metric_stats['value_std'].unstack('ds_name').reindex(
        index=list(models_rename),
        columns=list(ds_rename),
    )
    best_values = (
        mean_matrix.min(axis=0) if metric == 'ibs' else mean_matrix.max(axis=0)
    )
    arrow = r'\downarrow' if metric == 'ibs' else r'\uparrow'

    result_strings = [
        r'\begin{table*}[h!]',
        r'\centering',
        rf'\caption{{Mean {metrics_rename[metric]} (${arrow}$) by model and '
        r'dataset; standard deviations are calculated across repeats}',
        r'\resizebox{\textwidth}{!}{%',
        rf'\begin{{tabular}}{{l*{{{len(ds_rename)}}}{{c}}}}',
        r'\toprule',
        'Model & ' + ' & '.join(ds_rename.values()) + r' \\',
        r'\midrule',
    ]

    for model, model_name in models_rename.items():
        cells = [model_name]
        for dataset in ds_rename:
            mean = mean_matrix.loc[model, dataset]
            std = std_matrix.loc[model, dataset]
            cells.append(_format_mean_std(
                mean,
                std,
                precision,
                np.isclose(mean, best_values[dataset]),
            ))
        result_strings.append(' & '.join(cells) + r' \\')

    result_strings.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        rf'\label{{tab:res_big_{metric}}}',
        r'\end{table*}',
    ])
    return '\n'.join(result_strings)


def draw_losses_figure():
    df = pd.read_csv('losses.csv')
    plt.figure(figsize=(6, 6))

    sns.boxplot(
        data=df,
        x="params.loss",
        y="metrics.c_index",
        palette="muted",
        width=0.7,
        showfliers=False,
    )

    plt.xlabel("Loss")
    plt.ylabel("Validation C-index")
    # plt.title("Distribution of validation accuracy by model type", fontsize=16, pad=15)

    # plt.tight_layout()
    plt.ylim(0.6, 0.85)
    plt.grid(which='major', axis='y')
    # plt.yticks([0.6, 0.7, 0.8])
    plt.savefig('loss_boxplot.png')
    
    
def get_sim_plot():
    with open('pkg/experiments/results/sim_rotterdam/sim_rotterdam_results.json', 'r') as file:
        results = json.load(file)
    data = np.load('data/sim_rotterdam_full.npz')
    cens_list = data['cens_list']
    data.close()

    manual_mean = {
        'c-index': defaultdict(list),
        'ibs': defaultdict(list),
        'auc': defaultdict(list),
        'emd': defaultdict(list),
        'ks': defaultdict(list),
        'vm': defaultdict(list),
    }
    manual_std = {
        'c-index': defaultdict(list),
        'ibs': defaultdict(list),
        'auc': defaultdict(list),
        'emd': defaultdict(list),
        'ks': defaultdict(list),
        'vm': defaultdict(list),
    }
    
    def get_metric_codename(metric_name: str) -> str:
        if metric_name == 'Kolmogorov-Smirnov test':
            return 'ks'
        if metric_name == 'Cramer-von Mises test':
            return 'vm'
        return metric_name.lower()
    
    for i, cur_dict in enumerate(results):
        for metric_name in ['Kolmogorov-Smirnov test']:
            metric = get_metric_codename(metric_name)
            for model, metrics_list in cur_dict[metric].items():
                manual_mean[metric][model].append(np.mean(metrics_list))
                manual_std[metric][model].append(np.std(metrics_list))
                
    model_grouping = [('tabm', 'mlp'), ('weibull_params', 'weibull_proba'), ('rsf', 'deepsurv')]
    markers = {
        'tabm': 'D',
        'mlp': 'o',
        'weibull_params': 'D',
        'weibull_proba': 'o',
        'rsf': 'D',
        'deepsurv': 'o'
    }
    
    for metric_name in ['Kolmogorov-Smirnov test']:
        palette = sns.color_palette("muted", n_colors=6)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        clr_i = 0
        for i, group in enumerate(model_grouping):
            ax = axes[i]

            for model, vals in manual_mean[get_metric_codename(metric_name)].items():
                if model in group:
                    # error = manual_std[get_metric_codename(metric_name)][model]
                    # ax.errorbar(cens_list, vals, yerr=error, label=models_rename[model],
                    #             markerfacecolor='white', markeredgewidth=2.5, marker=markers[model],
                    #             markersize=9, color=palette[clr_i], capsize=5, lw=2.5)
                    ax.plot(cens_list, vals, label=models_rename[model], markerfacecolor='white',
                            markeredgewidth=2.5, marker=markers[model], markersize=9, color=palette[clr_i], lw=2.5)
                    clr_i += 1
            ax.legend(loc='upper left')
            ax.grid()
            ax.set_xticks(ticks=cens_list, labels=[f'{int(100 * (c))}%' for c in cens_list])
            ax.set_xlabel('Proportion of censored data')
            if i == 0:
                ax.set_ylabel(metric_name)
    plt.tight_layout()
    plt.savefig('KS_figure.png')
    # plt.show()
        
def do_kde_figures():
    results = do_hist_experiment('cpu')
    fig, axes =  plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    palette = sns.color_palette("muted", n_colors=len(results['models']) + 1)
    alpha = 0.33
    ec = None
    ts = np.concatenate((results['time_ruler'], [results['time_ruler'].max() + 10]))
    hist, _ = np.histogram(results['t_test'], bins=ts)
    true_proba = hist / np.sum(hist)
    for i, ax in enumerate(axes):
        sns.kdeplot(results['t_test'], bw_adjust=0.7, fill=True, ax=ax, color=palette[0], label='Test density',
                    alpha=0.5 * alpha, zorder=100, clip=(0, 110))
        model_proba = results[results['models'][i]].mean(0)
        n_repeats = (model_proba * 20000).astype(int)
        repeated_grid = np.repeat(ts[:-1], n_repeats)
        u = np.random.uniform(0, 1, size=repeated_grid.shape[0])
        estimated_data = np.repeat(ts[1:] - ts[:-1], n_repeats) * u + repeated_grid
        print(f"--- For model {results['models'][i]} ---")
        tp_nn = true_proba.copy()
        tp_nn[true_proba < 1e-15] = 1e-15
        log = np.log(model_proba / tp_nn)
        log[~np.isfinite(log)] = 0
        kl = np.sum(model_proba * log)
        ks_res = ks_2samp(estimated_data, results['t_test'])
        ks = ks_res.statistic
        vm_res = cramervonmises_2samp(estimated_data, results['t_test'])
        vm = vm_res.statistic
        
        print('\tKL: ', kl)
        print('\tKS: ', ks)
        print('\tCvM: ', vm)
        sns.kdeplot(estimated_data, bw_adjust=0.7, fill=True, ax=ax, color=palette[i + 1],
                    label=f'Estimated density', alpha=alpha, zorder=200, clip=(0, 110))
        # sns.histplot(estimated_data, stat='density', alpha=0.25, ax=ax)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_title(models_rename[results['models'][i]])
        ax.set_ylabel('')
            
        # F_model = np.cumsum(model_proba)
        # ax = axes_f[i]
        # ax.step(results['time_ruler'], F_true, where='post', label='Test')
        # ax.step(results['time_ruler'], F_model, where='post', label=results['models'][i])
        # ax.grid()
        # print('\tMax CDF diff: ', np.max(np.abs(F_model - F_true)))
    plt.tight_layout()
    # plt.show()
    plt.savefig('density_figure.png')

    
if __name__=='__main__':
    for metric in ['c_index', 'ibs', 'auc']:
        with open(f'tex_tables/table_{metric}.tex', 'w') as file:
            file.write(get_metric_table_big(metric))
            # for ds in ds_rename:
            #     table = get_metric_table_small(ds)
            #     file.write(table)
            #     file.write('\n\n')
    # get_metric_figure()
    # get_sim_plot()
    # do_kde_figures()
    # draw_losses_figure()
