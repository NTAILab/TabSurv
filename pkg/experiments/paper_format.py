from .exp_util import get_ranks, models_rename, do_hist_experiment
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
    'c-index': 'C-index',
    'ibs': 'IBS',
    'auc': 'AUC'
}

@cache
def get_results(ds_name: str) -> dict:
    path = Path(f'pkg/experiments/results/{ds_name}')
    file = open(path / f'{ds_name}_all_results.json', 'r')
    results = json.load(file)
    file.close()
    return results

def get_metric_figure():
    for i, (metric, metric_name) in enumerate(metrics_rename.items()):
        bigger_is_better = False if metric == 'ibs' else True
        mean_ranks = defaultdict(list)
        for ds in ds_rename:
            results = get_results(ds)[metric]
            results = dict(filter(lambda pair: pair[0] in models_rename, results.items()))
            rank_mean, rank_std = get_ranks(results, bigger_is_better)
            for k, v in rank_mean.items():
                mean_ranks[models_rename[k]].append(np.mean(v))
        # sns.set_context('talk')
        # plt.figure()
        # sns.set(style="whitegrid", font_scale=1.2)
        # models_metrics = np.asarray(mean_metrics.keys())
        df = pd.DataFrame.from_dict(mean_ranks)
        df_mean = df.mean(axis=0)
        df_sorted = df_mean.T.sort_values(ascending=False)
        figs, ax = plt.subplots(1, 1, figsize=(4, 6))
        palette_scatter = sns.color_palette('bright', n_colors=len(df.columns))
        palette_scatter = [palette_scatter[df.columns.get_loc(cat)] for cat in df_sorted.index]
        sns.boxplot(
            data=df,
            orient='h',
            palette='muted',
            width=0.5,
            showfliers=False,
            order=df_sorted.index,
        )
        sns.scatterplot(
            x=df_sorted.values,
            y=df_sorted.index,
            palette=palette_scatter,
            hue=df_sorted.index,
            s=80,
            marker="d",
            zorder=10,
            # facecolors='none',
            linewidth=1,
            edgecolors='black',
        )
        ax.set_xlabel('Model rank')
        ax.set_ylabel(None)
        ax.set_title(metric_name)
        ax.grid(which='major', axis='x', ls=':')
        plt.tight_layout()
        plt.xticks([1, 3, 5, 7, 9, 11])
        plt.legend([],[], frameon=False)
        plt.savefig(f'ranks_{metric}.png', dpi=350)
    # plt.show()

def get_metric_table_small(ds: str) -> str:
    metrics_list = list(metrics_rename.keys())
    results_all = get_results(ds)
    aling_list = ['c||', 'c||', 'c|']
    result_strings = [
        r'\begin{table}[H]',
        r'\centering', 
        r'\footnotesize',
        r'\begin{tabular}{|c|c|c||c|c||c|c|}',
        r'\hline',
        r'\multirow{2}{*}{}& ' + '& '.join([rf'\multicolumn{{2}}{{{aling_list[i]}}}{{{name}}}' for i, name in enumerate(metrics_rename.values())]) + r'\\',
        rf'\cline{{{'2-' + str(1 + 2 * len(metrics_list))}}}'
    ]
    for metric in metrics_list:
        bigger_is_better = False if metric == 'ibs' else True
        arrow = r'\uparrow' if bigger_is_better else r'\downarrow'
        result_strings[-1] += fr'& Mean (${arrow}$) & Rank ($\downarrow$)'
    result_strings.append(r'\\')
    result_strings.append(r'\hline')
    best_model_mean = {}
    best_model_rank = {}
    res_mean = {}
    res_std = {}
    rank_mean = {}
    rank_std = {}
    for metric_name in metrics_list:
        results = dict(filter(lambda pair: pair[0] in models_rename, results_all[metric_name].items()))
        res_mean_cur = {key: np.mean(val) for key, val in results.items()}
        res_std_cur = {key: np.std(val) for key, val in results.items()}
        res_mean[metric_name] = res_mean_cur
        res_std[metric_name] = res_std_cur
        bigger_is_better = False if metric_name == 'ibs' else True
        rank_mean_cur, rank_std_cur = get_ranks(results, bigger_is_better)
        rank_mean[metric_name] = rank_mean_cur
        rank_std[metric_name] = rank_std_cur
        index_getter = lambda d: np.argmax(d).item() if bigger_is_better else np.argmin(d).item()
        best_model_mean[metric_name] = list(res_mean_cur.keys())[index_getter(list(res_mean_cur.values()))]
        best_model_rank[metric_name] = list(rank_mean_cur.keys())[np.argmin(list(rank_mean_cur.values())).item()]
    for model, model_name in models_rename.items():
        cur_str = fr'{model_name} '
        for metric_name in metrics_list:
            if best_model_mean[metric_name] == model:
                bf_start = r'\mathbf{'
                bf_end = '}'
            else:
                bf_start = bf_end = ''
            cur_str += fr'& ${bf_start}{res_mean[metric_name][model]:.3f}\pm{res_std[metric_name][model]:.3f}{bf_end}$'
            if best_model_rank[metric_name] == model:
                bf_start = r'\mathbf{'
                bf_end = '}'
            else:
                bf_start = bf_end = ''
            cur_str += fr'& ${bf_start}{rank_mean[metric_name][model]:.1f}\pm{rank_std[metric_name][model]:.1f}{bf_end}$ '
        cur_str += r'\\'
        result_strings.append(cur_str)
        result_strings.append(r'\hline')
    result_strings.append(r'\end{tabular}')
    result_strings.append(rf'\caption{{Comparison of model performance on the {ds_rename[ds]} dataset across C-index, Integrated Brier Score (IBS), and time-dependent AUC}}')
    result_strings.append(r'\label{tab:res_' + ds + '}')
    result_strings.append(r'\end{table}')
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
    ts = np.concat((results['time_ruler'], [results['time_ruler'].max() + 10]))
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
    with open(f'tex_tables/big_table.tex', 'w') as file:
        for ds in ds_rename:
            table = get_metric_table_small(ds)
            file.write(table)
            file.write('\n')
    # get_metric_figure()
    # get_sim_plot()
    # do_kde_figures()
    # draw_losses_figure()
