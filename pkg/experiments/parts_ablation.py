from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
from torch import set_num_threads
import argparse
from .exp_util import do_parts_ablation_exp, dict_to_df
from .paper_format import ds_rename


DEFAULT_METRICS = ("c-index", "ibs", "auc")
LOWER_IS_BETTER_METRICS = {"ibs"}


def _load_json(path: Path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _metric_mean(value: list[float] | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(np.mean(value))


def _collect_metric_means(results: dict, metric_name: str) -> dict[str, float]:
    metric_results = results.get(metric_name, {})
    return {
        model_name: _metric_mean(metric_values)
        for model_name, metric_values in metric_results.items()
    }


def _get_relative_score(
    baseline_value: float,
    current_value: float,
    metric_name: str,
) -> float:
    if np.isclose(baseline_value, 0.0):
        return np.nan
    if metric_name in LOWER_IS_BETTER_METRICS:
        return 100.0 * (baseline_value - current_value) / baseline_value
    return 100.0 * (current_value - baseline_value) / baseline_value


def get_parts_ablation_tables(
    datasets: Iterable[str] | None = None,
    baseline_model: str = "tabm",
    ablation_suffix: str = "ablation_results",
    metrics: Iterable[str] = DEFAULT_METRICS,
    results_root: str | Path = "pkg/experiments/results",
    use_pretty_dataset_names: bool = True,
    include_baseline_column: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build ablation comparison tables for all datasets.

    The function reads baseline values for ``baseline_model`` from
    ``<dataset>_results.json`` and compares them with models found in the
    neighboring ``<dataset>_<ablation_suffix>.json`` file.

    Returned relative tables contain percentage change with respect to the
    baseline. Positive values always mean "better than baseline":
    - for ``c-index`` and ``auc``: larger is better
    - for ``ibs``: smaller is better
    """
    if datasets is None:
        datasets = ds_rename.keys()

    results_root = Path(results_root)
    metrics = tuple(metrics)

    abs_tables = {metric_name: {} for metric_name in metrics}
    rel_tables = {metric_name: {} for metric_name in metrics}
    metric_columns: dict[str, set[str]] = {metric_name: set() for metric_name in metrics}

    for dataset_name in datasets:
        ds_dir = results_root / dataset_name
        baseline_path = ds_dir / f"{dataset_name}_results.json"
        ablation_path = ds_dir / f"{dataset_name}_{ablation_suffix}.json"

        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline results file was not found: {baseline_path}")
        if not ablation_path.exists():
            raise FileNotFoundError(f"Ablation results file was not found: {ablation_path}")

        baseline_results = _load_json(baseline_path)
        ablation_results = _load_json(ablation_path)
        row_name = ds_rename.get(dataset_name, dataset_name) if use_pretty_dataset_names else dataset_name

        for metric_name in metrics:
            baseline_metric_results = _collect_metric_means(baseline_results, metric_name)
            if baseline_model not in baseline_metric_results:
                raise KeyError(
                    f"Baseline model '{baseline_model}' was not found for metric "
                    f"'{metric_name}' in {baseline_path}"
                )

            baseline_value = baseline_metric_results[baseline_model]
            ablation_metric_results = _collect_metric_means(ablation_results, metric_name)

            abs_row = {}
            rel_row = {}
            if include_baseline_column:
                abs_row[baseline_model] = baseline_value
                rel_row[baseline_model] = 0.0

            for model_name, metric_value in ablation_metric_results.items():
                abs_row[model_name] = metric_value
                rel_row[model_name] = _get_relative_score(
                    baseline_value=baseline_value,
                    current_value=metric_value,
                    metric_name=metric_name,
                )

            abs_tables[metric_name][row_name] = abs_row
            rel_tables[metric_name][row_name] = rel_row
            metric_columns[metric_name].update(abs_row.keys())

    output = {}
    row_order = [
        ds_rename.get(dataset_name, dataset_name) if use_pretty_dataset_names else dataset_name
        for dataset_name in datasets
    ]

    for metric_name in metrics:
        columns = []
        if include_baseline_column and baseline_model in metric_columns[metric_name]:
            columns.append(baseline_model)
        columns.extend(
            sorted(model_name for model_name in metric_columns[metric_name] if model_name != baseline_model)
        )

        abs_df = pd.DataFrame.from_dict(abs_tables[metric_name], orient="index")
        rel_df = pd.DataFrame.from_dict(rel_tables[metric_name], orient="index")

        abs_df = abs_df.reindex(index=row_order, columns=columns)
        rel_df = rel_df.reindex(index=row_order, columns=columns)
        abs_df.index.name = "dataset"
        rel_df.index.name = "dataset"

        safe_metric_name = metric_name.replace("-", "_")
        output[f"{safe_metric_name}_absolute"] = abs_df
        output[f"{safe_metric_name}_relative"] = rel_df

    return output

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
    results = do_parts_ablation_exp(ds_name, iters, device=device)
    with open(path / f'{ds_name}_ablation_results.json', 'w') as file:
        json.dump(results, file)
    df_c_ind = dict_to_df(results['c-index'], 'C-index')
    print(df_c_ind)
    df_ibs = dict_to_df(results['ibs'], 'IBS', bigger_is_better=False)
    print(df_ibs)
    df_auc = dict_to_df(results['auc'], 'AUC')
    print(df_auc)
    # with pd.ExcelWriter(str(path / f'{ds_name}_ablation_results.xlsx')) as writer:
    #    df_c_ind.to_excel(writer, sheet_name="C-index")
    #    df_ibs.to_excel(writer, sheet_name="IBS")
    #    df_auc.to_excel(writer, sheet_name="AUC")
