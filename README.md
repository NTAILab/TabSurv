# TabSurv

Official research implementation of **TabSurv** for tabular survival analysis.
The repository contains the model, baseline adapters,
nested cross-validation experiments, Optuna search spaces, ablation studies, and
the results used to build the paper tables and figures.

## Setup

Create the pinned Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate tabsurv
```

The environment uses Python 3.13 and includes PyTorch, scikit-survival, pycox,
XGBoost/XGBSE, Optuna, MLflow, and the plotting/Jupyter dependencies. Commands
below must be run from the repository root because data and output paths are
relative to it.

The exact Conda build strings in `environment.yml` were exported on macOS
Apple Silicon. On another platform, use it as the dependency/version reference
and install platform-appropriate builds (including a CUDA-enabled PyTorch build
when GPU execution is required).

## Repository layout

```text
.
├── parts_ablation/               # saved component-ablation results and notebook
├── pkg/
│   ├── model/                    # TabSurv, MLP, RealMLP, and loss implementations
│   ├── experiments/
│   │   ├── comparison.py         # nested-CV comparison CLI
│   │   ├── experiment.py         # compatibility alias for comparison.py
│   │   ├── data.py               # unified dataset loader/converter
│   │   ├── objective_factory.py  # model search spaces and objectives
│   │   ├── mlp_loss_search.py    # MLP loss search with MLflow logging
│   │   ├── models/               # baseline model adapters
│   │   ├── paper_format.py       # LaTeX table and figure generation
│   │   ├── parts_ablation.py     # component-ablation experiment
│   │   ├── simulation.py         # synthetic-data experiment
│   │   └── results/results.csv   # bundled nested-CV results
│   └── params_selection/         # parameter selection for ablation studies
└── environment.yml
```

## Data

We omit the dataset files from the repository because of their size. If you need
them to run the experiments, please [open a GitHub issue](https://github.com/NTAILab/TabSurv/issues/new)
and specify which datasets you need.

The main benchmark contains ten datasets:

`flchain`, `gbsg2`, `METABRICK`, `SEER`, `SUPPORT`, `rotterdam`,
`TCGA_GBM`, `whas500`, `lung`, and `PBC`.

The comparison expects one NPZ file per dataset with three arrays:

- `X_num`: two-dimensional numerical features;
- `X_cat`: two-dimensional categorical features;
- `y`: a one-dimensional structured array with `censor` and `time` fields.

`data/sim_rotterdam.npz` and `data/sim_rotterdam_full.npz` support the separate
simulation workflow. To convert an older train/validation/test NPZ file to the
unified format in place, run:

```bash
python -m pkg.experiments.data path/to/dataset.npz
```

With no positional paths, the converter checks every `.npz` file under `data/`;
already unified or incompatible simulation files are skipped.

## Model comparison

`pkg.experiments.comparison` performs repeated nested cross-validation. By
default it uses 10 repeats, four outer test folds, 100 Optuna trials per fold,
and a single stratified 25% validation holdout shared by all trials in each outer fold. It records C-index, integrated Brier score
(IBS), time-dependent AUC, training time, seeds, and selected parameters.

The comparison reads raw covariates from `data/raw/<name>.npz` (override with
`-data_dir`). Supply the same three arrays described above, with one column per
original covariate. Use NaN for missing numerical values and either numeric
codes with NaN or Unicode labels with an empty string for missing categories.

Imputation, standardization, and one-hot encoding are fitted on the training
subset. The final fit uses the full outer training fold for CoxPH and RSF.
Other methods reserve a stratified 25% of that fold for early stopping.

For a short smoke run after supplying raw SEER data:

```bash
python -m pkg.experiments.comparison \
  -data SEER \
  -model tabm \
  -trials 2 \
  -repeats 1 \
  -device cpu \
  -table_filename pkg/experiments/results/smoke.csv
```

For the full default run, omit `-trials` and `-repeats`:

```bash
python -m pkg.experiments.comparison \
  -data SEER \
  -model tabm \
  -device cpu \
  -table_filename pkg/experiments/results/comparison_table.csv
```

Use a PyTorch device such as `cuda:0` instead of `cpu` when a compatible GPU
environment is installed.

`python -m pkg.experiments.experiment` is an equivalent compatibility entry
point. Available models are:

```text
mlp, realmlp, tabm, weibull_params, weibull_proba, deepsurv,
deephit, survtrace, coxph, rsf, gbm_km, gbm_wb
```

Useful options include:

- `-data_dir`: directory of raw benchmark datasets (default: `data/raw`);
- `-repeats_spec 1 3 ...`: run only selected 1-based repeat numbers;
- `-seed`: outer random seed (default: `42`);
- `-threads`: PyTorch CPU threads (default: `8`);
- `-logger_filename`: additionally write logs to a file.

Results are appended as semicolon-separated rows. Use a distinct output file for
independent runs to avoid mixing configurations.

## Paper results and formatting

The repository includes the complete benchmark output at
`pkg/experiments/results/results.csv`: 12 models × 10 datasets × 10 repeats ×
4 folds. Formatting utilities validate that all expected repeats and folds are
present and discard duplicate model/dataset/repeat/fold rows by keeping the most
recent one.

Generate the three model-rank figures from the bundled results:

```bash
python -c "from pkg.experiments.paper_format import get_metric_figure; get_metric_figure('pkg/experiments/results/results.csv', 'figures')"
```

Generate the large LaTeX tables:

```bash
mkdir -p tex_tables
python -c "from pathlib import Path; from pkg.experiments.paper_format import get_metric_table_big; [Path(f'tex_tables/table_{m}.tex').write_text(get_metric_table_big(m, 'pkg/experiments/results/results.csv')) for m in ('c_index', 'ibs', 'auc')]"
```

The lower-level `get_metric_table_small`, `get_metric_table_big`, and
`get_metric_figure` functions accept another results path and an `outer_seed`
when formatting newly generated runs.

## MLP loss search

`mlp_loss_search.py` compares three MLP objectives (`c-index`, `likelihood`, and
`soft-llh`) using independent Optuna studies and logs validation metrics and best
parameters to MLflow. Start a local tracking server if one is not already
available:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then run, for example:

```bash
python -m pkg.experiments.mlp_loss_search \
  -data SEER \
  -device cpu \
  -trials_per_loss 50 \
  -k_optuna 1 \
  -val_metric c_index
```

The default tracking URI is `http://127.0.0.1:5000`; override it with
`-tracking_uri`. `-trials_per_loss` is the budget for each loss, so the total
number of trials is three times that value.

## Other experiments

Run the component ablation for one dataset:

```bash
python -m pkg.experiments.parts_ablation -data SEER -device cpu -iters 20
```

Its JSON output is written below `pkg/experiments/results/<dataset>/`. The
precomputed results and the notebook used for their analysis are in
`parts_ablation/`.

Run the synthetic Rotterdam experiment with one or more devices/workers:

```bash
python -m pkg.experiments.simulation -device cpu -iters 20 -n_workers 1
```

## Ablation parameter selection

The scripts in `pkg/params_selection/` contain only the Optuna parameter
searches used for the ablation studies. Selected configurations are stored
under `pkg/params_selection/params/ablation/<dataset>/`. For example:

```bash
python -m pkg.params_selection.ablation_empty -data SEER -device cpu
python -m pkg.params_selection.ablation_non_loss -data SEER -device cpu
```
