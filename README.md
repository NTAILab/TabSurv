# TabSurv

This repository contains the official implementation of **TabSurv**, proposed in the accompanying paper.  
The code is provided for reproducibility and research purposes.

## Environment Setup

The repository includes an `environment.yml` file for creating a Conda environment with all required dependencies.

```bash
conda env create -f environment.yml
conda activate tabsurv
```

## Repository Structure

The repository is organized as follows:

```
.
├── environment.yml
├── data/
│   └── *.npz
├── parts_ablation/
│   ├── *.json
│   └── parts_ablation.ipynb
├── pkg/
│   ├── model/
│   │   ├── *.py
│   ├── experiments/
│   │   ├── models/
│   │   ├── results/
│   │   ├── losses.csv
│   │   ├── paper_format.py
│   │   ├── simulation.py
│   │   ├── experiment.py
│   │   └── *.py
│   └── params_selection/
│       ├── params/
│       └── *.py
```

### `data/`

Contains compressed datasets (`.npz`) used in the experiments reported in the paper.

### `parts_ablation/`

Contains experiments for Figure 4 in the paper.

### `pkg/model/`

Implementation of the proposed **TabSurv** method.

### `pkg/experiments/`

Code for running experiments and reproducing results from the paper.

- `experiment.py` — main entry point for running experiments.
- `models/` — source code and adapters for baseline models used for comparison.
- `results/` — directory where experimental results are stored.
- `simulation.py` — code for generating the synthetic dataset used in experiments.
- `paper_format.py` — utilities for aggregating results and producing tables and figures as reported in the paper.

### `pkg/params_selection/`

Scripts for hyperparameter optimization using **Optuna**.

- Each script corresponds to a specific model.
- `params/` contains the selected hyperparameters used in all reported experiments.

## Running Experiments

All experiments are executed from the root of the repository using Python module execution.

For example, to run the experiment corresponding to **Table 3** on the SEER dataset:

```bash
python -m pkg.experiments.experiment -data SEER -device cuda:1
```

After the experiments are completed, the results can be aggregated and formatted using:

```bash
python -m pkg.experiments.paper_format
```

## Hyperparameter Selection

Hyperparameters for each model are selected using Optuna.

For example, to perform hyperparameter optimization for the **TabSurv_LS(MLP)** model on the SEER dataset:

```bash
python -m pkg.params_selection.mlp -data SEER
```

All selected hyperparameters used in the experiments are stored in:

```
pkg/params_selection/params/
```
