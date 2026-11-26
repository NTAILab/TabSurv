from __future__ import annotations

import argparse
import os
import tempfile
from functools import cache
from pathlib import Path
from typing import Dict

import numpy as np


DATA_DIR = Path("data")
UNIFIED_KEYS = ("X_num", "X_cat", "y")
SPLITS = ("train", "val", "test")


def _validate_unified(data: Dict[str, np.ndarray], source: Path) -> None:
    missing = set(UNIFIED_KEYS) - set(data)
    if missing:
        raise ValueError(f"{source} is missing unified arrays: {sorted(missing)}")

    x_num, x_cat, y = (data[key] for key in UNIFIED_KEYS)
    if x_num.ndim != 2 or x_cat.ndim != 2 or y.ndim != 1:
        raise ValueError(f"{source} must contain 2D X_num/X_cat and 1D y")
    if not (x_num.shape[0] == x_cat.shape[0] == y.shape[0]):
        raise ValueError(f"{source} arrays have inconsistent row counts")
    if y.dtype.names is None or not {"censor", "time"}.issubset(y.dtype.names):
        raise ValueError(f"{source} y must contain 'censor' and 'time' fields")


@cache
def load_data(ds_name: str, data_dir: str | Path = DATA_DIR) -> Dict[str, np.ndarray]:
    path = Path(data_dir) / f"{ds_name}.npz"
    with np.load(path) as data_file:
        data = {key: data_file[key] for key in data_file.files}
    _validate_unified(data, path)
    return {key: data[key] for key in UNIFIED_KEYS}


def convert_split_dataset(path: str | Path) -> bool:
    """Convert one legacy train/val/test NPZ in place.

    Returns False for already-unified files and files such as simulation inputs
    that do not contain the complete split schema.
    """
    path = Path(path)
    with np.load(path) as data_file:
        keys = set(data_file.files)
        if set(UNIFIED_KEYS).issubset(keys):
            data = {key: data_file[key] for key in UNIFIED_KEYS}
            _validate_unified(data, path)
            return False

        required = {
            *(f"X_num_{split}" for split in SPLITS),
            *(f"X_cat_{split}" for split in SPLITS),
            *(f"y_{split}" for split in SPLITS),
        }
        if not required.issubset(keys):
            return False

        num_widths = {data_file[f"X_num_{split}"].shape[1] for split in SPLITS}
        cat_widths = {data_file[f"X_cat_{split}"].shape[1] for split in SPLITS}
        if len(num_widths) != 1 or len(cat_widths) != 1:
            raise ValueError(f"{path} has incompatible feature widths across splits")

        data = {
            "X_num": np.concatenate([data_file[f"X_num_{split}"] for split in SPLITS]),
            "X_cat": np.concatenate([data_file[f"X_cat_{split}"] for split in SPLITS]),
            "y": np.concatenate([data_file[f"y_{split}"] for split in SPLITS]),
        }

    _validate_unified(data, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **data)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    load_data.cache_clear()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify split survival datasets in place")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    paths = args.paths or sorted(args.data_dir.glob("*.npz"))
    for path in paths:
        status = "converted" if convert_split_dataset(path) else "skipped"
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
