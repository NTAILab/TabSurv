import torch
import numpy as np
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc

TYPE = [('censor', '?'), ('time', 'f8')]

def get_str_array(time: np.ndarray, c: np.ndarray) -> np.recarray:
    assert time.shape == c.shape
    str_array = np.ndarray(shape=time.shape, dtype=TYPE)
    str_array['censor'] = c
    str_array['time'] = time
    return str_array

def np2torch(array: np.ndarray, 
             device: torch.device | str | None = None, 
             dtype: torch.dtype | None = None):
    device = 'cpu' if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return torch.tensor(array, device=device, dtype=dtype)

def calc_sf(logits: torch.Tensor):
    proba = torch.softmax(logits, dim=-1)
    cumsum = torch.cumsum(proba, dim=-1)
    sf = 1 - cumsum
    return sf.clamp_min_(1e-7)

def calc_exp_time(sf: torch.Tensor, time_ruler: torch.Tensor):
    assert time_ruler.ndim == 1 and sf.ndim == 2
    t_diff = time_ruler[None, 1:] - time_ruler[None, :-1]
    integral = time_ruler[None, 0] + torch.sum(sf[:, :-1] * t_diff, dim=-1)
    return integral

def score(y_train: np.recarray, sf: np.ndarray,
          sf_time: np.ndarray, y_label: np.recarray):
    t_diff = sf_time[None, 1:] - sf_time[None, :-1]
    exp_time = sf_time[None, 0] + np.sum(sf[:, :-1] * t_diff, axis=-1)
    try:
        c_index, *_ = concordance_index_censored(y_label['censor'], y_label['time'], -exp_time)
    except Exception as e:
        c_index = float('nan')
    brd = np.percentile(y_train['time'], [0, 95])
    test_idx = (brd[0] <= y_label['time']) & (y_label['time'] <= brd[1])
    t_test = y_label['time'][test_idx]
    c_test = y_label['censor'][test_idx]
    picked_time = np.unique(t_test)[:-1]
    first_event = t_test[c_test].min()
    picked_time = picked_time[picked_time > first_event]
    picked_sf_time = np.tile(sf_time[None, :], (np.sum(test_idx), 1))
    picked_idx = torch.searchsorted(torch.tensor(picked_sf_time), 
            torch.tensor(picked_time)[None, :].repeat(picked_sf_time.shape[0], 1), side='left')
    picked_idx = picked_idx.clamp_max_(picked_sf_time.shape[-1] - 1).numpy()
    picked_sf = np.take_along_axis(sf[test_idx], picked_idx, axis=-1)
    y_test = get_str_array(t_test, c_test)
    try:
        ibs = integrated_brier_score(y_train, y_test, picked_sf, picked_time)
    except Exception as e:
        ibs = float('nan')
    try:
        _, auc = cumulative_dynamic_auc(y_train, y_test, -np.log(picked_sf.clip(min=1e-15)), picked_time)
    except Exception as e:
        auc = float('nan')
    return c_index, ibs, auc
