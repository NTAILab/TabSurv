from typing import Literal, Tuple
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from .base import CompetitiveSksurvBase

class CoxNet(CompetitiveSksurvBase):
    def __init__(self, alpha: float, l1_ratio: float, normalize: bool):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.model = CoxnetSurvivalAnalysis(n_alphas=1, alphas=[alpha],
                                            l1_ratio=l1_ratio, normalize=normalize,
                                            fit_baseline_model=True)
        
    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray,
            val_set: Tuple[np.ndarray, np.ndarray, np.recarray] | None = None):
        if val_set is not None:
            X_num = np.concatenate((X_num, val_set[0]), axis=0)
            X_cat = np.concatenate((X_cat, val_set[1]), axis=0)
        X_train = np.concatenate((X_num, X_cat), axis=-1)
        if val_set is not None:
            y = np.concatenate((y, val_set[2]), axis=0)
        self.y_train = y.copy()
        self.model.fit(X_train, y)
        
class RSF(CompetitiveSksurvBase):
    def __init__(self, n_estimators: int, max_depth: int, 
                 min_samples_split: int | float, max_features: Literal['sqrt', 'log2'],
                 time_ruler: np.ndarray):
        super().__init__(time_ruler)
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            n_jobs=8
        )
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.max_features=max_features
        
    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray,
            val_set: Tuple[np.ndarray, np.ndarray, np.recarray] | None = None):
        if val_set is not None:
            X_num = np.concatenate((X_num, val_set[0]), axis=0)
            X_cat = np.concatenate((X_cat, val_set[1]), axis=0)
        X_train = np.concatenate((X_num, X_cat), axis=-1)
        if val_set is not None:
            y = np.concatenate((y, val_set[2]), axis=0)
        self.y_train = y.copy()
        self.model.fit(X_train, y)
        
    def predict_proba(self, X_num: np.ndarray, X_cat: np.ndarray):
        X = np.concatenate((X_num, X_cat), axis=-1)
        sf_list = self.model.predict_survival_function(X, False)
        sf = np.empty((X.shape[0], self.time_ruler.shape[0]))
        for i, step_func in enumerate(sf_list):
            sf[i, :] = step_func(self.time_ruler)
        proba = np.empty_like(sf)
        proba[:, :-1] = sf[:, :-1] - sf[:, 1:]
        proba[:, -1] = sf[:, -1]
        return proba
