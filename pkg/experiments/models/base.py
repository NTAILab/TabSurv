from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from ...util import score
import torch

TYPE = [('censor', '?'), ('time', 'f8')]

def get_str_array(time: np.ndarray, c: np.ndarray) -> np.recarray:
    assert time.shape == c.shape
    str_array = np.ndarray(shape=time.shape, dtype=TYPE)
    str_array['censor'] = c
    str_array['time'] = time
    return str_array

class CompetitiveModelBase(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray, 
            val_set: Tuple[np.ndarray, np.ndarray, np.recarray] | None = None):
        ...
    
    # @abstractmethod
    # def predict(self, x_num: np.ndarray, x_cat: np.ndarray):
    #     ...
    
    @abstractmethod
    def score(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray) -> Tuple[float, float, float]:
        ...

class CompetitivePycoxBase(CompetitiveModelBase):
    def __init__(self):
        super().__init__()
        self.model = None
        self.y_train = None
        self.batch_size = None
        
    def score(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray):
        X_test = np.concatenate((X_num, X_cat), axis=-1)
        sf_df = self.model.predict_surv_df(X_test.astype(np.float32), batch_size=self.batch_size)
        ruler = sf_df.index.to_numpy()
        sf = sf_df.to_numpy().T
        return score(self.y_train, sf, ruler, y)

class CompetitiveSksurvBase(CompetitiveModelBase):
    def __init__(self, time_ruler: np.ndarray):
        super().__init__()
        self.model = None
        self.y_train = None
        self.time_ruler = time_ruler
        
    def score(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray):
        X_test = np.concatenate((X_num, X_cat), axis=-1)
        ruler = self.time_ruler
        sf_list = self.model.predict_survival_function(X_test, return_array=False)
        sf = np.empty((X_test.shape[0], self.time_ruler.shape[0]))
        for i, step_func in enumerate(sf_list):
            sf[i, :] = step_func(self.time_ruler)
        return score(self.y_train, sf, ruler, y)
