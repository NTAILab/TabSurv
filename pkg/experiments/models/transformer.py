from typing import Tuple
from .base import CompetitiveModelBase
import numpy as np
from easydict import EasyDict
from .survtrace.utils import LabelTransform
import pandas as pd
from .survtrace.model import SurvTraceSingle
from .survtrace.train_utils import Trainer
import torch
from .base import get_str_array
from ...util import score
import random
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

class SurvTRACE(CompetitiveModelBase):
    def __init__(self,
                 device_id: int | None,
                 time_ruler: np.ndarray,
                 hidden_size: int,
                 intermediate_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 hidden_dropout_prob: float,
                 early_stop: int,
                 batch_size: int,
                 epochs: int,
                 lr: float,
                 weight_decay: float,
                 verbose: bool
                 ):
        super().__init__()
        self.config = EasyDict({
            'seed': 1234,
            'checkpoint': f'./checkpoints/survtrace{generate_random_string(8)}.pt',
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'hidden_dropout_prob': hidden_dropout_prob,
            'num_event': 1,
            'hidden_act': 'gelu',
            'attention_probs_dropout_prob': 0.1,
            'early_stop_patience': early_stop,
            'initializer_range': 0.001,
            'layer_norm_eps': 1e-12,
            'max_position_embeddings': 512,
            'chunk_size_feed_forward': 0,
            'output_attentions': False,
            'output_hidden_states': False,
            'tie_word_embeddings': True,
            'pruned_heads': {},
            'device_id': device_id
        })
        self.time_ruler = time_ruler
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.device_id = device_id
        self.verbose = verbose
        self.labtrans = None
        self.vocab_list = None
        self.cat_chosen_mask = None
        self.verbose = verbose
        
    def _process_data(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        f_fit = False
        if self.labtrans is None or self.vocab_list is None:
            f_fit = True
            self.vocab_list = [0]
            empty_idx = np.min(x_cat, 0) == np.max(x_cat, 0)
            self.cat_chosen_mask = ~empty_idx
        x_cat = x_cat.copy()[:, self.cat_chosen_mask]
        for j in range(x_cat.shape[1]):
            x_cat[:, j] += self.vocab_list[j]
            if f_fit:
                self.vocab_list.append(np.max(x_cat[:, j]) + 1)
        if f_fit:
            self.labtrans = LabelTransform(cuts=self.time_ruler)
            self.labtrans.fit(y['time'].copy(), y['censor'].copy())
        X = np.concatenate((x_cat, x_num), axis=-1)
        Y = self.labtrans.transform(y['time'].copy(), y['censor'].copy())
        df_y = pd.DataFrame({"duration": Y[0], "event": Y[1], "proportion": Y[2]})
        return pd.DataFrame(X), df_y
        
    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray,
            val_set: Tuple[np.ndarray, np.ndarray, np.recarray]):
        self.y_train = y.copy()
        df_train, df_y_train = self._process_data(X_num, X_cat, y)
        df_val, df_y_val = self._process_data(*val_set)
        self.config['labtrans'] = self.labtrans
        self.config['num_numerical_feature'] = X_num.shape[1]
        self.config['num_categorical_feature'] = len(self.vocab_list) - 1
        self.config['num_feature'] = df_train.shape[1]
        self.config['vocab_size'] = int(self.vocab_list[-1])
        self.config['duration_index'] = self.labtrans.cuts
        self.config['out_feature'] = int(self.labtrans.out_features)
        
        self.model = SurvTraceSingle(self.config)
        trainer = Trainer(self.model, self.device_id, self.verbose)
        trainer.fit(
            (df_train, df_y_train), (df_val, df_y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.lr,
            weight_decay=self.weight_decay
        )
        
    def score(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.recarray) -> Tuple[float, float, float]:
        df_test, _ = self._process_data(X_num, X_cat, y)
        df_pred = self.model.predict_surv_df(df_test, batch_size=self.batch_size)
        ruler = df_pred.index.to_numpy()
        sf = df_pred.to_numpy().T
        return score(self.y_train, sf, ruler, y)
