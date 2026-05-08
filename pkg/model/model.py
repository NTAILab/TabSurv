from .baseline import BaselineBase
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Literal
import numpy as np
from tqdm import tqdm
from ..util import get_str_array, np2torch, calc_exp_time, calc_sf, score
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from .loss import LossBase
from .realmlp import RealMLP
from sklearn.preprocessing import MinMaxScaler
from .loss import LossBase, SoftLlhLoss


class SurvMLP(torch.nn.Module):
    def __init__(self, device: str,
                 baseline: BaselineBase,
                 time_ruler: np.ndarray,
                 loss: LossBase,
                 epochs_n: int,
                 val_period: int,
                 batch_size: int,
                 optimizer: str,
                 optimizer_params: Dict,
                 verbose: int = 1,
                 patience: int = 10):
        super().__init__()
        self.device = torch.device(device)
        self.baseline: torch.nn.Module = baseline.to(self.device)
        self.epochs_n = epochs_n
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.val_period = val_period
        self.processed_epochs = 0
        self.verbose = verbose
        self.y_train = None
        self.cur_ptc = 0
        self.patience = patience
        self.best_loss = 0
        self.bl_params = None
        self.df_params = None
        self.loss = loss
        self.time_ruler = np2torch(time_ruler, self.device)
        
    def _get_optimizer(self, model: torch.nn.Module):
        if isinstance(model, RealMLP):
            params = list(model.model.parameters())
            scale_params = [params[0]]
            weights = params[1::2]
            biases = params[2::2]
            self.lr = self.optimizer_params['lr']
            all_params = [dict(params=scale_params), dict(params=weights), dict(params=biases)]
            if model.num_embeddings is not None:
                all_params.append(dict(params=list(model.num_embeddings.parameters())))
            return torch.optim.Adam(all_params, betas=(0.9, 0.95))
        optim_cls = getattr(torch.optim, self.optimizer)
        return optim_cls(params=model.parameters(), **self.optimizer_params)
    
    def _clip_gradients(self, model: torch.nn.Module) -> int:
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                            foreach=True, error_if_nonfinite=True)
            return 0
        except RuntimeError:
            if self.verbose > 0:
                print('NaN in gradients!')
            count = 0
            for param in model.parameters():
                mask = ~torch.isfinite(param.grad)
                count += int(torch.count_nonzero(mask).item())
                param.grad[mask] = 0
            return count
        
    def _save_params(self):
        self.bl_params = deepcopy(self.baseline.state_dict())
        
    def _restore_params(self):
        if self.bl_params is None:
            return
        self.baseline.load_state_dict(self.bl_params)
        
    def _patience_mechanism(self, c_index: float | None):
        if c_index is None:
            return
        if c_index > self.best_loss:
            self.best_loss = c_index
            self._save_params()
            self.cur_ptc = 0
        else:
            self.cur_ptc += 1
    
    # predicts logits of the event probability
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, f_fit: bool):
        logits = self.baseline(x_num=x_num, x_cat=x_cat)
        if not f_fit and logits.ndim == 3:
            logits = torch.mean(logits, dim=1)
        return logits
    
    # predict survival function
    @torch.inference_mode()
    def predict(self, X_num: np.ndarray, X_cat: np.ndarray, batch_size: int = 512):
        X_num_t = np2torch(X_num, device='cpu')
        X_cat_t = np2torch(X_cat, device='cpu', dtype=torch.long)
        dl = DataLoader(TensorDataset(X_num_t, X_cat_t), batch_size=batch_size)
        res_list = []
        for x_n_b, x_c_b in dl:
            logits = self(x_num=x_n_b.to(self.device), x_cat=x_c_b.to(self.device), f_fit=False)
            sf = calc_sf(logits)
            res_list.append(sf.cpu())
        return torch.cat(res_list, dim=0).numpy()
    
    @torch.inference_mode()
    def score(self, X_num: np.ndarray, X_cat: np.ndarray, 
              y: np.recarray, batch_size: int = 512):
        sf = self.predict(X_num, X_cat, batch_size)
        ruler = self.time_ruler.cpu().numpy()
        return score(self.y_train, sf, ruler, y)

    def _validate(self, epoch: int,
                  val_set: Tuple[np.ndarray, np.ndarray | None, np.recarray]):
        if val_set is None or (epoch - 1) % self.val_period != 0:
            return None, None, None
        c_index, ibs, auc = self.score(*val_set)
        return c_index, ibs, auc
    
    def _make_step(self, loss: torch.Tensor,
                   model: BaselineBase,
                   optimizer: torch.optim.Optimizer):
        loss.backward()
        if torch.isnan(loss):
            if self.verbose > 0:
                print('NaN in loss!')
        else:
            nan_grad = self._clip_gradients(model)
            if nan_grad > 0 and self.verbose > 0:
                print(f'NaN in gradients! {nan_grad} params')
            optimizer.step()
        optimizer.zero_grad()
            
    def fit(self, X_num: np.ndarray,
            X_cat: np.ndarray | None,
            y: np.recarray,
            val_set: Tuple[np.ndarray, np.ndarray | None, np.recarray] | None):
        assert X_num.ndim == 2 and (X_cat is None or X_cat.ndim == 2) and y.ndim == 1
        self.y_train = y.copy()
        
        # X_t, y_t, c_t, weights = self._prepare(model, X, cat_idx, y)
        X_n_t = np2torch(X_num, self.device)
        X_c_t = None if X_cat is None else np2torch(X_cat, self.device, torch.long)
        y_t = np2torch(self.y_train['time'].copy(), self.device)
        c_t = np2torch(self.y_train['censor'].copy(), self.device, torch.long) 
        optimizer = self._get_optimizer(self.baseline)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.patience)
        self.cur_ptc, self.best_loss = 0, 0
        self._save_params()
        for e in range(1, self.epochs_n + 1):
            ds = TensorDataset(X_n_t, y_t, c_t) if X_cat is None else TensorDataset(X_n_t, X_c_t, y_t, c_t)
            dl = DataLoader(ds, self.batch_size, True)
            if self.verbose > 0 and (e % self.verbose == 0 or e == 1):
                bar = tqdm(iterable=dl, desc=f'Epoch {e}')
                cur_verb = True
            else:
                bar = dl
                cur_verb = False
            cum_loss = 0
            for i, data in enumerate(bar, 1):
                if len(data) == 3:
                    x_n_b, y_b, c_b = data
                    x_c_t = None
                else:
                    x_n_b, x_c_t, y_b, c_b = data
                output = self(x_n_b, x_c_t, f_fit=True)
                repeat_n = 0
                if isinstance(output, dict): # corner case for weibull models, unfortunately
                    repeat_n = output['logits'].shape[1]
                    output['logits'] = output['logits'].flatten(end_dim=1)
                    output['log_sf'] = output['log_sf'].flatten(end_dim=1)
                elif output.ndim == 3: # ensemble models
                    repeat_n = output.shape[1]
                    output = output.flatten(end_dim=1)
                if repeat_n > 0: # ensemble models
                    y_b = y_b[:, None].repeat(1, repeat_n).ravel()
                    c_b = c_b[:, None].repeat(1, repeat_n).ravel()
                loss = self.loss(output, y_b, c_b)
                cum_loss += loss.item()
                if cur_verb:
                    bar.set_postfix_str(f'Loss: {cum_loss / i:.3}')
                loss = -loss # maximize
                self._make_step(loss, self.baseline, optimizer)
            # scheduler.step()
            val_c_ind, val_ibs, val_auc = self._validate(e, val_set)
            self._patience_mechanism(val_c_ind)
            if val_c_ind is not None and self.verbose > 0:
                print('Validation C-index:', val_c_ind)
                print('Validation IBS:', val_ibs)
                print('Validation AUC:', val_auc)
                print('Patience:', self.cur_ptc)
            if self.cur_ptc == self.patience:
                if self.verbose > 0:
                    print('Early stopping!')
                break
        self._restore_params()
        
    @torch.inference_mode()
    def predict_proba(self, X_num: np.ndarray, X_cat: np.ndarray, batch_size: int = 512):
        X_num_t = np2torch(X_num, device='cpu')
        X_cat_t = np2torch(X_cat, device='cpu', dtype=torch.long)
        dl = DataLoader(TensorDataset(X_num_t, X_cat_t), batch_size=batch_size)
        res_list = []
        for x_n_b, x_c_b in dl:
            logits = self(x_num=x_n_b.to(self.device), x_cat=x_c_b.to(self.device), f_fit=False)
            res_list.append(torch.softmax(logits, dim=-1).cpu())
        return torch.cat(res_list, dim=0).numpy()

class SurvMLPWeibull(SurvMLP):
    # Same as soft llh, but with convergence trick
    class WeibullLoss(SoftLlhLoss):
        def forward(self, output: dict, t_target: torch.Tensor, c_target: torch.Tensor):
            uncens_mask = c_target == 1
            cens_mask = c_target == 0
            t_uncens = t_target[uncens_mask]
            t_cens = t_target[cens_mask]
            uncens_idx = torch.searchsorted(self.time_ruler, t_uncens).clamp_max_(self.time_ruler.shape[0] - 1)
            cens_idx = torch.searchsorted(self.time_ruler, t_cens).clamp_max_(self.time_ruler.shape[0] - 1)
            logits = output['logits']
            logs_uncens = logits[uncens_mask, :] * self.coeffs[uncens_idx]
            log_sf = output['log_sf'][cens_mask, :]
            logs_cens = torch.take_along_dim(log_sf, cens_idx[:, None], dim=-1)
            total_sum = torch.sum(logs_uncens) + torch.sum(logs_cens)
            return 1 / t_target.shape[0] * total_sum
    
    # mode description:
    # Baseline is ensemble, predicts parameters of the Weibull disrtibution (scale and shape)
    # params_mean - predicted params are averaged over the ensemble instances to calculate final probability
    # params_proba_mean - final probability is calculated as averaged Weibull probabilities
    def __init__(self, device: str,
                 baseline: BaselineBase,
                 mode: Literal['wbl_params_mean', 'wbl_proba_mean'],
                 time_ruler: np.ndarray,
                #  loss: LossBase, # no loss, it is always soft llh
                 epochs_n: int,
                 val_period: int,
                 batch_size: int,
                 optimizer: str,
                 optimizer_params: Dict,
                 verbose: int = 1,
                 patience: int = 10,
                 llh_radius: int = 3):
        loss = self.WeibullLoss(device, time_ruler, llh_radius)
        super().__init__(
            device=device,
            baseline=baseline,
            time_ruler=time_ruler,
            loss=loss,
            epochs_n=epochs_n,
            val_period=val_period,
            batch_size=batch_size,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            verbose=verbose,
            patience=patience
        )
        self.mode = mode
        self.wb_t = torch.log(self.time_ruler.clamp_min(1e-12))

    # predicts probas of the event probability
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, f_fit: bool):
        raw_predict = self.baseline(x_num=x_num, x_cat=x_cat)
        assert raw_predict.ndim == 3, "Only ensemble models are supported"
        assert raw_predict.shape[-1] == 2, "Model must predict 2 parameters of the Weibull distribution"
        k = torch.nn.functional.softplus(raw_predict[..., 0, None])
        log_lmd = raw_predict[..., 1, None]
        tens_shape = (raw_predict.shape[0], raw_predict.shape[1], self.wb_t.shape[0])
        log_time = self.wb_t[None, None, :]
        
        if not f_fit and self.mode == 'wbl_params_mean':
            k = torch.mean(k, dim=1)
            log_lmd = torch.mean(log_lmd, dim=1)
            tens_shape = (tens_shape[0], tens_shape[-1])
            log_time = log_time[0]
                
        # u = exp(k * (log t - log lmd))
        log_u = k * (log_time - log_lmd)
        u = torch.exp(log_u)
        u_left = u[..., :-1]
        u_right = u[..., 1:]
        
        # log f = -u_left + log(1 - exp(u_left - u_right))
        logits = torch.empty(tens_shape, device=self.device)
        logits[..., :-1] = -u_left + torch.log1p(-torch.exp((u_left - u_right).clamp_max_(-1e-7)))
        logits[..., -1] = -u_right[..., -1] # right edge because there will be problems with the left when k < 1
        
        log_S = -u

        if not f_fit and self.mode == 'wbl_proba_mean':
            logits = torch.mean(logits, dim=1)
            log_S = torch.mean(log_S, dim=1)
        return {'logits': logits, 'log_sf': log_S}

    @torch.inference_mode()
    def predict(self, X_num: np.ndarray, X_cat: np.ndarray, batch_size: int = 512):
        X_num_t = np2torch(X_num, device='cpu')
        X_cat_t = np2torch(X_cat, device='cpu', dtype=torch.long)
        dl = DataLoader(TensorDataset(X_num_t, X_cat_t), batch_size=batch_size)
        res_list = []
        for x_n_b, x_c_b in dl:
            output = self(x_num=x_n_b.to(self.device), x_cat=x_c_b.to(self.device), f_fit=False)
            sf = torch.exp(output['log_sf']).clamp_min_(1e-7)
            res_list.append(sf.cpu())
        return torch.cat(res_list, dim=0).numpy()

    @torch.inference_mode()
    def predict_proba(self, X_num: np.ndarray, X_cat: np.ndarray, batch_size: int = 512):
        X_num_t = np2torch(X_num, device='cpu')
        X_cat_t = np2torch(X_cat, device='cpu', dtype=torch.long)
        dl = DataLoader(TensorDataset(X_num_t, X_cat_t), batch_size=batch_size)
        res_list = []
        for x_n_b, x_c_b in dl:
            output = self(x_num=x_n_b.to(self.device), x_cat=x_c_b.to(self.device), f_fit=False)
            proba = torch.exp(output['logits'])
            res_list.append(proba.cpu())
        return torch.cat(res_list, dim=0).numpy()
