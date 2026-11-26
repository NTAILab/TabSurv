from abc import ABC, abstractmethod
import torch
import numpy as np
from ..util import np2torch, calc_exp_time, calc_sf

class LossBase(ABC, torch.nn.Module):
    def __init__(self, device: str, time_ruler: np.ndarray):
        super().__init__()
        self.device = torch.device(device)
        self.time_ruler = np2torch(time_ruler).to(self.device)
    
    @abstractmethod
    def forward(self, logits: torch.Tensor, t_target: torch.Tensor, c_target: torch.Tensor) -> torch.Tensor:
        ...

class CIndexLoss(LossBase):
    def __init__(self, device: str, time_ruler: np.ndarray, sigma_temp: float = 1):
        super().__init__(device, time_ruler)
        self.sigma_temp = sigma_temp
    
    def forward(self, logits: torch.Tensor, t_target: torch.Tensor, c_target: torch.Tensor):
        exp_time = calc_exp_time(logits, self.time_ruler)
        assert t_target.ndim == c_target.ndim == exp_time.ndim == 1
        T_diff = t_target[:, None] - t_target[None, :]
        T_mask = T_diff < 0
        sigma = torch.nn.functional.sigmoid(self.sigma_temp * (exp_time[None, :] - exp_time[:, None]))
        C = torch.sum(T_mask * sigma * c_target[:, None]) / torch.sum(T_mask * c_target[:, None])
        return C

class LikelihoodLoss(LossBase):
    def __init__(self, device: str, time_ruler: np.ndarray):
        super().__init__(device, time_ruler)
    
    def forward(self, logits: torch.Tensor, t_target: torch.Tensor, c_target: torch.Tensor):
        uncens_mask = c_target == 1
        cens_mask = c_target == 0
        t_uncens = t_target[uncens_mask]
        t_cens = t_target[cens_mask]
        uncens_idx = torch.searchsorted(self.time_ruler, t_uncens).clamp_max_(self.time_ruler.shape[0] - 1)
        cens_idx = torch.searchsorted(self.time_ruler, t_cens).clamp_max_(self.time_ruler.shape[0] - 1)
        uncens_logits = torch.log_softmax(logits[uncens_mask], dim=-1)
        logs_uncens = torch.take_along_dim(uncens_logits, uncens_idx[:, None], dim=-1)
        sf = calc_sf(logits[cens_mask, :])
        logs_cens = torch.take_along_dim(sf, cens_idx[:, None], dim=-1)
        logs_cens = torch.log(logs_cens)
        total_sum = torch.sum(logs_uncens) + torch.sum(logs_cens)
        return 1 / t_target.shape[0] * total_sum

class SoftLlhLoss(LossBase):
    def __init__(self, device: str, time_ruler: np.ndarray, radius: int):
        super().__init__(device, time_ruler)
        x = torch.arange(time_ruler.shape[0])[None, :].repeat(time_ruler.shape[0], 1)
        sigma = radius / 3
        coef = 1 / (np.sqrt(2 * np.pi) * sigma)
        self.coeffs = coef * torch.exp(-(x - torch.arange(time_ruler.shape[0])[:, None]) ** 2 / (2 * sigma ** 2))
        self.coeffs = self.coeffs / self.coeffs.sum(dim=-1, keepdim=True)
        self.coeffs = self.coeffs.to(device)
        
    def forward(self, logits: torch.Tensor, t_target: torch.Tensor, c_target: torch.Tensor):
        uncens_mask = c_target == 1
        cens_mask = c_target == 0
        t_uncens = t_target[uncens_mask]
        t_cens = t_target[cens_mask]
        uncens_idx = torch.searchsorted(self.time_ruler, t_uncens).clamp_max_(self.time_ruler.shape[0] - 1)
        cens_idx = torch.searchsorted(self.time_ruler, t_cens).clamp_max_(self.time_ruler.shape[0] - 1)
        logs_uncens = torch.log_softmax(logits[uncens_mask, :], dim=-1) * self.coeffs[uncens_idx]
        sf = calc_sf(logits[cens_mask, :])
        logs_cens = torch.take_along_dim(sf, cens_idx[:, None], dim=-1)
        logs_cens = torch.log(logs_cens)
        total_sum = torch.sum(logs_uncens) + torch.sum(logs_cens)
        return 1 / t_target.shape[0] * total_sum
