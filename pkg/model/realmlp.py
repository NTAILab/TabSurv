import torch
import torch.nn as nn
import numpy as np
from .baseline import BaselineBase
from typing import Literal

class PBLDEmbeddings(nn.Module):
    def __init__(self, n_features: int,
                 n_frequencies: int,
                 frequency_scale: float,
                 d_embedding: int,
                 plr_act_name: str = 'linear',
                 plr_use_densenet: bool = True):
        super().__init__()
        # print(f'Constructing PBLD embeddings')
        hidden_2 = d_embedding-1 if plr_use_densenet else d_embedding
        self.weight_1 = nn.Parameter(frequency_scale * torch.randn(n_features, 1, n_frequencies))
        self.weight_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, n_frequencies, hidden_2))
                / np.sqrt(n_frequencies))
        self.bias_1 = nn.Parameter(np.pi * (-1 + 2 * torch.rand(n_features, 1, n_frequencies)))
        self.bias_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, 1, hidden_2)) / np.sqrt(n_frequencies))
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet

    def forward(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight_1)  # matmul is automatically batched
        x = x + self.bias_1
        # x = torch.sin(x)
        x = torch.cos(x)
        x = x.matmul(self.weight_2)  # matmul is automatically batched
        x = x + self.bias_2
        if self.plr_act_name == 'relu':
            x = torch.relu(x)
        elif self.plr_act_name == 'linear':
            pass
        else:
            raise ValueError(f'Unknown plr_act_name "{self.plr_act_name}"')
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.plr_use_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x

class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))


class RealMLP(BaselineBase):
    def __init__(self, activation: Literal['SELU', 'Mish'],
                 n_hidden_layers: int, hidden_dim: int, output_dim: int,
                 rtdl_embeddings, rtdl_dim: int | None, num_dim: int, cat_dim: int):
        super().__init__(output_dim)
        act = torch.nn.SELU if activation == 'SELU' else Mish
        if rtdl_dim is None:
            rtdl_dim = 1
        input_dim = cat_dim + num_dim * rtdl_dim
        layers = [ScalingLayer(input_dim), NTPLinear(input_dim, hidden_dim), act()]
        for _ in range(n_hidden_layers):
            layers += [NTPLinear(hidden_dim, hidden_dim), act()]
        layers += [NTPLinear(hidden_dim, output_dim, zero_init=True)]
        self.model = nn.Sequential(
            *layers
        )
        self.num_embeddings = rtdl_embeddings
        
    def forward(self, x_num: torch.Tensor | None, x_cat: torch.Tensor | None):
        if self.num_embeddings is None:
            X_emb = x_num
        else:
            X_emb = self.num_embeddings(x_num).flatten(start_dim=1)
        X = torch.cat((X_emb, x_cat), dim=-1) if x_cat is not None else x_num
        return self.model(X)
