from abc import abstractmethod
import torch
from typing import Optional

class BaselineBase(torch.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x_num: Optional[torch.Tensor],
                x_cat: Optional[torch.Tensor]):
        ...
