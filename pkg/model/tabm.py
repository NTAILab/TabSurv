from tabm import TabM
from .baseline import BaselineBase
from typing import List
from torch import Tensor

class TabMBaseline(BaselineBase):
    def __init__(self, 
                 n_num_features: int,
                 cat_cardinalities: List[int],
                 output_dim: int,
                 n_blocks: int,
                 d_block: int,
                 k: int,
                 dropout: float,
                 activation: str,
                 num_embeddings
                 ):
        super().__init__(output_dim=output_dim)
        self.tabm = TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_out=output_dim,
            n_blocks=n_blocks,
            d_block=d_block,
            k=k,
            dropout=dropout,
            activation=activation,
            num_embeddings=num_embeddings,
        )
        
    def forward(self, x_num: Tensor | None, x_cat: Tensor | None):
        return self.tabm(x_num=x_num, x_cat=x_cat)
