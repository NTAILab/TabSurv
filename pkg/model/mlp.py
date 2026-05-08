from .baseline import BaselineBase
import torch

class MLPBaseline(BaselineBase):
    def __init__(self,
                 layer_n: int,
                 num_dim: int,
                 cat_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 activation: str,
                 normalization: bool,
                 num_embeddings: torch.nn.Module | None = None,
                 rtdl_dim: int | None = None,
                 ):
        super().__init__(output_dim=output_dim)
        assert not (num_embeddings is not None and rtdl_dim is None)
        assert not (rtdl_dim is not None and num_embeddings is None)
        layers = []
        if rtdl_dim is None:
            rtdl_dim = 1
        inp_d = num_dim * rtdl_dim + cat_dim
        act = getattr(torch.nn, activation)
        for _ in range(layer_n - 1):
            out_d = max(16, hidden_dim)
            layers.append(torch.nn.Linear(inp_d, out_d))
            if normalization:
                layers.append(torch.nn.LayerNorm(out_d))
            layers.append(act())
            inp_d = out_d
        # mid = layer_n // 2
        # for i in range(layer_n - 1):
        #     if i < mid and layer_n % 2 == 0:
        #         i += 1
        #     out_d = max(16, hidden_dim // 2 ** abs(mid - i))
        #     layers.append(torch.nn.Linear(inp_d, out_d))
        #     if normalization:
        #         layers.append(torch.nn.LayerNorm(out_d))
        #     layers.append(act())
        #     inp_d = out_d
        layers.append(torch.nn.Linear(inp_d, output_dim))
        self.nn = torch.nn.Sequential(*layers)
        self.num_embeddings = num_embeddings
        
    def forward(self, x_num: torch.Tensor | None, x_cat: torch.Tensor | None):
        if self.num_embeddings is None:
            X_emb = x_num
        else:
            X_emb = self.num_embeddings(x_num).flatten(start_dim=1)
        X_cat = x_cat.clone()
        X_cat[X_cat == 0] = -1
        X = torch.cat((X_emb, X_cat), dim=-1) if x_cat is not None else x_num
        return self.nn(X)
