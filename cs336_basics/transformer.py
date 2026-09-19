import torch

from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        x = torch.empty(
            size=[out_features, in_features],
            device=device,
            dtype=dtype
        )

        std = (2 / (in_features + out_features)) ** 0.5
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                x,
                mean=0,
                std=std,
                a=-3*std,
                b=3*std
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.W, "... d_in, d_out d_in -> ... d_out"
        )

class Embedding(torch.nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        empty_embedding_lookup = torch.empty(
            size=[num_embeddings, embedding_dim],
            device=device,
            dtype=dtype
        )

        self.embedding_lookup = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                empty_embedding_lookup,
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_lookup[token_ids,:]