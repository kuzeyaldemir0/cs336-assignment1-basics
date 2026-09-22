import torch

from einops import einops, einsum, reduce


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

class RMSNorm(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(
            torch.ones(
                d_model,
                dtype=dtype,
                device=device
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""

        in_dtype = x.dtype
        x = x.to(torch.float32)

        sum_across_d_model = einops.reduce(
            x ** 2,
            "... d_model -> ... 1",
            "sum"
        )
        rms = (sum_across_d_model / self.d_model + self.eps) ** 0.5
        result = x / rms * self.gain

        return result.to(in_dtype)

class swiglu_FFN(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
    ):
        super().__init__()
        self.linear_layer_1 = Linear(d_model, d_ff)
        self.linear_layer_2 = Linear(d_ff, d_model)
        self.linear_layer_3 = Linear(d_model, d_ff)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_branch_1 = self.linear_layer_1(x)
        x_branch_1 = x_branch_1 * torch.sigmoid(x_branch_1)

        x_branch_2 = self.linear_layer_3(x)
        x = x_branch_1 * x_branch_2

        return self.linear_layer_2(x)

class RoPE(torch.nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ):
        super().__init__()
        # Position indices ranging from (0, max_seq_len)
        m = torch.arange(max_seq_len, device=device)

        # Frequencies for each pair in the dimension axis of queries and keys
        freqs = theta ** (-2 * torch.arange(d_k / 2, device=device) / d_k)

        # Calculate all the possible angles and create the T, D_2 matrix
        angles = einops.einsum(m, freqs, "T, D_2 -> T D_2")

        # Calculate the cos and sin of all the angles
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        # Stack them 
        stacked_rotation_matrix = torch.stack(
            [cos_angles, -sin_angles, sin_angles, cos_angles],
            dim=-1
        )
        rotation_matrix = einops.rearrange(
            stacked_rotation_matrix,
            "... (cos_negsin sin_cos) -> ... cos_negsin sin_cos",
            cos_negsin=2,
            sin_cos=2
        )
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        sliced_rotation_matrix = self.rotation_matrix[token_positions[:]]
        x = einops.rearrange(x, "... (pairs coords) -> ... pairs coords 1", coords=2)
        x = sliced_rotation_matrix @ x
        return einops.rearrange(
            x,
            "... pairs coords 1 -> ... (pairs coords)"
        )

if __name__ == "__main__":
    rope_layer = RoPE(theta=10000.0, d_k=128, max_seq_len=256)
    x = torch.ones([16, 5, 128])
    token_positions = torch.arange(80)
    token_positions = token_positions.reshape([16, 5])
    rotated_x = rope_layer(x, token_positions)
    print(rotated_x.shape)

