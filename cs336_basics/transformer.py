import torch

from einops import einops, einsum, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        empty_init = torch.empty(
            size=[out_features, in_features],
            device=device,
            dtype=dtype
        )

        std = (2 / (in_features + out_features)) ** 0.5
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                empty_init,
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
        sliced_rotation_matrix = self.rotation_matrix[token_positions]
        x = einops.rearrange(x, "... (pairs coords) -> ... pairs coords 1", coords=2)
        x = sliced_rotation_matrix @ x
        return einops.rearrange(
            x,
            "... pairs coords 1 -> ... (pairs coords)"
        )

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    pre_softmax = einops.einsum(
        Q, K,
        "... query_positions d_k, ... kv_positions d_k -> ... query_positions kv_positions") / (K.shape[-1] ** 0.5
    )
    if mask is not None:
        pre_softmax = torch.where(mask, pre_softmax, float("-inf"))
    attention_weights = softmax(pre_softmax, dim=-1)
    return einops.einsum(
        attention_weights, V,
        "... query_positions kv_positions, ... kv_positions d_v -> ... query_positions d_v"
    )


class MultiHead_self_attention(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int = None,
            theta: float = None,
            token_positions: Int[Tensor, " ... seq_len"] = None,
            apply_rope: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.Q = Linear(d_model, d_model)
        self.K = Linear(d_model, d_model)
        self.V = Linear(d_model, d_model)
        self.output_projection = Linear(d_model, d_model)
        self.apply_rope = apply_rope
        if apply_rope:
            self.RoPE = RoPE(theta, self.d_head, max_seq_len)
            self.token_positions = token_positions


    def forward(self, x):
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)
        seq_len = queries.shape[-2]
        # We have Q, K, and V as shape (batch_size, seq_len, d_model)

        # Let's split the d_model into multiple heads for MHA
        queries = einops.rearrange(
            queries,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
            d_head=self.d_head
        )
        keys = einops.rearrange(
            keys,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
            d_head=self.d_head
        )
        values = einops.rearrange(
            values,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
            d_head=self.d_head
        )

        if self.apply_rope:
            # Apply the same RoPE to each head separately
            token_positions = einops.rearrange(
                self.token_positions,
                "... seq_len -> ... 1 seq_len"
            )
            queries = self.RoPE(queries, token_positions)
            keys = self.RoPE(keys, token_positions)

        boolean_mask = torch.ones(size=(seq_len, seq_len), device=x.device, dtype=torch.bool)
        causal_mask = torch.tril(boolean_mask)
        attention_scores = scaled_dot_product_attention(
            queries,
            keys,
            values,
            causal_mask
        )

        attention_heads_concated = einops.rearrange(
            attention_scores,
            "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)"
        )
        return self.output_projection(attention_heads_concated)



if __name__ == "__main__":
    q = torch.ones([16, 10, 128])
    k = torch.ones([16, 10, 128])
    qk = scaled_dot_product_attention(q, k, k)
    print(qk.shape)
