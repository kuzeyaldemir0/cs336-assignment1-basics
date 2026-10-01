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
        self.weight = torch.nn.Parameter(
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
            x, self.weight, "... d_in, d_out d_in -> ... d_out"
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

        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                empty_embedding_lookup,
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids,:]

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
        self.weight = torch.nn.Parameter(
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

        mean_square = einops.reduce(
            x ** 2,
            "... d_model -> ... 1",
            "mean"
        )
        rms = (mean_square + self.eps) ** 0.5
        result = x / rms * self.weight

        return result.to(in_dtype)

class swiglu_FFN(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_branch_1 = self.w1(x)
        x_branch_1 = x_branch_1 * torch.sigmoid(x_branch_1)

        x_branch_2 = self.w3(x)
        x = x_branch_1 * x_branch_2

        return self.w2(x)

class RoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        context_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        # Position indices ranging from (0, context_length)
        m = torch.arange(context_length, device=device, dtype=torch.float32)

        # Frequencies for each pair in the dimension axis of queries and keys
        freqs = theta ** (-2 * torch.arange(d_k / 2, device=device, dtype=torch.float32) / d_k)

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
        if dtype is not None:
            rotation_matrix = rotation_matrix.to(dtype=dtype)
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
        context_length: int | None = None,
        theta: float = 10000.0,
        apply_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.apply_rope = apply_rope
        if apply_rope:
            self.RoPE = RoPE(
                theta, self.d_head, context_length, device=device, dtype=dtype
            )


    def forward(
        self,
        x,
        token_positions: Int[Tensor, " ... seq_len"] = None
    ):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
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
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len, device=x.device
                ).unsqueeze(dim=0)
            token_positions = einops.rearrange(
                token_positions,
                "... seq_len -> ... 1 seq_len"
                # Add one more batch dimension since we want same rotation across different heads
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
        return self.output_proj(attention_heads_concated)


class transformer_block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHead_self_attention(
            d_model, num_heads, context_length,
            theta=theta, apply_rope=True,
            device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = swiglu_FFN(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class transformer_lm(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model,
            device=device, dtype=dtype
        )
        self.layers = torch.nn.ModuleList([
            transformer_block(
                d_model, num_heads, d_ff,
                context_length, theta=theta,
                device=device, dtype=dtype
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size,
            device=device, dtype=dtype
        )


    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

def cross_entropy_loss(
    logits: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    # Left part of the equation
    batch_size, vocab_size = logits.shape
    batch_indices = torch.arange(end=batch_size, dtype=torch.int32)
    target_logits = logits[batch_indices, targets].unsqueeze(dim=-1)
    max_logits = einops.reduce(
        logits,
        "batch_size vocab_size -> batch_size 1",
        "max"
    )
    left_part = max_logits - target_logits

    # Right part of the equation
    logits = logits - max_logits
    exp_logits = torch.exp(logits)
    right_part = torch.log(einops.reduce(
        exp_logits,
        "batch_size vocab_size -> batch_size 1",
        "sum"
    ))
    avg_loss = einops.reduce(
        left_part + right_part,
        "batch_size 1 -> ", # scalar output
        "mean"
    )
    return avg_loss



if __name__ == "__main__":
    # batch_size = 2, vocab_size = 4
    logits = torch.Tensor([[10.2, 1.3, 2.4, 4.3], [1.8, 11.7, 101.9, 12.1]]) # shape: (2, 4)
    targets = torch.Tensor([0, 2]).to(torch.int32) # shape: (2, )
    print(cross_entropy_loss(logits, targets))

