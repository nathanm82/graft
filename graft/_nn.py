"""Small reusable building blocks shared by the connectors."""

from __future__ import annotations

from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}


def get_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


def init_linear(module: nn.Linear, std: float = 0.02) -> None:
    """Truncated-normal weight init with zeroed bias (GPT-style)."""
    nn.init.trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FeedForward(nn.Module):
    """A standard transformer MLP block: Linear -> act -> Linear."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a feed-forward block.

    The ``latents`` attend to ``context``. When ``self_attention`` is set the
    latents first attend to one another (used by the Q-Former-style connector).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        if self_attention:
            self.self_norm = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        latents: Tensor,
        context: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if self.self_attention:
            normed = self.self_norm(latents)
            attended, _ = self.self_attn(normed, normed, normed)
            latents = latents + attended
        query = self.q_norm(latents)
        keyval = self.kv_norm(context)
        attended, _ = self.cross_attn(query, keyval, keyval, key_padding_mask=key_padding_mask)
        latents = latents + attended
        latents = latents + self.ff(self.ff_norm(latents))
        return latents
