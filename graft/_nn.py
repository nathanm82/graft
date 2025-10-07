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
