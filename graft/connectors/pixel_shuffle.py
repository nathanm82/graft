"""A pixel-shuffle (space-to-depth) downsampler."""

from __future__ import annotations

from math import isqrt

from torch import Tensor, nn

from graft._nn import init_linear
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.exceptions import ConfigError, ShapeContractError
from graft.registry import register_connector


@register_connector("pixel_shuffle")
class PixelShuffleConnector(Connector):
    """Group neighbouring tokens to trade sequence length for channels.

    The ``N`` visual tokens are treated as a square ``h x h`` grid. Each
    ``scale_factor x scale_factor`` neighbourhood is folded into a single token
    whose channel dimension grows by ``scale_factor**2`` before a linear
    projection to ``output_dim``. The token count therefore drops by
    ``scale_factor**2``. ``scale_factor`` is read from ``config.extra``.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self.scale_factor = int(config.extra.get("scale_factor", 2))
        if self.scale_factor < 1:
            raise ConfigError(f"scale_factor must be >= 1, got {self.scale_factor}")
        in_features = config.input_dim * self.scale_factor**2
        self.proj = nn.Linear(in_features, config.output_dim, bias=config.bias)
        init_linear(self.proj)

    def _grid_side(self, num_tokens: int) -> int:
        side = isqrt(num_tokens)
        if side * side != num_tokens:
            raise ShapeContractError(
                f"pixel_shuffle expects a square token grid, got {num_tokens} tokens"
            )
        if side % self.scale_factor != 0:
            raise ShapeContractError(
                f"grid side {side} is not divisible by scale_factor {self.scale_factor}"
            )
        return side

    def num_output_tokens(self, num_input_tokens: int) -> int:
        self._grid_side(num_input_tokens)
        return num_input_tokens // (self.scale_factor**2)

    def forward(self, features: Tensor) -> Tensor:
        self._check_input(features)
        b, n, d = features.shape
        side = self._grid_side(n)
        s = self.scale_factor
        x = features.view(b, side, side, d)
        x = x.view(b, side, side // s, d * s)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, side // s, side // s, d * s * s)
        x = x.reshape(b, (side // s) * (side // s), d * s * s)
        return self.proj(x)
