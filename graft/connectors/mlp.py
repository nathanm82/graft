"""The MLP projector used by LLaVA v1.5 and friends."""

from __future__ import annotations

from torch import Tensor, nn

from graft._nn import get_activation, init_linear
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.registry import register_connector


@register_connector("mlp")
class MLPConnector(Connector):
    """A multi-layer projector: ``Linear -> act -> ... -> Linear``.

    ``depth`` is the number of linear layers. ``depth=2`` reproduces the
    two-layer GELU projector from LLaVA v1.5. The hidden width defaults to
    ``output_dim`` but can be set via ``hidden_dim``.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        hidden = config.hidden_dim or config.output_dim
        dims = [config.input_dim] + [hidden] * (config.depth - 1) + [config.output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1], bias=config.bias)
            init_linear(linear)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(get_activation(config.activation))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        self._check_input(features)
        return self.mlp(features)
