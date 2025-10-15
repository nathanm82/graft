"""The simplest connector: a single linear projection."""

from __future__ import annotations

from torch import Tensor, nn

from graft._nn import init_linear
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.registry import register_connector


@register_connector("linear")
class LinearConnector(Connector):
    """A single linear projection from ``input_dim`` to ``output_dim``.

    This is the connector used by the original LLaVA: it keeps every visual
    token and simply re-projects the channel dimension.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)
        init_linear(self.proj)

    def forward(self, features: Tensor) -> Tensor:
        self._check_input(features)
        return self.proj(features)
