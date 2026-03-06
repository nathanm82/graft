"""A cheap pooling connector that shrinks the visual token budget."""

from __future__ import annotations

from torch import Tensor, nn

from graft._nn import init_linear
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.registry import register_connector


@register_connector("avgpool")
class AvgPoolConnector(Connector):
    """Adaptive average pooling over the token axis, then a linear projection.

    When ``num_query_tokens`` is set the sequence is pooled down to that many
    tokens; otherwise every token is kept. This is the most parameter-light
    way to reduce the number of visual tokens fed to the LLM.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self.num_queries = config.num_query_tokens
        self.pool = nn.AdaptiveAvgPool1d(self.num_queries) if self.num_queries else None
        self.proj = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)
        init_linear(self.proj)

    def num_output_tokens(self, num_input_tokens: int) -> int:
        return self.num_queries or num_input_tokens

    def forward(self, features: Tensor) -> Tensor:
        self._check_input(features)
        if self.pool is not None:
            # (B, N, D) -> (B, D, N) -> pool -> (B, D, M) -> (B, M, D)
            features = self.pool(features.transpose(1, 2)).transpose(1, 2)
        return self.proj(features)
