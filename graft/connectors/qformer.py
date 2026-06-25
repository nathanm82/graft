"""A lightweight querying transformer (BLIP-2 inspired)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from graft._nn import CrossAttentionBlock
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.exceptions import ConfigError
from graft.registry import register_connector


@register_connector("qformer")
class QFormerConnector(Connector):
    """Query tokens that self-attend and cross-attend to the visual features.

    This is a trimmed-down Q-Former: ``num_query_tokens`` learned queries pass
    through ``depth`` blocks, each doing self-attention over the queries and
    then cross-attention into the image features. The output is exactly
    ``num_query_tokens`` tokens of size ``output_dim``.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        if config.num_query_tokens is None:
            raise ConfigError("qformer requires num_query_tokens")
        dim = config.output_dim
        self.num_queries = config.num_query_tokens
        self.input_proj = nn.Linear(config.input_dim, dim, bias=config.bias)
        self.queries = nn.Parameter(torch.randn(self.num_queries, dim))
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    self_attention=True,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def num_output_tokens(self, num_input_tokens: int) -> int:
        return self.num_queries

    def forward(self, features: Tensor) -> Tensor:
        self._check_input(features)
        context = self.input_proj(features)
        batch = features.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.norm(queries)
