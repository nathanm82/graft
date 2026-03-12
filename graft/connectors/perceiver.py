"""A Perceiver-style resampler (Flamingo / IDEFICS)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from graft._nn import CrossAttentionBlock
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.exceptions import ConfigError
from graft.registry import register_connector


@register_connector("perceiver")
class PerceiverResampler(Connector):
    """Compress a variable number of visual tokens to a fixed query budget.

    A set of ``num_query_tokens`` learned latent vectors repeatedly
    cross-attend to the projected visual features, producing exactly
    ``num_query_tokens`` output tokens regardless of the input length.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        if config.num_query_tokens is None:
            raise ConfigError("perceiver requires num_query_tokens")
        dim = config.output_dim
        self.num_queries = config.num_query_tokens
        self.input_proj = nn.Linear(config.input_dim, dim, bias=config.bias)
        self.latents = nn.Parameter(torch.empty(self.num_queries, dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(dim, config.num_heads, config.mlp_ratio, config.dropout)
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
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        # TODO: cache the expanded latents when the batch size is static.
        for block in self.blocks:
            latents = block(latents, context)
        return self.norm(latents)
