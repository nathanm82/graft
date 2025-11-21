"""Built-in connector implementations.

Importing this package registers every built-in connector with the registry.
"""

from __future__ import annotations

from graft.connectors.linear import LinearConnector
from graft.connectors.mlp import MLPConnector
from graft.connectors.perceiver import PerceiverResampler
from graft.connectors.pixel_shuffle import PixelShuffleConnector
from graft.connectors.pooling import AvgPoolConnector
from graft.connectors.qformer import QFormerConnector

__all__ = [
    "LinearConnector",
    "MLPConnector",
    "AvgPoolConnector",
    "PixelShuffleConnector",
    "PerceiverResampler",
    "QFormerConnector",
]
