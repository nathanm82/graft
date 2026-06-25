"""The :class:`Connector` base class and its shape contract."""

from __future__ import annotations

from torch import Tensor, nn

from graft.config import ConnectorConfig
from graft.exceptions import ShapeContractError


class Connector(nn.Module):
    """Base class for every visual connector.

    A connector maps vision-encoder features of shape
    ``(batch, num_input_tokens, input_dim)`` to language-model-ready
    embeddings of shape ``(batch, num_output_tokens, output_dim)``.

    Token-preserving connectors (linear, MLP) keep the token count; resampler
    connectors compress it to a fixed number of query tokens. Subclasses
    override :meth:`num_output_tokens` when the count changes.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def num_output_tokens(self, num_input_tokens: int) -> int:
        """Number of output tokens produced for ``num_input_tokens`` inputs."""
        return num_input_tokens

    def forward(self, features: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def freeze(self) -> Connector:
        """Set ``requires_grad = False`` on every parameter; returns ``self``."""
        for param in self.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze(self) -> Connector:
        """Set ``requires_grad = True`` on every parameter; returns ``self``."""
        for param in self.parameters():
            param.requires_grad_(True)
        return self

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total number of parameters (optionally only trainable ones)."""
        return sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad or not trainable_only
        )

    def _check_input(self, features: Tensor) -> None:
        if features.dim() != 3:
            raise ShapeContractError(
                "expected a 3D tensor (batch, tokens, dim), "
                f"received shape {tuple(features.shape)}"
            )
        if features.shape[-1] != self.input_dim:
            raise ShapeContractError(
                f"expected last dimension {self.input_dim}, got {features.shape[-1]}"
            )

    def extra_repr(self) -> str:
        return (
            f"name={self.config.name!r}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )
