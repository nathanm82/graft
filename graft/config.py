"""Configuration object shared by all connectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from graft.exceptions import ConfigError

#: Activation functions understood by the connectors.
ACTIVATIONS = ("gelu", "relu", "silu", "tanh", "identity")


@dataclass
class ConnectorConfig:
    """Describes how to build a connector.

    The two required fields are ``input_dim`` (the vision encoder's hidden
    size) and ``output_dim`` (the language model's hidden size). Everything
    else has a sensible default; resampler-style connectors additionally read
    ``num_query_tokens``.
    """

    name: str
    input_dim: int
    output_dim: int
    num_input_tokens: int | None = None
    num_query_tokens: int | None = None
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    hidden_dim: int | None = None
    activation: str = "gelu"
    dropout: float = 0.0
    bias: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ConfigError("connector name must be a non-empty string")
        for attr in ("input_dim", "output_dim"):
            value = getattr(self, attr)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"{attr} must be a positive integer, got {value!r}")
        for attr in ("num_input_tokens", "num_query_tokens"):
            value = getattr(self, attr)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ConfigError(f"{attr} must be a positive integer or None, got {value!r}")
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")
        if self.num_heads < 1:
            raise ConfigError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.activation.lower() not in ACTIVATIONS:
            raise ConfigError(
                f"unknown activation {self.activation!r}; choose from {list(ACTIVATIONS)}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` representation, suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConnectorConfig:
        """Build a config from a mapping.

        Unknown keys are tucked into :attr:`extra` rather than raising, which
        keeps user-defined connectors free to add their own knobs.
        """
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = dict(data.get("extra", {}))
        for key, value in data.items():
            if key == "extra":
                continue
            if key in known:
                kwargs[key] = value
            else:
                extra[key] = value
        kwargs["extra"] = extra
        return cls(**kwargs)

    def replace(self, **changes: Any) -> ConnectorConfig:
        """Return a copy with ``changes`` applied (validated again)."""
        return replace(self, **changes)
