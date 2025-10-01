"""Exception types raised by graft."""

from __future__ import annotations


class GraftError(Exception):
    """Base class for every error raised by graft."""


class ConfigError(GraftError, ValueError):
    """Raised when a :class:`~graft.config.ConnectorConfig` is invalid."""


class ShapeContractError(GraftError, ValueError):
    """Raised when an input tensor violates a connector's shape contract."""


class ConnectorNotFoundError(GraftError, LookupError):
    """Raised when a connector name cannot be found in the registry."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        message = f"no connector registered under {name!r}"
        if self.available:
            message += f"; available: {', '.join(self.available)}"
        super().__init__(message)
