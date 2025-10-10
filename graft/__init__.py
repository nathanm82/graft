"""graft: lightweight visual connectors that bridge vision encoders to LLMs."""

from __future__ import annotations

from graft import connectors as connectors  # noqa: F401  (registers built-ins)
from graft.__about__ import __version__
from graft.base import Connector
from graft.config import ConnectorConfig
from graft.exceptions import (
    ConfigError,
    ConnectorNotFoundError,
    GraftError,
    ShapeContractError,
)
from graft.registry import (
    build_connector,
    get_connector_class,
    list_connectors,
    register_connector,
)

__all__ = [
    "__version__",
    "Connector",
    "ConnectorConfig",
    "build_connector",
    "get_connector_class",
    "list_connectors",
    "register_connector",
    "GraftError",
    "ConfigError",
    "ShapeContractError",
    "ConnectorNotFoundError",
]
