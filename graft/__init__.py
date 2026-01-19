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
from graft.serialization import load_connector, save_connector
from graft.specs import (
    ENCODER_SPECS,
    LLM_SPECS,
    EncoderSpec,
    LLMSpec,
    get_encoder_spec,
    get_llm_spec,
    resolve_dims,
)
from graft.summary import ConnectorSummary, connector_summary, estimate_sequence_length

__all__ = [
    "__version__",
    "Connector",
    "ConnectorConfig",
    "build_connector",
    "get_connector_class",
    "list_connectors",
    "register_connector",
    "connector_summary",
    "ConnectorSummary",
    "estimate_sequence_length",
    "save_connector",
    "load_connector",
    "resolve_dims",
    "get_encoder_spec",
    "get_llm_spec",
    "EncoderSpec",
    "LLMSpec",
    "ENCODER_SPECS",
    "LLM_SPECS",
    "GraftError",
    "ConfigError",
    "ShapeContractError",
    "ConnectorNotFoundError",
]
