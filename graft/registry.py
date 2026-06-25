"""The connector registry and the :func:`build_connector` factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from graft.config import ConnectorConfig
from graft.exceptions import ConfigError, ConnectorNotFoundError

if TYPE_CHECKING:
    from graft.base import Connector

_REGISTRY: dict[str, type[Connector]] = {}


def register_connector(name: str) -> Callable[[type[Connector]], type[Connector]]:
    """Class decorator that registers a connector under ``name``."""

    def decorator(cls: type[Connector]) -> type[Connector]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"connector {name!r} is already registered")
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_connector_class(name: str) -> type[Connector]:
    """Look up a connector class by registered name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ConnectorNotFoundError(name, available=sorted(_REGISTRY))
    return _REGISTRY[key]


def list_connectors() -> list[str]:
    """Return the sorted list of registered connector names."""
    return sorted(_REGISTRY)


def _config_from_name(name: str, overrides: dict[str, Any]) -> ConnectorConfig:
    overrides = dict(overrides)
    encoder = overrides.pop("encoder", None)
    llm = overrides.pop("llm", None)
    if encoder is not None or llm is not None:
        if encoder is None or llm is None:
            raise ConfigError("pass both encoder and llm to resolve dims, or neither")
        from graft.specs import resolve_dims

        input_dim, output_dim, num_input_tokens = resolve_dims(encoder, llm)
        overrides.setdefault("input_dim", input_dim)
        overrides.setdefault("output_dim", output_dim)
        overrides.setdefault("num_input_tokens", num_input_tokens)
    if "input_dim" not in overrides or "output_dim" not in overrides:
        raise ConfigError(
            f"building {name!r} by name requires input_dim and output_dim "
            "(or an encoder/llm pair, or a ConnectorConfig)"
        )
    return ConnectorConfig(name=name, **overrides)


def build_connector(spec: str | ConnectorConfig | dict[str, Any], **overrides: Any) -> Connector:
    """Build a connector from a name, a dict, or a :class:`ConnectorConfig`.

    Examples
    --------
    >>> build_connector("mlp", input_dim=1024, output_dim=4096)  # doctest: +SKIP
    >>> build_connector({"name": "linear", "input_dim": 768, "output_dim": 2048})  # doctest: +SKIP
    """
    if isinstance(spec, ConnectorConfig):
        config = spec.replace(**overrides) if overrides else spec
    elif isinstance(spec, dict):
        config = ConnectorConfig.from_dict({**spec, **overrides})
    elif isinstance(spec, str):
        config = _config_from_name(spec, overrides)
    else:
        raise TypeError(f"spec must be str, dict, or ConnectorConfig, got {type(spec).__name__}")

    cls = get_connector_class(config.name)
    return cls(config)
