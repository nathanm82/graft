"""Save and load connectors (architecture config + weights)."""

from __future__ import annotations

from pathlib import Path

import torch

from graft.base import Connector
from graft.config import ConnectorConfig
from graft.registry import build_connector

PathLike = str | Path


def save_connector(connector: Connector, path: PathLike) -> Path:
    """Save a connector's config and weights to ``path``; returns the path."""
    target = Path(path)
    payload = {
        "config": connector.config.to_dict(),
        "state_dict": connector.state_dict(),
    }
    torch.save(payload, target)
    return target


def load_connector(path: PathLike, map_location: str = "cpu") -> Connector:
    """Rebuild a connector previously written by :func:`save_connector`."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no connector checkpoint at {source}")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    config = ConnectorConfig.from_dict(payload["config"])
    connector = build_connector(config)
    connector.load_state_dict(payload["state_dict"])
    return connector
