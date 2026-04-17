import pytest

from graft import (
    ConnectorConfig,
    build_connector,
    get_connector_class,
    list_connectors,
    register_connector,
)
from graft.base import Connector
from graft.exceptions import ConnectorNotFoundError


def test_builtins_are_registered():
    names = list_connectors()
    for expected in ["linear", "mlp", "avgpool", "pixel_shuffle", "perceiver", "qformer"]:
        assert expected in names


def test_build_from_dict():
    conn = build_connector({"name": "linear", "input_dim": 16, "output_dim": 32})
    assert isinstance(conn, Connector)
    assert conn.output_dim == 32


def test_build_from_config_with_override():
    cfg = ConnectorConfig(name="linear", input_dim=16, output_dim=32)
    conn = build_connector(cfg, output_dim=48)
    assert conn.output_dim == 48


def test_unknown_connector_raises():
    with pytest.raises(ConnectorNotFoundError):
        get_connector_class("does-not-exist")


def test_build_rejects_unknown_spec_type():
    with pytest.raises(TypeError):
        build_connector(123)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):

        @register_connector("linear")
        class _Dupe(Connector):
            pass
