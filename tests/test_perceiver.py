import pytest
import torch

from graft import ConnectorConfig, build_connector
from graft.exceptions import ConfigError


def test_perceiver_outputs_query_tokens():
    cfg = ConnectorConfig(
        name="perceiver", input_dim=32, output_dim=64, num_query_tokens=8, num_heads=4
    )
    conn = build_connector(cfg)
    out = conn(torch.randn(2, 50, 32))
    assert out.shape == (2, 8, 64)
    assert conn.num_output_tokens(50) == 8


def test_perceiver_length_invariant():
    cfg = ConnectorConfig(
        name="perceiver", input_dim=32, output_dim=64, num_query_tokens=8, num_heads=4
    )
    conn = build_connector(cfg)
    short = conn(torch.randn(1, 10, 32))
    long = conn(torch.randn(1, 200, 32))
    assert short.shape == long.shape == (1, 8, 64)


def test_perceiver_requires_query_tokens():
    with pytest.raises(ConfigError):
        build_connector(ConnectorConfig(name="perceiver", input_dim=32, output_dim=64))
