import pytest
import torch

from graft import ConnectorConfig, build_connector
from graft.exceptions import ConfigError


def test_qformer_outputs_query_tokens():
    cfg = ConnectorConfig(
        name="qformer", input_dim=32, output_dim=64, num_query_tokens=16, num_heads=8, depth=2
    )
    conn = build_connector(cfg)
    out = conn(torch.randn(3, 40, 32))
    assert out.shape == (3, 16, 64)
    assert conn.num_output_tokens(40) == 16


def test_qformer_requires_query_tokens():
    with pytest.raises(ConfigError):
        build_connector(ConnectorConfig(name="qformer", input_dim=32, output_dim=64))
