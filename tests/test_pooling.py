import torch

from graft import ConnectorConfig, build_connector


def test_avgpool_reduces_tokens():
    conn = build_connector(
        ConnectorConfig(name="avgpool", input_dim=32, output_dim=64, num_query_tokens=4)
    )
    out = conn(torch.randn(2, 16, 32))
    assert out.shape == (2, 4, 64)
    assert conn.num_output_tokens(16) == 4


def test_avgpool_without_query_tokens_keeps_all():
    conn = build_connector(ConnectorConfig(name="avgpool", input_dim=32, output_dim=64))
    out = conn(torch.randn(2, 9, 32))
    assert out.shape == (2, 9, 64)
    assert conn.num_output_tokens(9) == 9
