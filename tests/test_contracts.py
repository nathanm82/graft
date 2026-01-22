import pytest
import torch

from graft import ConnectorConfig, build_connector, list_connectors

CONFIGS = {
    "linear": ConnectorConfig(name="linear", input_dim=32, output_dim=64),
    "mlp": ConnectorConfig(name="mlp", input_dim=32, output_dim=64),
    "avgpool": ConnectorConfig(name="avgpool", input_dim=32, output_dim=64, num_query_tokens=4),
    "pixel_shuffle": ConnectorConfig(name="pixel_shuffle", input_dim=32, output_dim=64),
    "perceiver": ConnectorConfig(
        name="perceiver", input_dim=32, output_dim=64, num_query_tokens=8, num_heads=4
    ),
    "qformer": ConnectorConfig(
        name="qformer", input_dim=32, output_dim=64, num_query_tokens=8, num_heads=4
    ),
}


def test_every_builtin_has_a_contract_config():
    assert set(CONFIGS) == set(list_connectors())


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_output_dim_and_token_count_are_honored(name):
    conn = build_connector(CONFIGS[name])
    num_tokens = 36  # a 6x6 grid, valid for pixel_shuffle too
    out = conn(torch.randn(2, num_tokens, 32))
    assert out.shape[-1] == 64
    assert out.shape[1] == conn.num_output_tokens(num_tokens)


def test_forward_is_deterministic_under_seed():
    torch.manual_seed(0)
    first = build_connector(CONFIGS["perceiver"])
    torch.manual_seed(0)
    second = build_connector(CONFIGS["perceiver"])
    x = torch.randn(1, 36, 32)
    assert torch.allclose(first(x), second(x))
