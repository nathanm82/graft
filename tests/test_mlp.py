import torch

from graft import ConnectorConfig, build_connector


def test_mlp_shapes():
    conn = build_connector("mlp", input_dim=32, output_dim=64)
    out = conn(torch.randn(2, 7, 32))
    assert out.shape == (2, 7, 64)


def test_mlp_depth_controls_layer_count():
    deep = build_connector(
        ConnectorConfig(name="mlp", input_dim=32, output_dim=64, depth=3)
    )
    linears = [m for m in deep.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3


def test_mlp_inserts_dropout_when_configured():
    conn = build_connector(
        ConnectorConfig(name="mlp", input_dim=8, output_dim=16, dropout=0.1)
    )
    assert any(isinstance(m, torch.nn.Dropout) for m in conn.modules())
