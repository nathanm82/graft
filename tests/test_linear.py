import torch

from graft import build_connector


def test_linear_preserves_tokens_and_maps_dim():
    conn = build_connector("linear", input_dim=32, output_dim=64)
    out = conn(torch.randn(2, 5, 32))
    assert out.shape == (2, 5, 64)
    assert conn.num_output_tokens(5) == 5
