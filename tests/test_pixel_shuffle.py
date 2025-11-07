import pytest
import torch

from graft import ConnectorConfig, build_connector
from graft.exceptions import ShapeContractError


def test_pixel_shuffle_token_math():
    conn = build_connector(
        ConnectorConfig(
            name="pixel_shuffle", input_dim=8, output_dim=64, extra={"scale_factor": 2}
        )
    )
    # 36 tokens -> 6x6 grid -> /2 -> 3x3 = 9 tokens
    out = conn(torch.randn(2, 36, 8))
    assert out.shape == (2, 9, 64)
    assert conn.num_output_tokens(36) == 9


def test_pixel_shuffle_requires_square_grid():
    conn = build_connector(ConnectorConfig(name="pixel_shuffle", input_dim=8, output_dim=64))
    with pytest.raises(ShapeContractError):
        conn(torch.randn(1, 30, 8))


def test_pixel_shuffle_grid_must_divide_scale():
    conn = build_connector(
        ConnectorConfig(
            name="pixel_shuffle", input_dim=8, output_dim=64, extra={"scale_factor": 4}
        )
    )
    # 36 -> 6x6, 6 not divisible by 4
    with pytest.raises(ShapeContractError):
        conn(torch.randn(1, 36, 8))
