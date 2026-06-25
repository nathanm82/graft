import pytest
import torch

from graft import build_connector
from graft.exceptions import ShapeContractError


def test_rejects_non_3d_input():
    conn = build_connector("linear", input_dim=8, output_dim=16)
    with pytest.raises(ShapeContractError):
        conn(torch.randn(4, 8))


def test_rejects_wrong_input_dim():
    conn = build_connector("linear", input_dim=8, output_dim=16)
    with pytest.raises(ShapeContractError):
        conn(torch.randn(1, 4, 9))


def test_freeze_and_unfreeze():
    conn = build_connector("linear", input_dim=8, output_dim=16)
    conn.freeze()
    assert all(not p.requires_grad for p in conn.parameters())
    assert conn.num_parameters(trainable_only=True) == 0
    conn.unfreeze()
    assert all(p.requires_grad for p in conn.parameters())
    assert conn.num_parameters(trainable_only=True) == conn.num_parameters()
