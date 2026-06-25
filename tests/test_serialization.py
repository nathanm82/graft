import pytest
import torch

from graft import ConnectorConfig, build_connector
from graft.serialization import load_connector, save_connector


def test_save_load_roundtrip(tmp_path):
    cfg = ConnectorConfig(name="mlp", input_dim=16, output_dim=32)
    conn = build_connector(cfg).eval()
    path = save_connector(conn, tmp_path / "conn.pt")
    loaded = load_connector(path).eval()
    x = torch.randn(1, 4, 16)
    assert torch.allclose(conn(x), loaded(x))


def test_loaded_connector_keeps_config(tmp_path):
    cfg = ConnectorConfig(
        name="perceiver", input_dim=16, output_dim=32, num_query_tokens=8, num_heads=4
    )
    path = save_connector(build_connector(cfg), tmp_path / "p.pt")
    loaded = load_connector(path)
    assert loaded.config.name == "perceiver"
    assert loaded.config.num_query_tokens == 8


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_connector(tmp_path / "nope.pt")
