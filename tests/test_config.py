import pytest

from graft import ConnectorConfig
from graft.exceptions import ConfigError


def test_defaults():
    cfg = ConnectorConfig(name="mlp", input_dim=1024, output_dim=4096)
    assert cfg.depth == 2
    assert cfg.activation == "gelu"
    assert cfg.num_query_tokens is None
    assert cfg.extra == {}


def test_to_from_dict_roundtrip():
    cfg = ConnectorConfig(
        name="perceiver", input_dim=1024, output_dim=4096, num_query_tokens=32
    )
    assert ConnectorConfig.from_dict(cfg.to_dict()) == cfg


def test_unknown_keys_go_to_extra():
    cfg = ConnectorConfig.from_dict(
        {"name": "pixel_shuffle", "input_dim": 1024, "output_dim": 4096, "scale_factor": 2}
    )
    assert cfg.extra["scale_factor"] == 2


def test_replace_returns_validated_copy():
    cfg = ConnectorConfig(name="mlp", input_dim=1024, output_dim=4096)
    bigger = cfg.replace(output_dim=5120)
    assert bigger.output_dim == 5120
    assert cfg.output_dim == 4096


def test_from_yaml(tmp_path):
    path = tmp_path / "connector.yaml"
    path.write_text("name: mlp\ninput_dim: 1024\noutput_dim: 4096\ndepth: 3\n")
    cfg = ConnectorConfig.from_yaml(path)
    assert cfg.depth == 3
    assert cfg.input_dim == 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "mlp", "input_dim": 0, "output_dim": 4096},
        {"name": "mlp", "input_dim": 1024, "output_dim": -1},
        {"name": "", "input_dim": 1024, "output_dim": 4096},
        {"name": "mlp", "input_dim": 1024, "output_dim": 4096, "activation": "banana"},
        {"name": "mlp", "input_dim": 1024, "output_dim": 4096, "dropout": 1.5},
        {"name": "mlp", "input_dim": 1024, "output_dim": 4096, "depth": 0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ConfigError):
        ConnectorConfig(**kwargs)
