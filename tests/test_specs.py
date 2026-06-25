import pytest

from graft import build_connector
from graft.exceptions import ConfigError
from graft.specs import get_encoder_spec, resolve_dims


def test_clip_l_dims():
    spec = get_encoder_spec("clip-vit-l-14")
    assert spec.hidden_dim == 1024
    assert spec.num_tokens == 256


def test_resolve_dims():
    assert resolve_dims("clip-vit-l-14", "llama-2-7b") == (1024, 4096, 256)


def test_unknown_encoder_raises():
    with pytest.raises(ConfigError):
        resolve_dims("not-a-model", "llama-2-7b")


def test_build_with_model_names():
    conn = build_connector("mlp", encoder="clip-vit-l-14", llm="llama-2-7b")
    assert conn.input_dim == 1024
    assert conn.output_dim == 4096
