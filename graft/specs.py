"""A small catalogue of common vision-encoder and LLM dimensions.

The numbers here let you build a connector from model *names* instead of
hand-copying hidden sizes:

>>> from graft import build_connector
>>> conn = build_connector("mlp", encoder="clip-vit-l-14", llm="llama-2-7b")  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass

from graft.exceptions import ConfigError


@dataclass(frozen=True)
class EncoderSpec:
    """Hidden size and token count of a vision encoder."""

    name: str
    hidden_dim: int
    num_tokens: int
    image_size: int
    patch_size: int


@dataclass(frozen=True)
class LLMSpec:
    """Hidden size of a language model."""

    name: str
    hidden_dim: int


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "clip-vit-b-32": EncoderSpec("clip-vit-b-32", 768, 49, 224, 32),
    "clip-vit-b-16": EncoderSpec("clip-vit-b-16", 768, 196, 224, 16),
    "clip-vit-l-14": EncoderSpec("clip-vit-l-14", 1024, 256, 224, 14),
    "clip-vit-l-14-336": EncoderSpec("clip-vit-l-14-336", 1024, 576, 336, 14),
    "siglip-so400m-14-384": EncoderSpec("siglip-so400m-14-384", 1152, 576, 384, 14),
    "dinov2-l-14": EncoderSpec("dinov2-l-14", 1024, 256, 224, 14),
}

LLM_SPECS: dict[str, LLMSpec] = {
    "llama-2-7b": LLMSpec("llama-2-7b", 4096),
    "llama-2-13b": LLMSpec("llama-2-13b", 5120),
    "vicuna-7b": LLMSpec("vicuna-7b", 4096),
    "mistral-7b": LLMSpec("mistral-7b", 4096),
    "qwen2-7b": LLMSpec("qwen2-7b", 3584),
    "phi-2": LLMSpec("phi-2", 2560),
    "gemma-2b": LLMSpec("gemma-2b", 2048),
    "tinyllama-1.1b": LLMSpec("tinyllama-1.1b", 2048),
}


def get_encoder_spec(name: str) -> EncoderSpec:
    try:
        return ENCODER_SPECS[name]
    except KeyError:
        raise ConfigError(
            f"unknown encoder {name!r}; known encoders: {sorted(ENCODER_SPECS)}"
        ) from None


def get_llm_spec(name: str) -> LLMSpec:
    try:
        return LLM_SPECS[name]
    except KeyError:
        raise ConfigError(f"unknown llm {name!r}; known llms: {sorted(LLM_SPECS)}") from None


def resolve_dims(encoder: str, llm: str) -> tuple[int, int, int]:
    """Return ``(input_dim, output_dim, num_input_tokens)`` for a name pair."""
    enc = get_encoder_spec(encoder)
    lm = get_llm_spec(llm)
    return enc.hidden_dim, lm.hidden_dim, enc.num_tokens
