# Architecture

`graft` is intentionally small. There are four moving parts.

## The connector contract

Every connector is an `nn.Module` subclass of `graft.base.Connector` with a
single, well-defined contract:

```
forward(features: Tensor[B, N, D_in]) -> Tensor[B, M, D_out]
```

- `D_in` is the vision encoder's hidden size (`config.input_dim`).
- `D_out` is the LLM's hidden size (`config.output_dim`).
- `N` is the number of incoming visual tokens; `M = num_output_tokens(N)`.

`Connector._check_input` validates the rank and the trailing dimension before
any module touches the tensor, so a mismatched encoder fails loudly instead of
producing a silently-wrong shape deep inside an attention block.

Token-preserving connectors (`linear`, `mlp`) leave `M == N`. Resampler-style
connectors (`avgpool`, `pixel_shuffle`, `perceiver`, `qformer`) override
`num_output_tokens` to advertise their fixed or reduced budget.

## Config

`ConnectorConfig` is a plain dataclass. It is the *only* thing a connector
needs to build itself, and it round-trips through `to_dict`/`from_dict` (and
`from_yaml`), which is what makes serialization and the CLI trivial. Unknown
keys are stashed in `config.extra`, so a custom connector can read its own
knobs (e.g. `scale_factor`) without changing the dataclass.

## Registry

Connectors register themselves with the `@register_connector("name")`
decorator at import time. `build_connector` is the single factory: give it a
name + dims, a dict, or a `ConnectorConfig`, and it returns a ready module.
The registry is also what powers `graft list` and lets you drop in third-party
connectors without patching the library.

## Spec catalogue

`graft.specs` maps common encoder/LLM names to their hidden sizes and token
counts so you can write `build_connector("mlp", encoder="clip-vit-l-14",
llm="llama-2-7b")` instead of memorising that CLIP ViT-L/14 is 1024-d with 256
tokens. It is a convenience layer only — nothing else depends on it.
