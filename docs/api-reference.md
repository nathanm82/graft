# API reference

Everything below is importable from the top-level `graft` package.

## Building

- `build_connector(spec, **overrides) -> Connector` — factory. `spec` is a
  connector name, a dict, or a `ConnectorConfig`. With a name you must supply
  `input_dim`/`output_dim` or an `encoder`/`llm` pair.
- `register_connector(name)` — class decorator to add a connector.
- `get_connector_class(name) -> type[Connector]`
- `list_connectors() -> list[str]`

## Core types

- `Connector` — base `nn.Module`. Key methods: `forward`,
  `num_output_tokens(n)`, `freeze()`, `unfreeze()`, `num_parameters()`.
- `ConnectorConfig` — dataclass of build parameters. Helpers: `to_dict`,
  `from_dict`, `from_yaml`, `replace`.

## Built-in connectors

| name           | class                  | output tokens     |
| -------------- | ---------------------- | ----------------- |
| `linear`       | `LinearConnector`      | unchanged         |
| `mlp`          | `MLPConnector`         | unchanged         |
| `avgpool`      | `AvgPoolConnector`     | `num_query_tokens` or unchanged |
| `pixel_shuffle`| `PixelShuffleConnector`| `n / scale_factor**2` |
| `perceiver`    | `PerceiverResampler`   | `num_query_tokens` |
| `qformer`      | `QFormerConnector`     | `num_query_tokens` |

## Summaries

- `connector_summary(connector, num_input_tokens=None) -> ConnectorSummary`
- `estimate_sequence_length(connector, num_input_tokens, num_text_tokens=0) -> int`

## Specs

- `resolve_dims(encoder, llm) -> (input_dim, output_dim, num_input_tokens)`
- `get_encoder_spec(name) -> EncoderSpec`, `get_llm_spec(name) -> LLMSpec`
- `ENCODER_SPECS`, `LLM_SPECS` — the underlying catalogues.

## Serialization

- `save_connector(connector, path) -> Path`
- `load_connector(path, map_location="cpu") -> Connector`

## Exceptions

`GraftError` is the base. `ConfigError`, `ShapeContractError`, and
`ConnectorNotFoundError` are the concrete types.
