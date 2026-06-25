# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.2.0] - 2026-06-25

### Added
- Encoder/LLM spec catalogue and name-based `build_connector`.
- `connector_summary` and `estimate_sequence_length` for token-budget planning.
- `save_connector` / `load_connector`.
- `graft` command line interface (`list`, `specs`, `describe`, `plan`).
- Optional YAML config loading via `ConnectorConfig.from_yaml`.

## [0.1.0] - 2026-06-25

### Added
- `Connector` base class with a strict `(B, N, D_in) -> (B, M, D_out)` shape
  contract.
- Built-in connectors: `linear`, `mlp`, `avgpool`, `pixel_shuffle`,
  `perceiver`, `qformer`.
- `ConnectorConfig` dataclass with validation and dict round-tripping.
- Connector registry and the `build_connector` factory.
