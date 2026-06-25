# Contributing

Thanks for taking the time to contribute!

## Development setup

```bash
python -m pip install -e ".[dev]"
pre-commit install   # optional, runs ruff on commit
```

`torch` is a runtime dependency; on CI it is installed from the CPU wheel
index to keep things fast:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Checks

The same three checks run in CI:

```bash
ruff check .
ruff format --check .
mypy graft
pytest
```

`ruff` is pinned to an exact version so `ruff format` output is stable across
machines — please don't bump it in an unrelated PR.

## Adding a connector

1. Create `graft/connectors/<name>.py` with a `Connector` subclass decorated
   with `@register_connector("<name>")`.
2. Import it in `graft/connectors/__init__.py` so it registers on import.
3. Override `num_output_tokens` if your connector changes the token count.
4. Add a test under `tests/` and an entry to `docs/api-reference.md`.

Keep each PR focused and add a `CHANGELOG.md` entry under "Unreleased".
