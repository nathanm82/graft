#!/usr/bin/env bash
# Run the same checks that CI runs.
set -euo pipefail

ruff check .
ruff format --check .
mypy graft
pytest
