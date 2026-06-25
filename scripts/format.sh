#!/usr/bin/env bash
# Auto-fix lint issues and format the code base.
set -euo pipefail

ruff check --fix .
ruff format .
