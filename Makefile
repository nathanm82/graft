.PHONY: install lint format test check

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy graft

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test
