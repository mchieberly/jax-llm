#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "==> Ruff lint"
uv run ruff check .

echo "==> Ruff format (apply formatting)"
uv run ruff format .

echo "==> Ruff format (verify formatting)"
uv run ruff format --check .

echo "==> Mypy"
uv run mypy jax_llm scripts tests

echo "==> Pytest"
uv run pytest
