# Repository Guidelines

## Project Structure

- `jax_llm/` contains the model implementation. `model.py` defines the Flax NNX decoder, attention, and Transformer configuration.
- `scripts/` contains runnable environment and model sanity checks; keep these small and focused.
- `roadmap.md` describes the planned pretraining, supervised fine-tuning, and verifier-guided training stages.
- `pyproject.toml`, `uv.lock`, and `.python-version` define dependencies, resolved versions, and the Python version. The project targets Python 3.12 and JAX with CUDA 13.
- Add automated tests under `tests/` as the codebase grows. There are no data or model assets checked in currently.

## Build and Development Commands

- `uv sync --all-groups` creates or updates `.venv` with development and CUDA dependencies.
- `./scripts/check.sh` runs Ruff lint, formats Python files, verifies formatting, runs mypy, and runs pytest; it stops on the first failure.
- `uv run pytest` runs the test suite; name test modules `test_*.py`.
- `uv run mypy jax_llm scripts tests` checks project type annotations. `uv run ruff check .` checks Python style.
- Run `./scripts/check.sh` after making updates.

## Coding Style

Use Python 3.12 syntax, four spaces, strict typing, and type hints for functions and public interfaces. Write Google-style docstrings and use descriptive `snake_case` names. Keep functions small and cohesive, and do not put implementations in `__init__.py` files. Keep model components as small Flax NNX modules and put shared architecture settings in `TransformerConfig`. Preserve deterministic seeds in examples and sanity checks. Keep line length at 100 characters, as configured in Ruff.

Prefer standard-library tools before adding dependencies. Avoid shims; prefer clean replacements for legacy code.

## JAX and JIT

- Keep computations intended for `jax.jit`, `jax.grad`, or `jax.vmap` in terms of JAX operations and arrays; avoid NumPy conversions, Python-side mutation, and side effects inside transformed functions.
- Treat array shapes, dtypes, and loop structure as compile-time design choices. Prefer fixed shapes and JAX control-flow primitives (`lax.scan`, `lax.cond`, `lax.fori_loop`) when control flow depends on traced values.
- Do not convert traced values to Python scalars or use them in Python conditionals. Keep configuration and other compile-time options explicit and static where needed.
- For Flax NNX modules, keep parameters and mutable model state in the module graph; use the NNX-aware transformation patterns when JIT-compiling module calls rather than closing over mutable module state.
- Remember that JIT compiles per shape/dtype and static-argument combination. Avoid creating new jitted functions repeatedly in hot paths, and keep compilation separate from timed execution when benchmarking.
- Preserve deterministic PRNG behavior by passing and splitting JAX keys explicitly; do not rely on global random state.

## Testing Guidelines

Run the relevant sanity script after changing the model or training path. Add focused pytest coverage for new math, masking, data, and checkpoint behavior; use tiny configurations so tests run quickly. GPU-dependent checks require an NVIDIA GPU and the CUDA dependency group. Do not treat a successful import alone as evidence that device execution works.

## Commits and Pull Requests

The Git history contains only the initial commit, so no established commit convention is available. Use short imperative subjects, such as `Add causal attention mask`. Pull requests should explain the change, list validation commands and results, link related roadmap work, and include sample outputs when model behavior changes. Never commit datasets, credentials, virtual environments, or generated checkpoints.

## Configuration and Artifacts

Keep dependency changes in `pyproject.toml` and refresh `uv.lock` with `uv lock`. Store local datasets, logs, and checkpoints outside tracked source files; record any paths or configuration needed to reproduce an experiment.
