# JAX LLM

A small math-focused language model built from scratch with JAX. The project is
intended to make the model and its training stages understandable and easy to
experiment with, rather than hiding them behind a high-level training framework.

## Current status

The repository currently contains a decoder-only Transformer implemented with
Flax NNX, plus small checks for its forward pass, next-token loss, gradients, and
the JAX environment. It does not yet include a dataset pipeline, tokenizer
training, a reusable training loop, text generation, or the planned SFT and RLVR
stages. Those are described in the [roadmap](roadmap.md).

The default `TransformerConfig` describes a 10-layer, 640-hidden-size model with
a 16,000-token vocabulary and a 512-token context. Its parameter count can be
inspected in Python. The default configuration is a design starting point; it
does not imply that a model has been trained or that these settings fit every
GPU.

## Features

- Decoder-only causal Transformer in Flax NNX.
- Rotary positional embeddings (RoPE), RMSNorm, and GELU feed-forward blocks.
- Tied input token embeddings and output projection weights.
- Configurable vocabulary, width, layer count, head count, MLP width, and context.
- Example sanity scripts and focused tests for output shapes and causal masking.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- JAX 0.12.0.1 and the dependencies declared in `pyproject.toml`

The default dependency group installs JAX according to its standard pip
installation. For NVIDIA GPU support using CUDA 13, install the `cuda` extra
group as described below. GPU availability also depends on a compatible NVIDIA
driver and hardware; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for current requirements.

## Setup

Install the default and development dependencies:

```bash
uv sync --all-groups
```

For the CUDA 13 dependency group on a compatible NVIDIA system:

```bash
uv sync --all-groups --extra cuda
```

## Checks

To run the test suite and code checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy jax_llm scripts tests
```

`./scripts/check.sh` runs the repository's full check sequence, including Ruff,
formatting, mypy, and pytest.

## Repository layout

```text
jax_llm/model.py          Transformer configuration and model components
scripts/                  Environment and model sanity checks
tests/                    Focused model tests
roadmap.md                Planned pretraining, SFT, and RLVR work
```

## Development

See the [roadmap](roadmap.md) for the planned path from the model skeleton to
pretraining, supervised fine-tuning (SFT), verifier-guided reinforcement
learning (RLVR), evaluation, and inference. Keep datasets, logs, and checkpoints
outside the tracked source tree. If you add data or training features, include
focused tests and update the README to describe the implemented behavior.
