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

## Quick start

Create a small model and run a batch of token IDs through it:

```python
import jax.numpy as jnp
from flax import nnx

from jax_llm import DecoderLM, TransformerConfig

config = TransformerConfig(
    vocab_size=128,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    mlp_size=128,
    max_seq_len=32,
)
model = DecoderLM(config, rngs=nnx.Rngs(0))

token_ids = jnp.array([[1, 5, 9, 2]])  # Shape: [batch, sequence]
logits = model(token_ids)  # Shape: [batch, sequence, vocab_size]
print(logits.shape)
```

`token_ids` must be a rank-2 integer array, and its sequence length cannot
exceed `max_seq_len`. The output contains a vocabulary-sized logit vector at
each input position. The model accepts token IDs directly; tokenizer loading
and text generation are not implemented yet.

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
