"""Run a small forward/loss/gradient check for the decoder Transformer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_llm import DecoderLM, TransformerConfig


def main() -> None:
    default_config = TransformerConfig()
    print(f"Default configuration: {default_config.parameter_count:,} parameters")
    config = TransformerConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_size=64,
        max_seq_len=16,
    )
    model = DecoderLM(config, rngs=nnx.Rngs(0))
    token_ids = jnp.array([[1, 5, 9, 2, 7, 3, 4, 6], [8, 2, 1, 3, 5, 7, 9, 4]])
    targets = jnp.roll(token_ids, shift=-1, axis=1)

    logits = model(token_ids)
    assert logits.shape == (2, 8, config.vocab_size), logits.shape
    print(f"Forward pass: logits {logits.shape} on {logits.device}")

    def loss_fn(current_model):
        predictions = current_model(token_ids)[:, :-1, :]
        labels = targets[:, :-1]
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, labels))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    leaves = jax.tree.leaves(gradients)
    if not leaves:
        raise RuntimeError("No parameter gradients were produced")
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
        raise RuntimeError("Model gradients contain non-finite values")
    grad_norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
    if not np.isfinite(float(grad_norm)) or float(grad_norm) <= 0:
        raise RuntimeError(f"Expected a finite, nonzero gradient norm; got {grad_norm}")
    print(f"Causal next-token loss: {float(loss):.5f}; gradient norm: {float(grad_norm):.5f}")


if __name__ == "__main__":
    main()
