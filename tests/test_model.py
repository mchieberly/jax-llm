import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_llm import DecoderLM, TransformerConfig


def make_small_model() -> DecoderLM:
    config = TransformerConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        mlp_size=32,
        max_seq_len=8,
    )
    return DecoderLM(config, rngs=nnx.Rngs(0))


def test_decoder_returns_logits_for_each_token() -> None:
    model = make_small_model()
    token_ids = jnp.array([[1, 2, 3], [4, 5, 6]])

    logits = model(token_ids)

    assert logits.shape == (2, 3, model.config.vocab_size)


def test_future_tokens_do_not_change_earlier_logits() -> None:
    model = make_small_model()
    token_ids = jnp.array([[1, 2, 3, 4]])
    changed_future = jnp.array([[1, 2, 3, 9]])

    original_logits = model(token_ids)
    changed_logits = model(changed_future)

    np.testing.assert_allclose(original_logits[:, :3], changed_logits[:, :3], atol=1e-6)
