"""A compact decoder-only Transformer built from Flax NNX layers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 16_000
    hidden_size: int = 640
    num_layers: int = 10
    num_heads: int = 10
    mlp_size: int = 2_560
    max_seq_len: int = 512
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if min(self.vocab_size, self.hidden_size, self.num_layers, self.num_heads) <= 0:
            raise ValueError("vocabulary size, dimensions, and layer count must be positive")
        if self.mlp_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("mlp_size and max_seq_len must be positive")

    @property
    def parameter_count(self) -> int:
        hidden = self.hidden_size
        per_block = 4 * hidden**2 + 2 * hidden * self.mlp_size + 2 * hidden
        return self.vocab_size * hidden + self.num_layers * per_block + hidden


def _apply_rope(x: jnp.ndarray, base: float) -> jnp.ndarray:
    """Apply interleaved rotary position embeddings to [batch, time, heads, dim]."""
    seq_len, head_dim = x.shape[1], x.shape[-1]
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    frequencies = base ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    angles = positions[:, None] * frequencies[None, :]
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]

    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(x.shape)


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rope_base = config.rope_base
        self.qkv = nnx.Linear(config.hidden_size, 3 * config.hidden_size, use_bias=False, rngs=rngs)
        self.output = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = jnp.moveaxis(qkv, 2, 0)
        query = _apply_rope(query, self.rope_base)
        key = _apply_rope(key, self.rope_base)

        scores = jnp.einsum("bthd,bshd->bhts", query, key) / jnp.sqrt(self.head_dim)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhts,bshd->bthd", weights, value)
        attended = attended.reshape(batch_size, seq_len, hidden_size)
        return self.output(attended)


class FeedForward(nnx.Module):
    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.up = nnx.Linear(config.hidden_size, config.mlp_size, use_bias=False, rngs=rngs)
        self.down = nnx.Linear(config.mlp_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down(jax.nn.gelu(self.up(x)))


class DecoderBlock(nnx.Module):
    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.attention_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.attention = CausalSelfAttention(config, rngs=rngs)
        self.feed_forward_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.feed_forward = FeedForward(config, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class DecoderLM(nnx.Module):
    """Decoder-only Transformer with RoPE, RMSNorm, and tied token embeddings."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.blocks = nnx.List([DecoderBlock(config, rngs=rngs) for _ in range(config.num_layers)])
        self.final_norm = nnx.RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds configured max_seq_len")
        hidden_states = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return self.token_embedding.attend(hidden_states)
