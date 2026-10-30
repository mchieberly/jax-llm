"""JAX Transformer"""

from dataclasses import dataclass, fields
from flax import nnx
import jax
import jax.numpy as jnp


class RMSNorm(nnx.Module):
	"""Root Mean Square Normalization"""

	def __init__(self, hidden_size: int, epsilon: float = 1e-6) -> None:
		"""Initialize the module"""
		self.scale = nnx.Param(jnp.ones((hidden_size,)))
		self.epsilon = epsilon

	def __call__(self, x: jax.Array) -> jax.Array:
		"""Call the RMSNorm"""
		# TODO
		raise NotImplementedError()


@dataclass
class TransformerConfig:
	"""Configuration for the JAX Transformer"""
	vocab_size: int = 16_000
	hidden_size: int = 640
	num_layers: int = 10
	num_heads: int = 10
	mlp_size: int = 2560
	max_seq_len: int = 512

	def __post_init__(self) -> None:
		"""Validate the configuration"""
		for field in fields(self):
			value = getattr(self, field.name)
			if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
				raise ValueError(
					f"{field.name} must be a positive integer, got {value!r}"
				)

		if self.hidden_size % self.num_heads != 0:
			raise ValueError("hidden_size must be divisible by num_heads")

class Transformer:
	"""JAX Transformer for an LLM"""

	def __init__(self):
		"""Initialize the Transformer"""
		raise NotImplementedError()


# def main():
# 	"""Main entrypoint"""

# if __name__ == "__main__":
# 	raise SystemExit(main())
