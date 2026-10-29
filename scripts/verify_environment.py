"""Check GPU execution, gradients, optimization, and Orbax checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp


def main() -> None:
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(f"JAX found no GPU; available devices: {devices}")
    device = gpu_devices[0]
    print(f"JAX {jax.__version__}; GPU: {device}")

    with jax.default_device(device):
        a = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        product = a @ a.T
        product.block_until_ready()
        if product.device != device:
            raise RuntimeError(f"Matrix multiplication ran on {product.device}, expected {device}")
        print(f"Matrix multiplication: {product.shape} on {product.device}")

        gradient = jax.grad(lambda value: jnp.sum(value**2))(a)
        np.testing.assert_allclose(gradient, 2 * a)
        print("Automatic differentiation: gradient check passed")

        inputs = jnp.linspace(-1.0, 1.0, 32)
        targets = 3.0 * inputs + 2.0
        params = {"weight": jnp.array(0.0), "bias": jnp.array(0.0)}
        optimizer = optax.adam(learning_rate=0.1)
        optimizer_state = optimizer.init(params)

        def loss_fn(current_params):
            predictions = current_params["weight"] * inputs + current_params["bias"]
            return jnp.mean((predictions - targets) ** 2)

        initial_loss = float(loss_fn(params))
        for _ in range(100):
            _, gradients = jax.value_and_grad(loss_fn)(params)
            updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
            params = optax.apply_updates(params, updates)
        final_loss = float(loss_fn(params))
        if not final_loss < initial_loss:
            raise RuntimeError(f"Training loss did not decrease: {initial_loss} -> {final_loss}")
        print(f"Optimizer step: loss {initial_loss:.5f} -> {final_loss:.5f}")

        with tempfile.TemporaryDirectory(prefix="jax-llm-checkpoint-") as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoint"
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(checkpoint_dir, params)
            restored = checkpointer.restore(checkpoint_dir, target=params)
            checkpointer.close()
            for key in params:
                np.testing.assert_allclose(restored[key], params[key])
        print("Orbax checkpoint: save and restore passed")


if __name__ == "__main__":
    main()
