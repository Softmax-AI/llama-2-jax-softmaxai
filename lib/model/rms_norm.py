from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
from jax.sharding import PositionalSharding

from .ModelConfig import ModelConfig

def check_rms_norm(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.d_model,)

def create_model_parallel_sharding_norm(sharding: PositionalSharding) -> PositionalSharding:
    return sharding.replicate((0,))

# Taken from https://github.com/ztjhz/t5-jax/blob/main/model/layer_norm.py#L23
@partial(jax.jit, static_argnames=('model_config',))
def rms_norm(params: Array, x: Array, *, model_config: ModelConfig) -> Array:
    x_rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + model_config.rms_norm_eps)
    y = x / x_rms * params
    return y