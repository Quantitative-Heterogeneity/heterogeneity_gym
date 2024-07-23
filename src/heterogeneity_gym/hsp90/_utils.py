import jax
import jax.numpy as jnp
from typing import Optional, Tuple


def initialize_latent_code(
    num_images: int, latent_code: Optional[int] = None, key=None
):
    if key is None:
        key = jax.random.key(0)

    if latent_code is None:
        latent_code = jax.random.randint(key, (num_images,), 0, 20)
    else:
        latent_code = jnp.ones(num_images, dtype="int32") * latent_code
    return latent_code
