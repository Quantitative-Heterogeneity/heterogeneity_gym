import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx
import heterogeneity_gym.rendering as rendering


class TestRenderImagesFromStructures:

    def test_noiseless_batched_rendering(self):
        positions = jnp.array(np.random.randn(10, 4, 3)) * 10
        identities = jnp.array(np.ones(4).astype("int")) * 12
        b_factors = None
        defocus = 1.0
        astigmatism = 0.0
        shape = (128, 128)
        pixel_size = 1.0
        voltage = 300
        noise_strength = 1.0
        imgs = rendering._render_clean_images_from_structures(
            positions,
            identities,
            b_factors,
            noise_strength,
            defocus,
            astigmatism,
            shape,
            pixel_size,
            voltage,
        )
        assert imgs.shape[0] == 10
        assert imgs.shape[1] == 128
        assert imgs.shape[2] == 128

    def test_noisy_batched_rendering(self):
        positions = jnp.array(np.random.randn(10, 8, 3)) * 10
        identities = jnp.array(np.ones(8).astype("int")) * 12
        key = jax.random.PRNGKey(0)
        # keys = jax.random.split(key, 8)

        b_factors = None
        defocus = 1.0
        astigmatism = 0.0
        shape = (128, 128)
        pixel_size = 1.0
        voltage = 300
        noise_strength = 1.0

        imgs = rendering._render_noisy_images_from_structures(
            positions,
            identities,
            b_factors,
            noise_strength,
            defocus,
            astigmatism,
            shape,
            pixel_size,
            voltage,
            key,
        )

        assert imgs.shape[0] == 10
        assert imgs.shape[1] == 128
        assert imgs.shape[2] == 128
