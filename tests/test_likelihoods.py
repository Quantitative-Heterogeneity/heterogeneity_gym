import numpy as np
import jax.numpy as jnp
import jax
import cryojax.simulator as cxs

from heterogeneity_gym import rendering
from heterogeneity_gym import likelihoods


def _build_grid_potentials():
    positions = jnp.array(np.random.randn(10, 4, 3)) * 10
    identities = jnp.array(np.ones(4).astype("int")) * 12
    voxel_size = 1.0
    volumes = []
    for conformation in positions:
        peng_potential = cxs.PengAtomicPotential(
            conformation, identities, b_factors=None
        )

        grid_potential = peng_potential.as_real_voxel_grid(
            shape=(128, 128, 128), voxel_size=voxel_size
        )
        volumes.append(grid_potential)
    return jnp.array(volumes)


class TestLogLikelihoodFromPotential:
    def test_noiseless_batched_rendering(self):
        volumes = _build_grid_potentials()

        euler_angles = jnp.array(np.random.randn(10, 6))
        defocus = 1.0
        astigmatism = 0.0
        shape = (128, 128)
        pixel_size = 1.0
        voltage = 300
        noise_strength = 1.0
        imgs = rendering._render_clean_images_from_potential_grid(
            volumes,
            euler_angles,
            noise_strength,
            defocus,
            astigmatism,
            shape,
            pixel_size,
            voltage,
        )

        likelihood_vals = likelihoods._calculate_log_likelihood_of_potential_grid(
            volumes,
            imgs,
            euler_angles,
            noise_strength,
            defocus,
            astigmatism,
            shape,
            pixel_size,
            voltage,
        )

        assert likelihood_vals.shape[0] == 10
        assert likelihood_vals.shape[1] == 10

        for i in range(len(likelihood_vals)):
            assert jnp.argmax(likelihood_vals[i]) == i
