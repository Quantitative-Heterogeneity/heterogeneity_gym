import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx
import heterogeneity_gym.rendering as rendering
import cryojax.simulator as cxs


class TestRenderImagesFromAtoms:

    def test_noiseless_batched_rendering(self):
        positions = jnp.array(np.random.randn(10, 4, 3)) * 10
        identities = jnp.array(np.ones(4).astype("int")) * 12
        euler_angles = np.random.randn(10, 6)
        b_factors = None
        defocus = 1.0
        astigmatism = 0.0
        shape = (128, 128)
        pixel_size = 1.0
        voltage = 300
        noise_strength = 1.0
        imgs = rendering._render_clean_images_from_atoms(
            positions,
            identities,
            euler_angles,
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
        euler_angles = np.random.randn(10, 3)
        key = jax.random.PRNGKey(0)
        # keys = jax.random.split(key, 8)

        b_factors = None
        defocus = 1.0
        astigmatism = 0.0
        shape = (128, 128)
        pixel_size = 1.0
        voltage = 300
        noise_strength = 1.0

        imgs = rendering._render_noisy_images_from_atoms(
            positions,
            identities,
            euler_angles,
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


class TestRenderImagesFromPotential:
    def test_noiseless_batched_rendering(self):
        volumes = _build_grid_potentials()
        print(volumes.shape)
        euler_angles = np.random.randn(10, 6)
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
        assert imgs.shape[0] == 10
        assert imgs.shape[1] == 128
        assert imgs.shape[2] == 128

    def test_noisy_batched_rendering(self):
        volumes = _build_grid_potentials()
        euler_angles = np.random.randn(10, 3)
        key = jax.random.PRNGKey(0)
        # keys = jax.random.split(key, 8)

        b_factors = None
        defocus = 1.0
        astigmatism = 0.0
        shape = (128, 128)
        pixel_size = 1.0
        voltage = 300
        noise_strength = 1.0

        imgs = rendering._render_noisy_images_from_potential_grid(
            volumes,
            euler_angles,
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
