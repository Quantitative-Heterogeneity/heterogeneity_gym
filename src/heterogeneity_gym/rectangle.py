"""
Simple toy example of heterogeneity that attempts to recover images of a 
rectangle of varying width and height.  Initial idea for the example
David Silva Sanchez.  Code by Erik Thiede and Joshua Rhodes
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import random, vmap
from typing import Tuple, Optional
import warnings

class Latent2DGaussianMixture:
    """
    Latent distribution of lengths and widths of four-atom molecule model RectangleModel.
    """

    def __init__(
        self,
        length_width_means: jax.Array = jnp.array([[0., 0.]]),
        length_width_covariances: jax.Array = jnp.array([[[1., 0.], [0., 1.]]]),
        length_width_weights: jax.Array = None,
    ):
        """
        Default construction is a single standard Gaussian.

        Parameters
        ----------
        length_width_means : tensor
            (N, 2) tensor of length, width pairs
        length_width_stds : tensor
            (N, 2, 2) tensor of length, width covariance matrices
        legnth_width_weights : tensor
            (1, N) tensor of weights (not needed to be normalized)
        """
        self.num_gaussians = length_width_means.shape[0]
        self.length_width_means = length_width_means
        self.length_width_covariances = length_width_covariances

        # If no weights are given, default to uniform weighting.
        if length_width_weights is None:
            length_width_weights = jnp.ones(self.num_gaussians)

        self.length_width_weights = length_width_weights / jnp.sum(
            length_width_weights
        )  # Ensure weights are normalized.

    def log_density(self, length_width: jax.Array) -> float:
        """
        Returns value of density at length_width.

        Parameters
        ----------
        length_width : tensor
            (2, ) tensor of length, width positions in length-width space

        Returns
        -------
        density : float
            log pdf evaluated at length_width
        """
        gaussian_log_densities = vmap(jsc.stats.multivariate_normal.logpdf, in_axes=(None, 0, 0))(length_width,
            self.length_width_means, self.length_width_covariances
        )
        log_weights = jnp.log(self.length_width_weights)
        weighted_gaussians = jsc.special.logsumexp(gaussian_log_densities + log_weights)
        return weighted_gaussians.astype(float)

    def calculate_num_samples(
        self, length_width_weights: jax.Array, N: int
    ) -> jax.Array:
        """
        Given a set of mixture weights, calculates how many samples to draw
        from each component for a total of num_samples samples.

        Parameters
        ----------
        length_width_weights : jax.Array
            Weight of each mixture component
        N : int
            Total number of samples to draw

        Returns
        -------
        number_samples : jax.Array[int]
            Number of samples to draw from each component,
            total should sum to N.
        """
        cumulative_num_samples = jnp.round(
            jnp.cumsum(length_width_weights * N,)
        )
        number_samples = jnp.diff(cumulative_num_samples, prepend=jnp.array([0])).astype(int)
        return number_samples

    def sample(self, num_samples: int, rng_key: jax.Array = None, shuffle: bool = True) -> jax.Array:
        """
        Returns lengths, widths samples from model.

        Parameters
        ----------
        num_samples : int
            Number of samples

        Returns
        -------
        sample : tensor
            (num_samples, 2) tensor of length, width pairs
        """
        sample_frequencies = self.calculate_num_samples(
            self.length_width_weights, num_samples
        )
        sample = []

        if rng_key is None:
            rng_key = random.PRNGKey(0)

        for gaussian_index, sample_size in enumerate(sample_frequencies):
            sample_key, rng_key = random.split(rng_key)
            gaussian_sample = random.multivariate_normal(
                key=sample_key,
                mean=self.length_width_means[gaussian_index],
                cov=self.length_width_covariances[gaussian_index],
                shape=(sample_size,),
            )
            sample.append(gaussian_sample)
        sample = jnp.concatenate(sample, 0)
        if shuffle:
            sample = random.permutation(rng_key, sample, )

        return sample

# TODO: Reparameterize to use SNR instead of noise_std
# TODO: Type hints and documentation


class RectangleModel:
    def __init__(
        self,
        latent_density=None,
        image_width_in_pixels=128,
        image_size=4,
        noise_std=0.0,
    ):
        """
        TODO: we should construct a "default" latent density.
        TODO: Add initialization to device.  How does OpenAI solve device when creating their "environments"?
        """
        if latent_density is None:
            latent_means = jnp.array([[0.5, 1.5], [1.0, 1.0]])
            latent_covariances = jnp.array(
                [
                    [[0.1, 0], [0, 0.1]],
                    [[0.3, 0.1], [0.1, 0.2]],
                ]
            )
            latent_weights = jnp.array([1, 4])  # Need not be normalized
            latent_density = Latent2DGaussianMixture(
                length_width_means=latent_means,
                length_width_covariances=latent_covariances,
                length_width_weights=latent_weights,
            )

        self.latent_density = latent_density

        # Atom parameters...
        self.atom_variance = 0.04  # standard deviation of 0.2
        self.image_size = image_size

        # Build the imaging grid
        self.grid_ticks = jnp.linspace(
            -image_size, image_size, image_width_in_pixels + 1,
        )[:-1]
        self.grid = jnp.stack(
            jnp.meshgrid(self.grid_ticks, self.grid_ticks, indexing="xy"), axis=0
        )
        self.noise_std = noise_std

    def evaluate_latent_density(self, x: jax.Array) -> jax.Array:
        """
        Evaluates the probability density function of the two Gaussian mixture model.

        Parameters
        -----------
        x: jax.Array
            Location(s) to evaluate the probability density function.
        """
        return self.latent_density.evaluate_density(x)

    def sample_rotations(self, num_samples: int, rng_key:jax.Array=None) -> jax.Array:
        if rng_key is None:
            rng_key = random.PRNGKey(1)
        theta = random.uniform(rng_key, (num_samples, )) * jnp.pi / 2.0 - jnp.pi / 4.0
        rotations = jnp.stack(
            [jnp.stack([jnp.cos(theta), -jnp.sin(theta)]),
             jnp.stack([jnp.sin(theta), jnp.cos(theta)])]
        )
        rotations = jnp.transpose(rotations, (2, 0, 1))
        return rotations


    def sample_images(
        self, num_samples: int, rng_key:jax.Array=None, sample_rotation=True, shuffle=True
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Returns sampled_images

        Parameters
        ----------
        num_samples : int
            Number of samples
        shuffle: bool
            Whether to shuffle the images

        Returns
        -------
        images: jax.Array
            Images
        structures: jax.Array
            Corresponding images,
        latent_samples: jax.Array
            Corresponding values of the latent.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(4)
        sample_key, pose_key, render_key = random.split(rng_key, 3)
        latent_samples = self.latent_density.sample(num_samples, rng_key=sample_key, shuffle=shuffle)
        if sample_rotation:
            rotations = self.sample_rotations(num_samples, rng_key=pose_key)
        else:
            rotations = None
        images, structures = self.render_images_from_latent(latent_samples, rotations, render_key)
        return images, structures, latent_samples, rotations

    def construct_structures(
        self, latent_samples: jax.Array,
            rotations: jax.Array = None
    ) -> jax.Array:
        """
        Converts a latent distribution on the angles into three-dimensional atomic structures
        """
        if len(latent_samples.shape) < 2:
            latent_samples = jnp.expand_dims(latent_samples, 0)
        N = len(latent_samples)
        dtype = latent_samples.dtype

        # Put atoms in space.
        atom_1 = latent_samples / 2.0
        atom_2 = atom_1 * jnp.array([1, -1], dtype=dtype)
        atom_3 = atom_1 * jnp.array([-1, 1], dtype=dtype)
        atom_4 = atom_1 * jnp.array([-1, -1], dtype=dtype)
        structures = jnp.stack([atom_1, atom_2, atom_3, atom_4], axis=-2).reshape((N, 4, 2))
        if rotations is not None:
            structures = jnp.einsum('nij,nkj->nki', rotations, structures)

        return structures

    def render_images_from_latent(
        self, latent_samples: jax.Array, rotations:jax.Array=None,
            rng_key: jax.Array=None, noise_std: Optional[float] = None
    ) -> Tuple[jax.Array, jax.Array]:
        if noise_std is None:
            noise_std = self.noise_std
        structures = self.construct_structures(latent_samples, rotations)
        images = self.render_images_from_structures(structures, rng_key, noise_std=noise_std)
        return images, structures

    def render_images_from_structures(
        self, structures: jax.Array, rng_key: jax.Array=None, noise_std: Optional[float] = None
    ) -> jax.Array:
        """ """
        if noise_std is None:
            noise_std = self.noise_std
        if rng_key is None:
            rng_key = random.PRNGKey(2)

        expand_structures = structures[..., None, None]  # N x Atom x 2 x 1 x 1
        print(expand_structures.shape, self.grid.shape)
        sq_displacements = (
            expand_structures - self.grid
        ) ** 2  # N x Atom x 2 x Npix x Npix
        sq_distances = jnp.sum(sq_displacements, axis=-3)  # ... x Atom x Npix x Npix
        kernel = jnp.exp(-sq_distances / (2 * self.atom_variance))
        image = jnp.sum(kernel, axis=-3)  # ... x Npix x Npix
        image = image + random.normal(rng_key, image.shape) * noise_std
        return image

    def evaluate_log_pij_matrix(
        self,
        experimental_images: jax.Array,
        simulated_images: jax.Array,
        noise_std: float,
    ) -> jax.Array:
        """ """
        experimental_images = jnp.expand_dims(experimental_images, -4)
        simulated_images = jnp.expand_dims(simulated_images, -3)
        difference = jnp.sum(
            (experimental_images - simulated_images) ** 2, axis=(-1, -2)
        )
        return -1 * difference / (2 * noise_std**2)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    latent_density = Latent2DGaussianMixture(
        jnp.array([[1., 3.], [3., 2.]]),
        jnp.array([[[0.1, 0], [0, 0.1]],
                [[0.3, 0.1], [0.1, 0.2]],]),
        jnp.array([1., 4.])
    )
    samples = latent_density.sample(10)
    print(vmap(latent_density.log_density)(samples))
    noise_std = 0.9
    image_width_in_pixels = 64
    max_N = 1000

    rectangle_model = RectangleModel(latent_density, noise_std=noise_std,
                                     image_width_in_pixels=image_width_in_pixels, )
    clean_model = RectangleModel(latent_density, noise_std=0.0,
                                 image_width_in_pixels=image_width_in_pixels, )

    raw_images, _, latent_samples, _ = rectangle_model.sample_images(max_N, rng_key=random.PRNGKey(10))
    clean_images, _, _, _ = clean_model.sample_images(max_N, rng_key=random.PRNGKey(10))
    print("SNR:", jnp.mean(clean_images * clean_images) / noise_std / noise_std)
    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(16, 8))

    for i, ax in enumerate(axes[0]):
        ax.imshow(raw_images[i], cmap='grey')
    for i, ax in enumerate(axes[1]):
        ax.imshow(clean_images[i], cmap='grey')
    plt.show()
    plt.clf()
    data = []
    for sample in latent_samples:
        data.append({'beta1': float(sample[0]), 'beta2': float(sample[1]), })
    data = pd.DataFrame(data)
    sns.kdeplot(data=data, x='beta1', y='beta2')
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.show()
    plt.clf()
