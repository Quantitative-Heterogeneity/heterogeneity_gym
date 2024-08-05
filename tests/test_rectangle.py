import jax.numpy as jnp
from jax import random
import pytest
from src.heterogeneity_gym import rectangle

def test_normalization():
    key = random.PRNGKey(0)
    latent_values = random.normal(key, (8, 2))
    rectangle_model = rectangle.RectangleModel(None)
    images = rectangle_model.render_images_from_latent(latent_values)[0]

    dx = rectangle_model.grid_ticks[1] - rectangle_model.grid_ticks[0]

    image_integral = jnp.sum(images, axis=(1, 2)) * dx**2
    avg_integral = jnp.mean(image_integral)
    assert jnp.all((image_integral - avg_integral) < 2e-3)


def test_square_images_are_symmetric():
    key = random.PRNGKey(0)
    latent_values = random.normal(key, (16, ))
    latent_values = jnp.stack([latent_values, latent_values], axis=1)
    rectangle_model = rectangle.RectangleModel(None)
    images = rectangle_model.render_images_from_latent(latent_values)[0]
    print(images.shape)
    image_transpose = jnp.transpose(images, (0, 2, 1))
    assert jnp.allclose(images, image_transpose)


def test_displacement_from_center_constant_magnitude():
    key = random.PRNGKey(0)
    latent_values = random.normal(key, (8, 2))
    rectangle_model = rectangle.RectangleModel(None)
    structures = rectangle_model.construct_structures(latent_values)
    abs_displacements = jnp.abs(structures)
    diff_in_displacements = abs_displacements - jnp.expand_dims(abs_displacements[:, 0, :], 1)
    assert jnp.all(diff_in_displacements < 1e-6)
