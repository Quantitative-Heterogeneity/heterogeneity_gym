import torch
import pytest
from heterogeneity_gym import rectangle

if torch.cuda.is_available():
    devices = ["cuda", "cpu"]
else:
    devices = ["cpu",]


@pytest.mark.parametrize("device", devices)
def test_normalization(device):
    latent_values = torch.randn(8, 2, device=device)
    rectangle_model = rectangle.RectangleModel(None)
    images = rectangle_model.render_images_from_latent(latent_values)[0]

    dx = rectangle_model.grid_ticks[1] - rectangle_model.grid_ticks[0]

    image_integral = torch.sum(images, dim=(1, 2)) * dx**2
    avg_integral = torch.mean(image_integral)
    assert torch.all((image_integral - avg_integral) < 2e-3)


@pytest.mark.parametrize("device", devices)
def test_square_images_are_symmetric(device):
    latent_values = torch.randn(16, device=device)
    latent_values = torch.stack([latent_values, latent_values], dim=1)
    rectangle_model = rectangle.RectangleModel(None)
    images = rectangle_model.render_images_from_latent(latent_values)[0]
    print(images.shape)
    image_transpose = torch.transpose(images, 1, 2)
    assert torch.allclose(images, image_transpose)


@pytest.mark.parametrize("device", devices)
def test_displacement_from_center_constant_magnitude(device):
    latent_values = torch.randn(8, 2, device=device)
    rectangle_model = rectangle.RectangleModel(None)
    structures = rectangle_model.construct_structures(latent_values)
    abs_displacements = torch.abs(structures)
    diff_in_displacements = abs_displacements - abs_displacements[:, 0, :].unsqueeze(1)
    assert torch.all(diff_in_displacements < 1e-6)
