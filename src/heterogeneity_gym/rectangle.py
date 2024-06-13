"""
Simple toy example of heterogeneity that attempts to recover images of a 
rectangle of varying width and height.  Reimplementation of an idea by 
David Silva Sanchez.
"""

import torch
from typing import Tuple, Optional

# TODO: Reparameterize to use SNR instead of noise_std
# TODO: Add an assert that checks if atoms have escaped the imaging region.
# TODO: Type hints and documentation
# TODO: Initialize reflection tensor in __init__


class RectangleModel:
    def __init__(self, latent_density, image_width_in_pixels=128, noise_std=0.0):
        """
        TODO: we should construct a "default" latent density.
        TODO: Add initialization to device.  How does OpenAI solve device when creating their "environments"?
        """
        self.latent_density = latent_density

        # Atom parameters...
        self.atom_variance = 0.04  # standard deviation of 0.2

        # Build the imaging grid
        self.grid_ticks = torch.linspace(-2, 2, image_width_in_pixels + 1)[:-1]
        self.grid = torch.stack(
            torch.meshgrid(self.grid_ticks, self.grid_ticks, indexing="xy"), dim=0
        )
        self.noise_std = noise_std

    def evaluate_latent_density(self, x: torch.tensor) -> torch.tensor:
        """
        Evaluates the probability density function of the two Gaussian mixture model.

        Parameters
        -----------
        x: torch.Tensor
            Location(s) to evaluate the probability density function.
        """
        return self.latent_density.evaluate_density(x)

    def sample_images(
        self, num_samples: int, shuffle=True
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
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
        images: torch.tensor
            Images
        structures: torch.tensor
            Corresponding images,
        latent_samples: torch.tensor
            Corresponding values of the latent.
        """
        latent_samples = self.latent_density.sample(num_samples, shuffle=shuffle)
        images, structures = self.render_images_from_latent(latent_samples)
        return images, structures, latent_samples

    def construct_structures(
        self, latent_samples: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Converts a latent distribution on the angles into three-dimensional atomic structures
        """
        N = len(latent_samples)
        dtype = latent_samples.dtype
        device = latent_samples.device

        # Put atoms in space.
        atom_1 = latent_samples / 2.0
        atom_2 = atom_1 * torch.tensor([1, -1], device=device, dtype=dtype)
        atom_3 = atom_1 * torch.tensor([-1, 1], device=device, dtype=dtype)
        atom_4 = atom_1 * torch.tensor([-1, -1], device=device, dtype=dtype)

        structures = torch.stack([atom_1, atom_2, atom_3, atom_4], dim=-2)

        # Add Dummy z dimension.

        return structures

    def render_images_from_latent(
        self, latent_samples: torch.tensor, noise_std: Optional[float] = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        if noise_std is None:
            noise_std = self.noise_std

        structures = self.construct_structures(latent_samples)
        images = self.render_images_from_structures(structures, noise_std=noise_std)
        return images, structures

    def render_images_from_structures(
        self, structures: torch.tensor, noise_std: Optional[float] = None
    ) -> torch.tensor:
        """ """
        if noise_std is None:
            noise_std = self.noise_std

        expand_structures = structures[..., None, None]  # N x Atom x 2 x 1 x 1
        sq_displacements = (
            expand_structures - self.grid.to(structures)
        ) ** 2  # N x Atom x 2 x Npix x Npix
        sq_distances = torch.sum(sq_displacements, dim=-3)  # ... x Atom x Npix x Npix
        kernel = torch.exp(-sq_distances / (2 * self.atom_variance))
        image = torch.sum(kernel, dim=-3)  # ... x Npix x Npix
        image = image + torch.randn_like(image) * noise_std
        return image

    def evaluate_log_pij_matrix(
        self,
        experimental_images: torch.tensor,
        simulated_images: torch.tensor,
        noise_std: float,
    ) -> torch.tensor:
        """ """
        experimental_images = experimental_images.unsqueeze(-4)
        simulated_images = simulated_images.unsqueeze(-3)
        difference = torch.sum(
            (experimental_images - simulated_images) ** 2, dim=(-1, -2)
        )
        return -1 * difference / (2 * noise_std**2)
