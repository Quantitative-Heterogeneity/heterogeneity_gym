"""
Simple toy example of heterogeneity that attempts to recover images of a 
rectangle of varying width and height.  Initial idea for the example
David Silva Sanchez.  Code by Erik Thiede and Joshua Rhodes
"""

import torch
from torch.distributions import MultivariateNormal
from typing import Tuple, Optional
import warnings

def rotate(point, theta):
    """
    Rotate a collection of 2D points given the angle
    """
    mat = torch.stack(
        [torch.stack([torch.cos(theta), -torch.sin(theta)]),
         torch.stack([torch.sin(theta), torch.cos(theta)])]
    ).permute((2, 0, 1))
    return torch.einsum('...ij,...j->...i', mat, point)

class Latent2DGaussianMixture:
    """
    Latent distribution of lengths and widths of four-atom molecule model RectangleModel.
    """

    def __init__(
        self,
        length_width_means: torch.tensor = torch.tensor([[0., 0.]]),
        length_width_covariances: torch.tensor = torch.tensor([[[1., 0.], [0., 1.]]]),
        length_width_weights: torch.tensor = None,
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
        self.num_gaussians = length_width_means.size(dim=0)
        self.length_width_means = length_width_means
        self.length_width_covariances = length_width_covariances

        # If no weights are given, default to uniform weighting.
        if length_width_weights == None:
            length_width_weights = torch.ones(self.num_gaussians)

        self.length_width_weights = length_width_weights / torch.sum(
            length_width_weights
        )  # Ensure weights are normalized.

    def evaluate_density(self, length_width: torch.tensor) -> float:
        """
        Returns value of density at length_width.

        Parameters
        ----------
        length_width : tensor
            (1, 2) tensor of length, width positions in length-width space

        Returns
        -------
        density : float
            pdf evaluated at length_width
        """
        gaussians = MultivariateNormal(
            loc=self.length_width_means, covariance_matrix=self.length_width_covariances
        )
        gaussian_densities = torch.exp(gaussians.log_prob(length_width))
        weighted_gaussians = gaussian_densities * self.length_width_weights
        return torch.sum(weighted_gaussians, dim=0).item()

    def calculate_num_samples(
        self, length_width_weights: torch.tensor, N: int
    ) -> torch.tensor:
        """
        Given a set of mixture weights, calculates how many samples to draw
        from each component for a total of num_samples samples.

        Parameters
        ----------
        length_width_weights : torch.tensor
            Weight of each mixture component
        N : int
            Total number of samples to draw

        Returns
        -------
        number_samples : torch.tensor[int64]
            Number of samples to draw from each component,
            total should sum to N.
        """
        cumulative_num_samples = torch.round(
            torch.cumsum(length_width_weights * N, dim=0)
        )
        number_samples = torch.diff(cumulative_num_samples, prepend=torch.tensor([0]))
        return number_samples.int()

    def sample(self, num_samples: int, shuffle: bool = True) -> torch.tensor:
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
        sample = torch.empty(
            0,
        )

        for gaussian_index, sample_size in enumerate(sample_frequencies):
            gaussian_model = MultivariateNormal(
                loc=self.length_width_means[gaussian_index],
                covariance_matrix=self.length_width_covariances[gaussian_index],
            )
            gaussian_sample = gaussian_model.rsample(torch.Size([sample_size]))
            sample = torch.cat((sample, gaussian_sample), dim=0)

        if shuffle:
            sample = sample[torch.randperm(len(sample))]

        return sample.reshape(num_samples, 2)


# TODO: Reparameterize to use SNR instead of noise_std
# TODO: Type hints and documentation


class RectangleModel:
    def __init__(
        self,
        latent_density=None,
        image_width_in_pixels=128,
        image_size=4,
        noise_std=0.0,
        device="cpu",
    ):
        """
        TODO: we should construct a "default" latent density.
        TODO: Add initialization to device.  How does OpenAI solve device when creating their "environments"?
        """
        if latent_density is None:
            latent_means = torch.tensor([[0.5, 1.5], [1.0, 1.0]])
            latent_covariances = torch.tensor(
                [
                    [[0.1, 0], [0, 0.1]],
                    [[0.3, 0.1], [0.1, 0.2]],
                ]
            )
            latent_weights = torch.tensor([1, 4])  # Need not be normalized
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
        self.grid_ticks = torch.linspace(
            -image_size, image_size, image_width_in_pixels + 1, device=device
        )[:-1]
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
        if len(latent_samples.shape) < 2:
            latent_samples = torch.unsqueeze(latent_samples, 0)
        N = len(latent_samples)
        dtype = latent_samples.dtype
        device = latent_samples.device

        # Put atoms in space.
        atom_1 = latent_samples / 2.0
        atom_2 = atom_1 * torch.tensor([1, -1], device=device, dtype=dtype)
        atom_3 = atom_1 * torch.tensor([-1, 1], device=device, dtype=dtype)
        atom_4 = atom_1 * torch.tensor([-1, -1], device=device, dtype=dtype)
        theta = torch.rand(N) * torch.pi / 2.0 - torch.pi / 4.0
        atom_1 = rotate(atom_1, theta)
        atom_2 = rotate(atom_2, theta)
        atom_3 = rotate(atom_3, theta)
        atom_4 = rotate(atom_4, theta)

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

#        if torch.any(torch.abs(structures) > self.image_size):
#            warnings.warn("One of the structures may have escaped the imaging window")

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
