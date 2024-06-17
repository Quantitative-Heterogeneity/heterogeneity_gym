import torch
from dataclasses import dataclass

# Parameters
alpha = 0.3
sigma_1 = 0.5
mu_1 = -1.0
sigma_2 = 0.5
mu_2 = 1.0


def _gaussian(x, sigma, mu):
    return (
        1.0
        / (sigma * (2 * torch.pi) ** 0.5)
        * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


@dataclass
class TwoGaussianModel:
    alpha: float = alpha
    sigma_1: float = sigma_1
    mu_1: float = mu_1
    sigma_2: float = sigma_2
    mu_2: float = mu_2

    def evaluate_latent_density(self, x):
        """
        Evaluates the probability density function of the two Gaussian mixture model.

        Parameters
        ----------
        x: torch.Tensor
            Location(s) to evaluate the probability density function.
        """
        return two_gaussian_boltzmann_density(
            x, self.sigma_1, self.mu_1, self.sigma_2, self.mu_2, self.alpha
        )

    def evalute_density(self, x):
        return self.evaluate_latent_density(x)

    def evaluate_potential(self, x):
        """
        Evaluates the potential energy of the two Gaussian mixture model.

        Parameters
        ----------
        x: torch.Tensor
            Location(s) to evaluate the potential.
        """

        return -torch.log(self.evaluate_boltzmann(x))

    def sample(self, N, shuffle=True):
        """
        Makes "images" from the two Gaussian mixture model.

        Parameters

        N: int
            Number of samples
        shuffle: bool
            Whether to shuffle the images

        Returns
        -------
        images: torch.Tensor
            Images
        """
        return generate_samples(
            N,
            self.sigma_1,
            self.mu_1,
            self.sigma_2,
            self.mu_2,
            self.alpha,
            shuffle=shuffle,
        )


def two_gaussian_boltzmann_density(
    x, sigma_1=sigma_1, mu_1=mu_1, sigma_2=sigma_2, mu_2=mu_2, alpha=alpha
):
    """
    Functional form of our two Gaussian mixture model.

    Parameters
    ----------
    x: torch.Tensor
        Location(s) to evaluate the probability density function.
    sigma_1: float
        Standard deviation of the first Gaussian
    mu_1: float
        Mean of the first Gaussian
    sigma_2: float
        Standard deviation of the second Gaussian
    mu_2: float
        Mean of the second Gaussian
    alpha: float
        Fraction of samples from the first Gaussian

    Returns
    -------
    pi: torch.Tensor
        Probability density function of the two Gaussian mixture model.
    """
    pi = alpha * _gaussian(x, sigma_1, mu_1) + (1 - alpha) * _gaussian(x, sigma_2, mu_2)
    return pi


def generate_samples(
    N, sigma_1=sigma_1, mu_1=mu_1, sigma_2=sigma_2, mu_2=mu_2, alpha=alpha, shuffle=True
):
    """
    Sample the two Gaussian mixture model.

    Parameters
    ----------
    N: int
        Number of samples
    alpha: float
        Fraction of samples from the first Gaussian
    sigma_1: float
        Standard deviation of the first Gaussian
    mu_1: float
        Mean of the first Gaussian
    sigma_2: float
        Standard deviation of the second Gaussian
    mu_2: float
        Mean of the second Gaussian
    shuffle: bool
        Whether to shuffle the images

    Returns
    -------
    images_w_noise: torch.Tensor
        Images with lenoise
    raw_images: torch.Tensor
        Images without noise
    structures: torch.Tensor
        The structures of each ``cluster'' in the mixture model
    log_Pij: torch.Tensor
        The log-likelihood of generating image i from cluster j.

    """
    y_no_noise_1 = torch.randn(int(np.round(N * alpha))) * sigma_1 + mu_1
    y_no_noise_2 = torch.randn(int(np.round(N * (1 - alpha)))) * sigma_2 + mu_2
    y = torch.concatenate([y_no_noise_1, y_no_noise_2])

    if shuffle:
        y = y[torch.randperm(len(y))]
    return y


def generate_dataset(
    num_images: int,
    num_clusters: int = 21,
    noise_std: float = 0.1,
    shuffle: bool = True,
):
    """
    Generate a dataset of images from the two Gaussian mixture model.

    Largely superseded by the TwoGaussianModel class, but I'm keeping for
    people who don't like classes.
    """
    raw_images = generate_images(num_images, shuffle=shuffle)
    structures = torch.linspace(-3, 3, num_clusters)
    images_w_noise = raw_images + torch.random.randn(num_images) * noise_std

    dist_matrix = (raw_images - structures.unsqueeze(1)) ** 2
    log_Pij = dist_matrix / (-2 * noise_std**2)

    return (
        images_w_noise,
        raw_images,
        structures,
        log_Pij,
    )
