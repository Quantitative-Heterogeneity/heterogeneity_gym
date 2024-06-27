import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from cryojax.io import get_atom_info_from_mdtraj
from heterogeneity_gym.hsp90.pdbs import _load_hsp90_traj
import heterogeneity_gym.rendering as rendering


# TODO: Reparameterize to use SNR instead of noise_std
# TODO: Add an assert that checks if atoms have escaped the imaging region.
# TODO: Type hints and documentation
# TODO: Initialize reflection tensor in __init__


class DiscreteClassModel:
    def __init__(
        self,
        atom_positions,
        identities,
        latent_density=None,
        image_width_in_pixels: int = 128,
        pixel_size: float = 1.1,
        defocus_range=(5000.0, 10000.0),
        astigmatism_range=(0, 0),
        voltage_in_kilovolts=300.0,
        noise_strength=0.0,
    ):
        """
        TODO: we should construct a "default" latent density.
        TODO: Add initialization to device.  How does OpenAI solve device when creating their "environments"?
        """
        self.grid_shape = (
            image_width_in_pixels,
            image_width_in_pixels,
            image_width_in_pixels,
        )
        self.img_width = image_width_in_pixels
        self.img_shape = (image_width_in_pixels, image_width_in_pixels)
        self.noise_strength = noise_strength
        self.pixel_size = pixel_size
        self.voltage = voltage_in_kilovolts
        self.defocus_range = defocus_range
        self.astigmatism_range = astigmatism_range

        self.structures = atom_positions
        self.identities = identities
        self.volumes = rendering._build_volumes(
            atom_positions,
            identities,
            None,
            pixel_size,
            self.grid_shape,
        )

        self.latent_density = latent_density

    def evaluate_latent_density(self, x):
        """
        # Evaluates the probability density function of the two Gaussian mixture model.

        Parameters
        -----------
        x:
            Location(s) to evaluate the probability density function.
        """
        return self.latent_density.evaluate_density(x)

    def sample_images(self, num_samples: int, shuffle=True):
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
        images:
            Images
        structures:
            Corresponding images,
        latent_samples:
            Corresponding values of the latent.
        """
        latent_samples = self.latent_density.sample(num_samples, shuffle=shuffle)
        images, structures, poses, ctf_params = self.render_images_from_latent(
            latent_samples
        )
        return images, structures, poses, ctf_params, latent_samples

    def render_images_from_latent(
        self, latent_samples, poses=None, noise_std: Optional[float] = None
    ):
        if noise_std is None:
            noise_std = self.noise_std

        if poses is None:
            rotations = Rotation.random(len(latent_samples)).as_euler(seq="ZYZ")
            # rotations *= 180.0 / np.pi
            poses = np.zeros((len(latent_samples), 6))
            poses[:, 3:] += rotations
            poses = jnp.array(poses)

        volumes = self.volumes[latent_samples]
        structures = self.structures[latent_samples]
        images, ctfs = self.render_images_from_volumes(
            volumes, poses, noise_std=noise_std
        )
        # images, ctfs render_random_image(rotations, self.cjx_ensemble)
        return images, structures, poses, ctfs

    # def construct_structures(self, )

    def render_images_from_structures(
        self, atomic_structures, rotations, noise_std: Optional[float] = None
    ):
        """ """
        if noise_std is None:
            noise_std = self.noise_std

        N = len(atomic_structures)
        # key = np.random.randint(int(1e8), size=N)

        key = jax.random.PRNGKey(0)
        # TODO: replace with calls to jax random
        defocus = jnp.array(
            np.random.uniform(
                low=self.defocus_range[0], high=self.defocus_range[1], size=N
            )
        )
        astigmatism = jnp.array(
            np.random.uniform(
                low=self.astigmatism_range[0], high=self.astigmatism_range[1], size=N
            )
        )

        images = rendering._render_noisy_images_from_atoms(
            atomic_structures,
            self.identities,
            rotations,
            noise_std,
            defocus,
            astigmatism,
            self.img_shape,
            self.pixel_size,
            self.voltage,
            key,
        )
        return images, (defocus, astigmatism)
        # expand_structures = structures[..., None, None]  # N x Atom x 2 x 1 x 1
        # sq_displacements = (
        #     expand_structures - self.grid.to(structures)
        # ) ** 2  # N x Atom x 2 x Npix x Npix
        # sq_distances = torch.sum(sq_displacements, dim=-3)  # ... x Atom x Npix x Npix
        # kernel = torch.exp(-sq_distances / (2 * self.atom_variance))
        # image = torch.sum(kernel, dim=-3)  # ... x Npix x Npix
        # image = image + torch.randn_like(image) * noise_std
        return image, ctfs

    def render_images_from_volumes(self, volumes, rotations, noise_std=None):
        if noise_std is None:
            noise_std = self.noise_std

        N = len(volumes)
        # key = np.random.randint(int(1e8), size=N)

        key = jax.random.PRNGKey(0)
        # TODO: replace with calls to jax random
        defocus = jnp.array(
            np.random.uniform(
                low=self.defocus_range[0], high=self.defocus_range[1], size=N
            )
        )
        astigmatism = jnp.array(
            np.random.uniform(
                low=self.astigmatism_range[0], high=self.astigmatism_range[1], size=N
            )
        )

        images = rendering._render_noisy_images_from_potential_grid(
            volumes,
            rotations,
            noise_std,
            defocus,
            astigmatism,
            self.img_shape,
            self.pixel_size,
            self.voltage,
            key,
        )
        return images, (defocus, astigmatism)

    def evaluate_log_pij_matrix(
        self,
        experimental_images,
        simulated_images,
        noise_std: float,
    ):
        """ """
        raise NotImplementedError
        return -1 * difference / (2 * noise_std**2)


class HSP90_Model(DiscreteClassModel):
    def __init__(
        self,
        latent_density=None,
        image_width_in_pixels: int = 128,
        pixel_size: float = 1.1,
        defocus_range=(5000.0, 10000.0),
        astigmatism_range=(0, 0),
        voltage_in_kilovolts=300.0,
        noise_strength=0.0,
    ):
        traj = _load_hsp90_traj()
        atom_positions, identities = get_atom_info_from_mdtraj(traj)

        super().__init__(
            atom_positions,
            identities,
            latent_density=latent_density,
            image_width_in_pixels=image_width_in_pixels,
            pixel_size=pixel_size,
            defocus_range=defocus_range,
            astigmatism_range=astigmatism_range,
            voltage_in_kilovolts=voltage_in_kilovolts,
            noise_strength=noise_strength,
        )
