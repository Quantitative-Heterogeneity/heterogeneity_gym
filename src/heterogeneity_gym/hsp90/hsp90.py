import torch
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from cryojax.io import get_atom_info_from_mdtraj
import cryojax.simulator as cxs
import equinox as eqx
from heterogeneity_gym.hsp90.pdbs import _load_hsp90_traj

# TODO: Reparameterize to use SNR instead of noise_std
# TODO: Add an assert that checks if atoms have escaped the imaging region.
# TODO: Type hints and documentation
# TODO: Initialize reflection tensor in __init__


class HSP90_1DLatent:
    def __init__(self, latent_density=None, image_width_in_pixels=128, noise_strength=0.0):
        """
        TODO: we should construct a "default" latent density.
        TODO: Add initialization to device.  How does OpenAI solve device when creating their "environments"?
        """
        self.img_width = image_width_in_pixels
        self.noise_strength = noise_strength
        if latent_density is None:
            # latent_density =
            raise NotImplementedError
        self.latent_density = latent_density
        self.pipeline = self._build_pipeline()
        ensemble = _HSP90_Ensemble()

    def _build_pipeline()

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
        images, structures, rotations, ctf_params = self.render_images_from_latent(
            latent_samples
        )
        return images, structures, rotations, ctf_params, latent_samples

    def render_images_from_latent(
        self, latent_samples: torch.tensor, noise_std: Optional[float] = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        if noise_std is None:
            noise_std = self.noise_std

        rotations = Rotation.random(len(latent_samples)).as_quat()
        rotations = torch.tensor(
            rotations, dtype=structures.dtype, device=structures.device
        )
        # structures = torch.einsum("nlj, nkj-> nlk", structures, rotations)
        render_random_image(rotations, self.cjx_ensemble)
        images, ctfs = self.render_images_from_structures(
            structures, noise_std=noise_std
        )
        return images, structures, rotations, ctfs

    def render_images_from_structures(
        self, structures: torch.tensor, noise_std: Optional[float] = None
    ) -> torch.tensor:
        """ """
        if noise_std is None:
            noise_std = self.noise_std

        raise NotImplementedError
        # expand_structures = structures[..., None, None]  # N x Atom x 2 x 1 x 1
        # sq_displacements = (
        #     expand_structures - self.grid.to(structures)
        # ) ** 2  # N x Atom x 2 x Npix x Npix
        # sq_distances = torch.sum(sq_displacements, dim=-3)  # ... x Atom x Npix x Npix
        # kernel = torch.exp(-sq_distances / (2 * self.atom_variance))
        # image = torch.sum(kernel, dim=-3)  # ... x Npix x Npix
        # image = image + torch.randn_like(image) * noise_std
        return image, ctfs

    def evaluate_log_pij_matrix(
        self,
        experimental_images: torch.tensor,
        simulated_images: torch.tensor,
        noise_std: float,
    ) -> torch.tensor:
        """ """
        raise NotImplementedError
        return -1 * difference / (2 * noise_std**2)


class _HSP90_Ensemble(cxs.DiscreteStructuralEnsemble):

    def __init__(self, shape=(256, 256, 256), voxel_size=1.0, b_factors=None):
        self.b_factors = b_factors
        self.shape = shape
        self.voxel_size = voxel_size
        volumes = self._build_volumes()
        super().__init__(volumes)

    def _build_ensemble(self):
        traj = _load_hsp90_traj()
        atom_positions, identities = get_atom_info_from_mdtraj(traj)

        # We could probably vmap over this, but for now the loop is fine.
        volumes = []
        for conformation in atom_positions:
            peng_potential = cxs.PengAtomicPotential(
                atom_positions, identities, b_factors=self.b_factors
            )

            grid_potential = peng_potential.as_real_voxel_grid(
                shape=self.shape,
                voxel_size=self.voxel_size,
            )
            volumes.append(grid_potential)

        return tuple(volumes)


@eqx.filter_vmap(  # Over structures
    in_axes=(
        0,
        0,
        None,
    )
)
def render_random_ensemble(rotation, latent_code, ensemble):
    new_pose = cxs.QuaternionPose(rotation)
    pose_return = lambda x: x.pose
    conf_return = lambda x: x.conformation
    ensemble = eqx.tree_at(pose_return, ensemble, new_pose)
    ensemble = eqx.tree_at(conf_return, ensemble, latent_code)

