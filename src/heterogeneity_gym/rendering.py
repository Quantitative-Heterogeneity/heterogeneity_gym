import torch
import cryojax
import jax
from dataclasses import dataclass
from typing import Tuple
import cryojax.simulator as cxs
import equinox as eqx
from typing import Optional
from cryojax.image import operators as op
from cryojax.inference import distributions as dist

# from heterogeneity_gym.pose import apply_poses


@eqx.filter_vmap(  # Over structures
    in_axes=(
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
)
@eqx.filter_vmap(  # Over reference images
    in_axes=(
        None,
        None,
        None,
        0,
        0,
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
    )
)
def _calculate_likelihood_(  # TODO: we could probably speed this up by building distributions once.
    atom_positions,
    atom_identities,
    b_factors,
    reference_image,
    rotation_as_euler_angle,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    # atom_positions = apply_poses(atom_positions, poses)
    pipeline = _build_distribution_from_atoms(
        atom_positions,
        atom_identities,
        rotation_as_euler_angle,
        b_factors,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )

    distribution = dist.IndependentGaussianFourierModes(
        pipeline,
        variance_function=op.Lorenzian(
            amplitude=noise_strength**2, length_scale=2.0 * pixel_size
        ),
    )
    return distribution.log_likelihood(reference_image)


@eqx.filter_vmap(
    in_axes=(
        0,
        None,
        0,
        None,
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
    )
)
def _render_clean_images_from_atoms(
    atom_positions,
    atom_identities,
    rotation_as_euler_angle,
    b_factors,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    """
    Renders a centered image
    """
    distribution = _build_distribution_from_atoms(
        atom_positions,
        atom_identities,
        rotation_as_euler_angle,
        b_factors,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )
    return distribution.compute_signal()


@eqx.filter_vmap(
    in_axes=(
        0,
        None,
        0,
        None,
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
        None,
    )
)
def _render_noisy_images_from_atoms(
    atom_positions,
    atom_identities,
    rotation_as_euler_angle,
    b_factors,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
    key,
):
    """
    Renders a centered image
    """
    distribution = _build_distribution_from_atoms(
        atom_positions,
        atom_identities,
        rotation_as_euler_angle,
        b_factors,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )
    return distribution.sample(key)


@eqx.filter_jit
def _build_distribution_from_atoms(
    atom_positions,
    atom_identities,
    rotation_as_euler_angle,
    b_factors,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    potential = cxs.PengAtomicPotential(
        atom_positions, atom_identities, b_factors=b_factors
    )
    potential_integrator = cxs.GaussianMixtureProjection()
    return _build_distribution_from_potential(
        potential,
        potential_integrator,
        rotation_as_euler_angle,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )


@eqx.filter_vmap(
    in_axes=(
        0,
        0,
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
    )
)
def _render_clean_images_from_potential_grid(
    real_space_potential_grid,
    rotation_as_euler_angle,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    """
    Renders a centered image
    """
    potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_space_potential_grid, pixel_size, pad_scale=2
    )
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    distribution = _build_distribution_from_potential(
        potential,
        potential_integrator,
        rotation_as_euler_angle,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )
    return distribution.compute_signal()


@eqx.filter_vmap(
    in_axes=(
        0,
        0,
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
        None,
    )
)
def _render_noisy_images_from_potential_grid(
    real_space_potential_grid,
    rotation_as_euler_angle,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
    key,
):
    """
    Renders a centered image
    """
    potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_space_potential_grid, pixel_size, pad_scale=2
    )
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    distribution = _build_distribution_from_potential(
        potential,
        potential_integrator,
        rotation_as_euler_angle,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )
    return distribution.sample(key)


# @eqx.filter_jit
# def _build_atom_


def _build_distribution_from_potential(
    potential,
    potential_integrator,
    rotation_as_euler_angle,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    pose = cxs.EulerAnglePose(*rotation_as_euler_angle)
    structural_ensemble = cxs.SingleStructureEnsemble(potential, pose)
    # potential_integrator = cxs.GaussianMixtureProjection()
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.ContrastTransferFunction(
            defocus_in_angstroms=defocus,
            astigmatism_in_angstroms=astigmatism,
        )
    )
    instrument_config = cxs.InstrumentConfig(shape, pixel_size, voltage, pad_scale=1.1)

    theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble, potential_integrator, transfer_theory
    )
    pipeline = cxs.ContrastImagingPipeline(instrument_config, theory)

    # TODO: revisit these parameters
    distribution = dist.IndependentGaussianFourierModes(
        pipeline,
        variance_function=op.Lorenzian(
            amplitude=noise_strength**2, length_scale=2.0 * pixel_size
        ),
    )
    return distribution


@eqx.filter_vmap(
    in_axes=(
        0,
        None,
        None,
        None,
        None,
    )
)
def _build_volumes(atom_positions, identities, b_factors, voxel_size, grid_shape):

    peng_potential = cxs.PengAtomicPotential(
        atom_positions, identities, b_factors=b_factors
    )

    grid_potential = peng_potential.as_real_voxel_grid(
        shape=grid_shape,
        voxel_size=voxel_size,
    )

    return grid_potential


# class DiscreteDistributionRenderer:
#     def __init__(self, distribution: dist.AbstractDistribution):
#         self.distribution = distribution

#     def render_random_projection(self, rotation, latent_code, defocus: float, astigmatism: float, key: float):


# @eqx.filter_vmap(  # Over structures
#     in_axes=(
#         None,
#         0,
#         0,
#         0,
#         0,
#         0,
#     )
# )
# def render_random_projection(
#     distribution, rotation, latent_code, defocus: float, astigmatism: float, key: float
# ):
#     new_pose = cxs.EulerAnglePose(rotation)
#     pose_return = (
#         lambda x: x.imaging_pipeline.scattering_theory.structural_ensemble.pose
#     )
#     conf_return = (
#         lambda x: x.imaging_pipeline.scattering_theory.structural_ensemble.conformation
#     )
#     defocus_return = (
#         lambda x: x.imaging_pipeline.scattering_theory.structural_ensemble.conformation
#     )
#     astigmatism_return = (
#         lambda x: x.imaging_pipeline.scattering_theory.structural_ensemble.conformation
#     )
#     distribution = eqx.tree_at(pose_return, distribution, new_pose)
#     distribution = eqx.tree_at(conf_return, distribution, latent_code)
#     distribution = eqx.tree_at(defocus_return, distribution, defocus)
#     distribution = eqx.tree_at(astigmatism_return, distribution, astigmatism)
#     return distribution.sample(key)
