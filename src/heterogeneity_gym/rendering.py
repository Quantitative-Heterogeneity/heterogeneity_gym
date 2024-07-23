import torch
import cryojax
import jax
from dataclasses import dataclass
from typing import Tuple
import cryojax.simulator as cxs
import equinox as eqx
from typing import Optional
from jaxtyping import Array, Complex, Float
from cryojax.image import operators as op
from cryojax.inference import distributions as dist
from cryojax.coordinates import make_frequency_grid
from cryojax.image import irfftn

# from heterogeneity_gym.pose import apply_poses


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
    euler_angle_pose,
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
        euler_angle_pose,
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
        eqx.if_array(0),
    )
)
def _render_noisy_images_from_atoms(
    atom_positions,
    atom_identities,
    euler_angle_pose,
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
        euler_angle_pose,
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
    euler_angle_pose,
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
        euler_angle_pose,
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
    euler_angle_pose,
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
        euler_angle_pose,
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
        eqx.if_array(0),
    )
)
def _render_noisy_images_from_potential_grid(
    real_space_potential_grid,
    euler_angle_pose,
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
        euler_angle_pose,
        noise_strength,
        defocus,
        astigmatism,
        shape,
        pixel_size,
        voltage,
    )
    return distribution.sample(key)


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
    euler_angle_pose,
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
        euler_angle_pose,
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
        None,
        None,
        None,
    )
)
def _render_projections_from_potential_grid(
    real_space_potential_grid,
    euler_angle_pose,
    shape,
    pixel_size,
    voltage,
):
    """
    Renders a centered image
    """
    pose = cxs.EulerAnglePose(*euler_angle_pose)
    potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_space_potential_grid, pixel_size, pad_scale=2
    )
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    instrument_config = cxs.InstrumentConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage,
    )

    # ... compute the integrated potential
    fourier_integrated_potential = (
        potential_integrator.compute_fourier_integrated_potential(
            potential.rotate_to_pose(pose), instrument_config
        )
    )
    return irfftn(fourier_integrated_potential, s=instrument_config.shape)


def _build_distribution_from_potential(
    potential,
    potential_integrator,
    euler_angle_pose,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    pose = cxs.EulerAnglePose(*euler_angle_pose)
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


def evaluate_ctf(defocus_in_angstroms, astigmatism_in_angstroms, shape, pixel_size):
    frequency_grid = make_frequency_grid(shape, pixel_size)

    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.ContrastTransferFunction(
            frequency_grid,
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
        )
    )
