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
from heterogeneity_gym.pose import apply_poses


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
        0
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
    )
)
def _calculate_likelihood( # TODO: fix order
    atom_positions,
    atom_identities,
    b_factors,
    reference_image,
    poses,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    atom_positions = apply_poses(atom_positions, poses)
    pipeline = build_pipeline(
        atom_positions,
        atom_identities,
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
        None,
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        None,
        None,
    )
)

def _render_clean_images_from_point_atoms(
    atom_positions,
    atom_identities,
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
    distribution = build_pipeline(
        atom_positions,
        atom_identities,
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
    distribution = build_pipeline(
        atom_positions,
        atom_identities,
        b_factors,
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


@eqx.filter_jit
def _build_pipeline(
    atom_positions,
    atom_identities,
    b_factors,
    noise_strength,
    defocus,
    astigmatism,
    shape,
    pixel_size,
    voltage,
):
    pose = cxs.EulerAnglePose()
    potential = cxs.PengAtomicPotential(
        atom_positions, atom_identities, b_factors=b_factors
    )
    structural_ensemble = cxs.SingleStructureEnsemble(potential, pose)
    potential_integrator = cxs.GaussianMixtureProjection()
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
