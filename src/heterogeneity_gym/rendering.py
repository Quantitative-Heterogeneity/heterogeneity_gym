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


def _calculate_likelihood(
    reference_images,
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
def _render_clean_images_from_structures(
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
def _render_noisy_images_from_structures(
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


@eqx.filter_jit
def build_pipeline(
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

    # # TODO: revisit these parameters
    # solvent = cxs.GaussianIce(
    #     variance_function=op.Lorenzian(
    #         amplitude=noise_strength**2, length_scale=2.0 * pixel_size
    #     )
    # )

    theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble, potential_integrator, transfer_theory
    )
    pipeline = cxs.ContrastImagingPipeline(instrument_config, theory)
    distribution = dist.IndependentGaussianFourierModes(
        pipeline,
        variance_function=op.Lorenzian(
            amplitude=noise_strength**2, length_scale=2.0 * pixel_size
        ),
    )
    return distribution
