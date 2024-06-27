import equinox as eqx
import cryojax.simulator as cxs
from heterogeneity_gym import rendering
from cryojax.image._fft import rfftn


@eqx.filter_vmap(  # Over provided potential values
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
def _calculate_log_likelihood_of_atom_structures(  # TODO: we could probably speed this up by building distributions once.
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
    distribution = rendering._build_distribution_from_atoms(
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

    fourier_space_image = rfftn(reference_image)
    return distribution.log_likelihood(fourier_space_image)


@eqx.filter_vmap(  # Over provided potential values
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
    )
)
@eqx.filter_vmap(  # Over reference images
    in_axes=(
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
def _calculate_log_likelihood_of_potential_grid(  # TODO: we could probably speed this up by building distributions once.
    potential_grid,
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
    potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        potential_grid, pixel_size, pad_scale=2
    )
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    distribution = rendering._build_distribution_from_potential(
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
    fourier_space_image = rfftn(reference_image)
    return distribution.log_likelihood(fourier_space_image)
