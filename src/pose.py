import jax.numpy as jnp


def apply_poses(atom_positions, poses):
    rotations = poses[..., :3, :3]
    translations = poses[..., :3, 3]
    atom_positions = atom_positions @ rotations + translations
    return atom_positions
