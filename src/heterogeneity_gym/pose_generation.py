import numpy as np
from scipy.spatial.transform import Rotation
from jaxtyping import Array, Int


def generate_random_3d_poses(
    num_images: int, translation_std_dev: float, rng: np.random.Generator
):
    # Generate random latent data
    translations = rng.standard_normal((num_images, 3)) * translation_std_dev
    rotations_np = Rotation.random(num_images).as_euler("zyz", degrees=True)
    poses = np.concatenate([translations, rotations_np], axis=1)
    return poses


def generate_icosahedral_poses(
    nu: int,
    n_theta: int,
):
    """
    Generates a set of poses roughly evenly distributed on the sphere
    Each rotation axis is a vertex on an icosphere
    Translations are zero.

    Parameters
    ----------
    nu : int
        Number of subdivisions in the icosphere
    n_theta : int
        Number of angles to rotate around axis in axis-angle representation

    Returns
    -------
    poses : np.ndarray
        Array of shape (nu * n_theta, 6) with each row being a pose.
        First 3 columns are translations, last 3 columns are rotations in zyz Euler angles

    TODO: Double check that we are in zyz, not ZYZ.
    """
    # Generate random latent data
    rotations = generate_rotation_grid(nu, n_theta)
    translations = np.zeros((len(rotations), 3))

    poses = np.concatenate([translations, rotations], axis=1)
    return poses


def generate_rotation_grid(
    nu: int, n_theta: int, flatten: bool = True
) -> Int[Array, "N 3"]:
    """
    Generate a grid of rotations relatively evenly spaced on the sphere
    using a subdivision of the icosphere and a rotation around the z-axis.
    """
    # Import moved into function to make dependency optional
    from icosphere import icosphere

    vertices, __ = icosphere(nu)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0]) * 180.0 / np.pi
    angle_from_z_axis = np.arccos(vertices[:, 2]) * 180.0 / np.pi
    angle_rotations = np.linspace(0, 360.0, n_theta, endpoint=False)

    zyz_angle = np.zeros((len(angle_rotations), len(vertices), 3))
    zyz_angle[:, :, 0] += angle_rotations.reshape(-1, 1)
    zyz_angle[:, :, 1] += angle_from_z_axis.reshape(1, -1)
    zyz_angle[:, :, 2] += theta.reshape(1, -1)

    if flatten:
        zyz_angle = zyz_angle.reshape(-1, 3)

    return zyz_angle
