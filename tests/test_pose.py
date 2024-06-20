import numpy as np
from heterogeneity_gym import pose
from scipy.spatial.transform import Rotation

def _build_poses(n):
    rotations = Rotation.random(n).as_matrix()
    translations = np.random.randn(n, 3)
    poses = np.concatenate([rotations, translations], axis=-1)


class TestApplyPoses():
    def test_runs_in_jax():
        