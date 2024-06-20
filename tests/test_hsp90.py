from heterogeneity_gym.hsp90.pdbs import _load_hsp90_traj


def test_load_hsp90():
    traj = _load_hsp90_traj()
    assert len(traj) == 20
