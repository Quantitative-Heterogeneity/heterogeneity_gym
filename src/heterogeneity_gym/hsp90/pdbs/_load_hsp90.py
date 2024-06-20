import importlib.resources as pkg_resources
import os
import mdtraj


def _load_hsp90_traj():
    """Internal function to load the atomic scattering factor parameter
    table."""
    with pkg_resources.as_file(
        pkg_resources.files("heterogeneity_gym").joinpath("hsp90", "pdbs")
    ) as path:
        pdb_path = os.path.join(path, "hsp90.pdb")
        dcd_path = os.path.join(path, "hsp90.dcd")

    return mdtraj.load(dcd_path, top=pdb_path)
