"""Where the 2-D BFS input data is (mesh.xdmf, velocity_field.xdmf, C_4000.vtk).

Looked up in this order: the environment variable BFS_DATA; Results_BFS/ next to the script
being run; ../2D-BFS/Results_BFS/ (the copy the CPU scripts use, see the top-level README);
the older folder name Results_BFS_InletBC_small/ in either place.
"""
import os
import sys

FILES = ("mesh.xdmf", "velocity_field.xdmf", "C_4000.vtk")
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("Results_BFS", "Results_BFS_InletBC_small")


def find_data_dir(explicit=None):
    cands = [explicit] if explicit else []
    if os.environ.get("BFS_DATA"):
        cands.append(os.environ["BFS_DATA"])
    for base in (os.getcwd(), HERE, os.path.join(HERE, "..", "2D-BFS")):
        cands += [os.path.join(base, n) for n in NAMES]
    for c in cands:
        if c and all(os.path.exists(os.path.join(c, f)) for f in FILES):
            return os.path.abspath(c)
    sys.exit("BFS data not found: put Results_BFS/ (mesh.xdmf, velocity_field.xdmf, C_4000.vtk, "
             "from the data bundle) in 2D-BFS-GPU/ or 2D-BFS/, or set BFS_DATA=/path/to/Results_BFS")
