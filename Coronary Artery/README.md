# Coronary Artery (3-D)

Inverse reconstruction of velocity, pressure, concentration transport and
**WSS** in a patient-specific coronary artery, from concentration observations.

## Scripts

| Script | Method | Reads | Writes |
|--------|--------|-------|--------|
| [`PINNs.py`](PINNs.py) | Physics-Informed NN (PyTorch) | `CFD_Data/` (`Data_*.vtk`, `Wall.vtk`, `Full_Domain.vtk`) | `WEIGHTS_pre/`, `Loss_pre/`, `Results_pre/` |
| [`Diff_Physics.py`](Diff_Physics.py) | Adjoint inverse (FEniCS + dolfin-adjoint) | `Coronary_Mesh/` (mesh) + `Forward_CFD/` (observations) + `Coronary_Mask.vtk` | `Inverse_TwoOutlet_1K_HardConstraint/` |

## Data folders

- **`CFD_Data/`** — unsteady CFD snapshots used as PINN inputs *(renamed from
  `Unsteady/`)*. Contains the per-timestep `Data_*.vtk` (velocity/concentration)
  for the training window, plus `Wall.vtk` and `Full_Domain.vtk`.
  - In `PINNs.py` the input window is set by `file_start` / `file_end`
    (currently `175`–`179`; the data bundle is pruned to that window).
- **`Coronary_Mesh/`** — volume mesh and boundary tags for the inverse solver
  (`Volume_mesh.xml`, `BCnodeFacets.xml`).
- **`Forward_CFD/`** — forward FEniCS results used as observations by the
  inverse solver (`concentration.xdmf/.h5`, `velocity_final.xdmf/.h5`).
- **`Coronary_Mask.vtk`** — spatial mask defining the region where the inverse
  objective compares predicted vs observed concentration
  (`MASK_VTK` in `Diff_Physics.py`).

## Inverse solver inlet/outlet setup

`Diff_Physics.py` is the **two-outlet** inverse (outlet tags `3` and `4`, both
`p = 0`). It reconstructs the inlet velocity profile so the simulated
concentration matches the observation inside the spatial mask `Coronary_Mask.vtk`
(set via `MASK_VTK` near the top of the script).

## Run

```bash
python PINNs.py           # GPU recommended
python Diff_Physics.py    # requires FEniCS + dolfin-adjoint
```
