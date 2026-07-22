# 2-D Backward-Facing Step (BFS)

Inverse reconstruction of velocity, pressure, concentration and **WSS** on a
2-D backward-facing-step geometry, from concentration data observed only inside
a chosen **y-window**.

## Scripts

| Script | Method | Reads | Writes |
|--------|--------|-------|--------|
| [`PINN.py`](PINN.py) | Physics-Informed NN (PyTorch) | `Results_BFS/` (`Data_*.vtk`, `Wall.vtk`, `WSS.csv`) | `Runs/<y-window>/<timestamp>/` |
| [`Diff_Physics.py`](Diff_Physics.py) | Adjoint inverse (FEniCS + dolfin-adjoint) | `Results_BFS/` (`mesh.xdmf`, `concentration.xdmf`, `velocity_field.xdmf`) | `Inverse_Results_*/` |

## The y-observation window (both scripts)

Both scripts only use concentration data inside one or two horizontal bands in
**physical** `y`. This is the single most important knob to keep consistent
between the two methods, so it lives in a clearly marked config block near the
top of each script's settings:

- `PINN.py` → `apply_two_region_y_mask(...)`, configured by
  `y1_min, y1_max, y2_min, y2_max` (in the `main()` settings block).
- `Diff_Physics.py` → the `mask` `Expression`, configured by the same four
  variables (in section *1) CONFIG*).

```
Region 1 : y1_min <= y <= y1_max          (always active)
Region 2 : y2_min <= y <= y2_max          (optional; set both to None to disable)
```

Defaults currently differ by design (PINN: `[0.0, 0.1]`; inverse: `[0.3, 0.4]`)
— set them to the same band when you want the two methods to see identical
observations.

## Data folder: `Results_BFS/`

The data bundle is **pruned to only the files actually used**. Contents:

| File(s) | Used by | Notes |
|---------|---------|-------|
| `Data_0,5,10,…,100.vtk` (21 files) | `PINN.py` — **near-wall** case | mask `y ∈ [0.0, 0.1]`; loaded as `Data_{n}.vtk` with `file_start=0, file_end=100, file_step=5` |
| `Early_100,105,…,200.vtk` (21 files) | `PINN.py` — **far-field** case | mask `y ∈ [0.3, 0.4]`; the `Early_*` snapshots covering the later window (100→200, step 5) |
| `mesh.{xdmf,h5}`, `velocity_field.{xdmf,h5}` | `Diff_Physics.py` | FEniCS mesh + ground-truth velocity for the adjoint inverse |
| `C_4000.vtk` | `Diff_Physics.py` | target concentration snapshot (forward step `final_idx=4000`), mapped onto the CG1 space by nearest-DOF match |
| `Wall.vtk`, `WSS.csv` | `PINN.py` | wall geometry + ground-truth WSS for the post-training WSS comparison |
| `read_mesh_info.py` | — | small helper/provenance script |

### Which snapshots go with which PINN case
- **Near-wall** (`y1_min, y1_max = 0.0, 0.1` in `PINN.py`): the `Data_*` series.
- **Far-field** (`y1_min, y1_max = 0.3, 0.4`): the `Early_*` series (100→200,
  step 5). The committed `PINN.py` loads `Data_{n}.vtk`; to reproduce the
  far-field run, point the loader at the `Early_*` files and set the y-window
  to `[0.3, 0.4]`.

## Run

```bash
python PINN.py            # GPU recommended
python Diff_Physics.py    # requires FEniCS + dolfin-adjoint
```
