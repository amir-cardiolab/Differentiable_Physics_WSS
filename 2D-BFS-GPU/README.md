# 2-D BFS — adjoint inverse on the GPU

GPU version of [`../2D-BFS/Diff_Physics.py`](../2D-BFS/Diff_Physics.py): same problem,
discretisation, objective and Adam loop, with every PDE solve and every adjoint solve on the GPU.

## Libraries

* [DOLFINx 0.10](https://github.com/FEniCS/dolfinx) — finite-element forms and meshes
* [cuDOLFINx 0.10](https://github.com/bpachev/cuda-dolfinx) — GPU assembly for DOLFINx, with the
  patch in [`cudolfinx/`](cudolfinx/), which adds:
  * `cudolfinx.adjoint`: discrete adjoints on a [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint)
    tape (the engine of dolfin-adjoint), with gradient assembly on the GPU and a fused
    time-loop block (`linear_time_loop`) that keeps the whole time integration on the device
  * `cudolfinx.cudss`: [NVIDIA cuDSS](https://developer.nvidia.com/cudss) as a GPU direct solver
  * device-resident coefficients and a fix for Dirichlet conditions on mixed-space sub-spaces
* PETSc (CUDA build) for the linear-algebra objects

## Build

Everything runs in the official cuDOLFINx container; [`cudolfinx/build.sh`](cudolfinx/build.sh)
applies the patch to cuDOLFINx and builds it (about 15 min):

```bash
cd 2D-BFS-GPU
docker run --gpus all -it -v $PWD/..:/work -w /work/2D-BFS-GPU benpachev/cudolfinx:v0.10.0-cuda12.6
bash cudolfinx/build.sh            # -> ./cudolfinx-build/
```

With Apptainer: `apptainer exec --nv docker://benpachev/cudolfinx:v0.10.0-cuda12.6 bash cudolfinx/build.sh`.

## Run

Data: the data bundle's `Results_BFS/` (`mesh.xdmf/.h5`, `velocity_field.xdmf/.h5`, `C_4000.vtk`),
in `2D-BFS/` or `2D-BFS-GPU/`, or `--data DIR`.

```bash
source cudolfinx-build/env.sh
python3 Diff_Physics_GPU.py                       # 200 epochs, N = 4000
python3 Diff_Physics_GPU.py --epochs 1 --steps 300 --taylor   # quick check of the gradient
```

Results go to `Inverse_Results_GPU_*/`: `history.json` (per epoch), `loss_history.npy`,
`final_flow_rate_error.txt`, `optimized_fields.xdmf`.

## Implementation

| per epoch | CPU script | this version |
|---|---|---|
| Navier–Stokes Jacobian and residual | FEniCS, CPU | cuDOLFINx, GPU |
| Newton linear solves and the Navier–Stokes adjoint | PETSc LU, CPU | cuDSS LU, GPU |
| 4000 advection–diffusion steps | GMRES + ILU, matrix re-assembled every step | matrix once, cuDSS factorised once, states kept on the GPU |
| their adjoint | dolfin-adjoint replays 8000 tape blocks | one fused tape block, on the GPU |

## Time

N = 4000 (37.8k triangles, 172k Navier–Stokes and 19.3k advection–diffusion unknowns):

| per epoch | CPU script, 1 core | this version, RTX 2000 Ada |
|---|---:|---:|
| Navier–Stokes | 14.7 s | 0.3 s |
| advection–diffusion, 4000 steps | 154.2 s | 1.3 s |
| gradient | 528.4 s | 2.3 s |
| **epoch** | **697 s** | **3.8 s** |
| 200 epochs | ≈ 39 h | 13 min |
