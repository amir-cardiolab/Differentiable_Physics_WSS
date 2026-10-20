# 2-D BFS — adjoint inverse on the GPU

GPU version of [`../2D-BFS/Diff_Physics.py`](../2D-BFS/Diff_Physics.py): same problem,
discretisation and objective, with the steady Navier–Stokes solve, the 4000-step
advection–diffusion and both of their adjoints on the GPU. dolfin-adjoint still drives the
gradient (`ReducedFunctional(J, Control(g))`).

| File | What |
|------|------|
| [`Diff_Physics_GPU.py`](Diff_Physics_GPU.py) | the inverse solver (configuration at the top) |
| [`gpu_ns.py`](gpu_ns.py) | steady Navier–Stokes: GPU assembly, Newton and adjoint, as a pyadjoint `Block` |
| [`gpu_ade.py`](gpu_ade.py) | advection–diffusion time loop and its discrete adjoint, as a pyadjoint `Block` |
| [`gpu_fem.py`](gpu_fem.py) | P1/P2 triangle tables, quadrature and sparse-matrix maps for the GPU assembly |
| [`gpu_cudss.py`](gpu_cudss.py) | binding to NVIDIA cuDSS (GPU sparse direct solver) |
| [`bfs_data.py`](bfs_data.py) | finds the input data folder |
| [`verify_gradient.py`](verify_gradient.py), [`validate_gpu.py`](validate_gpu.py), [`check_data/`](check_data/) | checks against the CPU code |
| [`postprocess.py`](postprocess.py) | figures and metrics for finished runs |
| [`gpu_timed_run/`](gpu_timed_run/) | timed Adam run with pre-flight checks |
| [`Diff_Physics_CPU_reference.py`](Diff_Physics_CPU_reference.py) | unmodified copy of the CPU script (timing baseline) |

## Requirements

An NVIDIA GPU with a CUDA-12 driver (≥ 525), FEniCS 2019.1, dolfin-adjoint 2019.1, PyTorch
with CUDA and NVIDIA cuDSS:

```bash
conda create -n fenics_gpu -c conda-forge python=3.9 fenics=2019.1.0 "numpy<1.24" scipy matplotlib vtk
conda activate fenics_gpu
pip install --no-deps "git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0"
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu118
pip install nvidia-cudss-cu12
```

## Data

The same files as the CPU script (`mesh.xdmf/.h5`, `velocity_field.xdmf/.h5`, `C_4000.vtk` from
the data bundle's `Results_BFS/`). They are looked up in `$BFS_DATA`, `2D-BFS-GPU/Results_BFS/`
and `2D-BFS/Results_BFS/`, so one copy next to the CPU script serves both.

## Run

```bash
cd 2D-BFS-GPU
python Diff_Physics_GPU.py                              # L-BFGS-B, 300 iterations
OPTIMIZER=adam ITERS=200 python Diff_Physics_GPU.py     # the original Adam loop
python verify_gradient.py                               # gradient check against dolfin-adjoint
python postprocess.py Inverse_Results_GPU_*             # figures and metrics
```

Environment variables `OPTIMIZER` (lbfgs | adam), `ITERS`, `FINAL_IDX`, `DEVICE` and `OUT_DIR`
override the configuration block.

## Implementation

| per epoch | original | this version |
|-----------|----------|--------------|
| NS Jacobian and residual assembly | FEniCS, CPU, every Newton step | GPU (Stokes part assembled once) |
| NS Newton linear solves | PETSc LU, CPU | GPU, cuDSS LU (symbolic analysis once) |
| NS adjoint | pyadjoint, CPU LU | GPU, one cuDSS solve with Jᵀ |
| ADE matrices | FEniCS, re-assembled every time step | GPU, once per epoch |
| ADE 4000 forward steps | GMRES + ILU, CPU | GPU, one factorisation, two triangular solves per step |
| ADE 4000 adjoint steps | pyadjoint replays 4000 blocks, CPU | GPU, same factorisation with Aᵀ |

## Time

N = 4000 (37.8k triangles, 172k Navier–Stokes and 19.3k advection–diffusion unknowns):

| per epoch | original, 1 CPU core | GPU, RTX 2000 Ada |
|---|---:|---:|
| Navier–Stokes solve | 14.7 s | 0.6 s |
| advection–diffusion, 4000 steps | 154.2 s | 1.3 s |
| gradient | 528.4 s | 2.0 s |
| **epoch** | **697 s** | **3.8 s** |
| 200 epochs | ≈ 39 h | ≈ 13 min |

Plus about 15 s of one-time start-up (FEniCS compilation, GPU set-up).

## Limitations

Serial, 2-D triangles, Taylor–Hood P2/P1 and CG1 concentration, i.e. this problem only.
