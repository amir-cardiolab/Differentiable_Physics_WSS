# 2-D BFS — adjoint inverse on the GPU

GPU version of [`../2D-BFS/Diff_Physics.py`](../2D-BFS/Diff_Physics.py) (FEniCS +
dolfin-adjoint). Same problem, same discretisation, same objective — but both PDE
solves (steady Navier–Stokes and the 4000-step advection–diffusion) **and both of
their adjoints** run on the GPU. dolfin-adjoint still drives the gradient
(`ReducedFunctional(J, Control(g))`).

The default optimiser is **L-BFGS-B** (`pyadjoint.minimize`); the original Adam
loop is kept as `OPTIMIZER=adam`. See [Optimiser](#optimiser-adam-vs-l-bfgs-b) for why.

| File | What |
|------|------|
| [`Diff_Physics_GPU.py`](Diff_Physics_GPU.py) | the inverse solver — documented, config at the top |
| [`gpu_ns.py`](gpu_ns.py) | steady NS: GPU assembly + Newton + adjoint, as a pyadjoint `Block` |
| [`gpu_ade.py`](gpu_ade.py) | ADE time loop + exact discrete adjoint, as a pyadjoint `Block` |
| [`gpu_fem.py`](gpu_fem.py) | P1/P2 triangle tables, quadrature, CSR scatter maps (GPU assembly) |
| [`gpu_cudss.py`](gpu_cudss.py) | ctypes binding to NVIDIA cuDSS (GPU sparse direct solver) |
| [`verify_gradient.py`](verify_gradient.py) | GPU vs pure dolfin-adjoint gradient + Taylor test |
| [`postprocess.py`](postprocess.py) | figures + metrics for finished runs |
| [`gpu_timed_run/`](gpu_timed_run/) | self-contained timed GPU run (Adam) with pre-flight checks, ETA and a report |
| [`validate_gpu.py`](validate_gpu.py) | one-file CPU-vs-GPU check (same point, finite differences, Adam epochs) |
| [`check_data/`](check_data/) | data-consistency and CPU-vs-GPU checks at the final control |
| [`Diff_Physics_CPU_reference.py`](Diff_Physics_CPU_reference.py) | unmodified copy of [`../2D-BFS/Diff_Physics.py`](../2D-BFS/Diff_Physics.py) (timing baseline) |
| [`bfs_data.py`](bfs_data.py) | finds the input data folder (see [Data](#data)) |

## Data

The same input files as the CPU script (`mesh.xdmf/.h5`, `velocity_field.xdmf/.h5`,
`C_4000.vtk` from the data bundle's `Results_BFS/`). Every script looks for them in this order
([`bfs_data.py`](bfs_data.py)): the environment variable `BFS_DATA`, `Results_BFS/` in this
folder, `../2D-BFS/Results_BFS/` (where the CPU script reads them), so a single copy of the
bundle next to `2D-BFS/` serves both.

## Requirements

* NVIDIA GPU (≥ 2 GB free memory), driver supporting CUDA 12 (≥ 525).
* FEniCS **2019.1** + dolfin-adjoint **2019.1**, PyTorch with CUDA, and NVIDIA cuDSS
  (`pip install nvidia-cudss-cu12`), plus `numpy<1.24`, `scipy`, `matplotlib`, `vtk`.
  With conda:

  ```bash
  conda create -n fenics_gpu -c conda-forge python=3.9 fenics=2019.1.0 "numpy<1.24" scipy matplotlib vtk
  conda activate fenics_gpu
  pip install --no-deps "git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0"
  pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu118
  pip install nvidia-cudss-cu12
  ```

## Quick start

```bash
cd 2D-BFS-GPU
python verify_gradient.py                      # GPU gradient vs pure dolfin-adjoint + Taylor test (~1 min)
OPTIMIZER=adam ITERS=200 python Diff_Physics_GPU.py   # the paper's Adam run (~8.5 min on an RTX A2000)
python Diff_Physics_GPU.py                     # L-BFGS-B, 300 iterations (default)
python postprocess.py Inverse_Results_GPU_*    # figures + metrics
```

## What runs where

| per epoch | original | this version |
|-----------|----------|--------------|
| NS Jacobian / residual assembly | FEniCS, CPU, every Newton step | **GPU** (convective term per cell; Stokes part is constant, assembled once by FEniCS at start-up) |
| NS Newton linear solves | PETSc LU, CPU | **GPU**, cuDSS LU (symbolic analysis once, numeric factorisation per step) |
| NS adjoint | pyadjoint, CPU LU | **GPU**, one cuDSS solve with Jᵀ |
| ADE matrices | FEniCS, re-assembled **every** time step | **GPU**, once per epoch |
| ADE 4000 forward steps | GMRES+ILU, CPU | **GPU**, factorise once (cuDSS), then 2 triangular solves / step |
| ADE 4000 adjoint steps | pyadjoint replays 4000 blocks, CPU | **GPU**, same factorisation idea with Aᵀ |
| loss `J`, regulariser, Adam, I/O | FEniCS / numpy | unchanged (tiny) |

FEniCS is only used at start-up (mesh, dof maps, the constant matrices, and a
check that the GPU-assembled matrices equal FEniCS' to ~1e-15) and for the cheap
loss assembly.

### The adjoints the blocks implement

**NS** (`w_B = g` on Dirichlet dofs, `F_I(w) = 0` elsewhere):
`[Jᵀ]_bc λ = ∂J/∂w` (bc rows → identity, rhs_B = 0), then
`∂J/∂g = ∂J/∂w_B − (Jᵀλ)_B`, mapped onto the inlet-control dofs (same P2 space,
so the Dirichlet values are exact copies of the control's dofs).

**ADE** (Crank–Nicolson, `P` zeroes the inlet rows):
forward `A_bc cⁿ⁺¹ = B_bc cⁿ + r`, `A = M/dt + ½K(u) + ½L`, `B = M/dt − ½K(u) − ½L`;
adjoint `Ã μᴺ = P ∂J/∂cᴺ`, `Ã μⁿ = P Bᵀ μⁿ⁺¹` (`Ã` = `Aᵀ` with inlet rows → identity), and
`dJ/du[v] = − Σₙ ∫ v · ( μⁿ⁺¹ ∇c_midⁿ ) dx`. `μ ∇c_mid` is exactly a DG1 vector
field, so the time-sum is accumulated on the GPU and mapped to the velocity with
one mixed-mass-matrix mat-vec.

## Verification (`verify_gradient.py`, N = 20, full-domain mask)

Reference: original FEniCS NS + per-step ADE, all taped by dolfin-adjoint (LU solves).

| | J | w (NS) | cᴺ | dJ/du_inlet |
|---|---|---|---|---|
| GPU ADE only | 2.7e-16 | — | 2.1e-15 | 6.1e-15 |
| **all GPU** | 7.0e-15 | 1.6e-12 | 9.3e-15 | **2.3e-14** |

Taylor test on the all-GPU path: rates 2.000 / 2.000 / 2.000.
Also: 1 epoch at N = 200 of `Diff_Physics_GPU.py` prints *identical* loss, gradient
norm, Vmax and Q_pred to the original script.

## Speed (N = 4000, this mesh: 37.8k triangles, 172k NS dofs, 19.3k ADE dofs)

| per epoch | original (CPU) | all-GPU | speed-up |
|---|---:|---:|---:|
| hardware | 1 CPU core (Intel Xeon cluster node) | NVIDIA RTX 2000 Ada | |
| NS solve | 14.7 s | 0.5–0.7 s | ~25× |
| ADE forward (4000 steps) | 154.2 s | 1.3 s | ~120× |
| gradient (`rf.derivative()`) | 528.4 s | 2.0 s | ~260× |
| **epoch total** | **697 s (11.6 min)** | **≈ 3.8 s** | **≈ 180×** |
| 200 epochs (extrapolated) | ≈ 39 h | ≈ 13 min | |
| peak host RAM | 5.0 GB | 1.8 GB | |

Plus a one-time start-up of ~15 s (FEniCS JIT, GPU setup, cuDSS analysis).
Loss history over 3 epochs is identical to the intermediate (GPU-ADE-only) version.
At N = 4000 the first-epoch loss differs from the original by 6.7e-4 relative;
this is most likely the original's per-step GMRES+ILU (default tolerance)
accumulating over 4000 steps — the GPU version solves every step exactly (LU).

## Optimiser: Adam vs L-BFGS-B

The original Adam loop (lr 1.0, halved 8 times) stalls on this problem. Checked at
its 200-epoch result, with the **same objective and an exact gradient**:

| control | J | data misfit | regulariser | Q (true 2.667) |
|---|---|---|---|---|
| Adam, 200 epochs | 4.53e-4 | 4.6e-5 | 4.07e-4 | 3.125 (+17 %) |
| true inlet profile, smoothest extension | 7.95e-5 | 1.8e-9 | 7.95e-5 | 2.667 |

The true inlet scores 5.7× better, so Adam is not at the optimum. Cause: the
control is a P2 field on the whole domain, but only its inlet values drive the
flow. Adam normalises every dof separately, so the ~150k dofs that only the
regulariser sees move by ~lr per step, roughen the field, and the regulariser
then dominates J. The original CPU script does exactly the same thing: at the
Adam result the CPU and GPU pipelines agree to 1e-11 (NS), 7e-5 (c_N) and 6e-4
(gradient, cosine 0.9999998, identical signs) — `check_data/check_final_grad.py`.

Same objective, same GPU gradient, N = 4000, RTX 2000 Ada:

| | Adam (original), 200 it. | **L-BFGS-B (default), 300 it.** |
|---|---|---|
| final J | 4.53e-4 | **6.39e-5** |
| data misfit | 4.6e-5 | **5.5e-7** |
| flow-rate error | 17.2 % | **1.6 %** (min 0.3 % near it. 140) |
| inlet-profile rel. L2 error | 47 % | **21 %** |
| velocity rel. L2 error (domain) | 23.7 % | **4.0 %** |
| concentration error in the band | 2.0 % | **0.22 %** |
| wall time | 8.4 min (2.5 s/it.) | 37.5 min (7.5 s/it.) |

L-BFGS-B ends *below* the true inlet's J (6.4e-5 < 7.95e-5): from there on the
regulariser, not the data, decides the profile (it flattens the peak), so the
flow-rate error creeps from 0.3 % back up to 1.6 %. That is the objective's own
bias; a smaller `gamma_reg` or stopping earlier trades it off.

## Settings

Settings: the config block at the top of `Diff_Physics_GPU.py`; the env vars
`OPTIMIZER` (lbfgs|adam), `ITERS`, `FINAL_IDX`, `DEVICE`, `OUT_DIR` override it.

GPU notes: torch is the CUDA 11.8 build and cuDSS is CUDA 12 (two runtimes, one
device context — works on CUDA-12 drivers). Tested on RTX A2000, RTX 2000/5000 Ada; should run on
V100, A100, H100/H200 and RTX 30xx. Blackwell cards need a cu128 torch build.

## Checking it yourself: `validate_gpu.py`

One Python file, no job scripts. It contains the original CPU code path (FEniCS
`solve` + per-step GMRES/ILU + dolfin-adjoint) and compares it with the GPU solver:
same-point loss/fields/gradient, a finite-difference check of the GPU gradient using
CPU solves only, and the original Adam loop run with both, side by side (costs + speed-up).

```bash
conda activate fenics_gpu            # env: see Requirements above
python validate_gpu.py                                    # quick: N=200, whole-domain mask, ~5 min
python validate_gpu.py --mask band --N 4000 --epochs 2    # the real configuration (CPU ~12 min/epoch)
python validate_gpu.py --no-cpu --mask band --N 4000 --epochs 10   # GPU cost only
```

## Limitations

* Serial (one MPI rank), 2-D triangles, Taylor–Hood P2/P1 NS, CG1 concentration —
  i.e. exactly this problem. The 3-D coronary case needs tetrahedral tables in
  `gpu_fem.py` (same structure) and much more GPU memory for the LU factors.
* The GPU keeps the whole ADE trajectory (N+1 × 19.3k doubles ≈ 0.6 GB at N = 4000).
* The original script's save section writes `.xdmf` checkpoints and then overwrites
  their `.h5` with `HDF5File`, which makes those `.xdmf` unreadable. This script
  writes each field to its own `.h5` + `.pvd` instead.
