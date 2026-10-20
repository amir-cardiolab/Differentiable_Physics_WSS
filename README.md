# WSS-PINNs-Diff

Reconstructing hemodynamics — velocity, pressure, concentration transport and
**Wall Shear Stress (WSS)** — from sparse concentration observations, using two
complementary inverse approaches:

- **PINN** — a Physics-Informed Neural Network that fits the data while
  satisfying the (Navier–Stokes + advection–diffusion) residuals.
- **Diff_Physics** — a differentiable / adjoint-based PDE solver
  ([FEniCS](https://fenicsproject.org/) + `dolfin-adjoint`) that optimises an
  inlet boundary condition so the forward simulation matches the observations.

Each method is applied to two geometries:

| Folder | Geometry | Dim |
|--------|----------|-----|
| [`2D-BFS/`](2D-BFS/) | 2-D backward-facing step | 2-D |
| [`Coronary Artery/`](Coronary%20Artery/) | Patient-specific coronary artery | 3-D |

[`2D-BFS-GPU/`](2D-BFS-GPU/) is a GPU implementation of the 2-D `Diff_Physics` inverse
(PyTorch + NVIDIA cuDSS): one optimisation epoch takes about 4 s on an RTX 2000 Ada instead
of about 12 min on one CPU core.

See the per-case READMEs for details:
- [2D-BFS/README.md](2D-BFS/README.md)
- [Coronary Artery/README.md](Coronary%20Artery/README.md)
- [2D-BFS-GPU/README.md](2D-BFS-GPU/README.md)

## Repository layout

This repo contains **code only**. The (large) input data is hosted separately —
see [Data](#data) below.

```
.
├── 2D-BFS/                       # 2-D backward-facing-step case
│   ├── PINN.py                   # PINN reconstruction
│   ├── Diff_Physics.py           # adjoint/differentiable-physics inverse (FEniCS)
│   └── Results_BFS/   # ← input data (from the data bundle)
├── 2D-BFS-GPU/                   # GPU version of 2D-BFS/Diff_Physics.py
│   ├── Diff_Physics_GPU.py       # the inverse solver (Adam or L-BFGS-B)
│   ├── gpu_ns.py, gpu_ade.py     # GPU Navier–Stokes / advection–diffusion + their adjoints
│   ├── gpu_fem.py, gpu_cudss.py  # GPU assembly tables, cuDSS binding
│   ├── verify_gradient.py, validate_gpu.py, check_data/   # checks against the CPU code
│   └── gpu_timed_run/            # timed run with pre-flight checks
└── Coronary Artery/              # 3-D coronary-artery case
    ├── PINNs.py                  # PINN reconstruction
    ├── Diff_Physics.py           # adjoint/differentiable-physics inverse (FEniCS)
    ├── CFD_Data/                 # ← PINN input snapshots (from the data bundle)
    ├── Coronary_Mesh/            # ← inverse-solver mesh (from the data bundle)
    ├── Forward_CFD/              # ← inverse-solver observations (from the data bundle)
    └── Coronary_Mask.vtk         # ← inverse-solver mask (from the data bundle)
```

## Data

The input data is **not** stored in this repo (it is several GB). Download the
data bundle from:

> **[Download the data bundle (Google Drive)](https://drive.google.com/drive/folders/1erqCDz22ayt-ipey71owG91gI8z9fPBX?usp=sharing)**

The bundle mirrors the repo layout; merge its contents into the repo root so each
data folder sits next to the script that reads it (see the bundle's own
`README.md` for the exact mapping). The coronary data is pruned to only the files
the scripts read; the 2-D data is the full forward-CFD dataset.

Generated outputs (trained weights, per-epoch fields, plots) are written to new
folders at run time and are ignored by git (see [`.gitignore`](.gitignore)).

## Dependencies

- **PINN scripts:** Python 3, `torch`, `numpy`, `matplotlib`, `vtk`
  (CUDA GPU recommended for training).
- **Diff_Physics scripts:** `fenics` (2019.x) and `fenics-adjoint`
  (`dolfin-adjoint`), plus `numpy`, `scipy`, `matplotlib`, `vtk`.
- **2D-BFS-GPU:** the above (FEniCS / dolfin-adjoint **2019.1**) plus `torch` with CUDA and
  `nvidia-cudss-cu12`; an NVIDIA GPU with a CUDA-12 driver. See
  [2D-BFS-GPU/README.md](2D-BFS-GPU/README.md#requirements).

## Quick start

```bash
# 2-D backward-facing step
cd 2D-BFS
python PINN.py            # PINN reconstruction
python Diff_Physics.py    # adjoint inverse (needs FEniCS)

# 2-D backward-facing step, adjoint inverse on the GPU (reads ../2D-BFS/Results_BFS/)
cd ../2D-BFS-GPU
OPTIMIZER=adam ITERS=200 python Diff_Physics_GPU.py

# 3-D coronary artery
cd "Coronary Artery"
python PINNs.py           # PINN reconstruction
python Diff_Physics.py    # adjoint inverse (needs FEniCS)
```
