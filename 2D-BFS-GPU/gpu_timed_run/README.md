# gpu_timed_run — timed GPU run of the 2-D BFS inverse problem

Runs the original Adam optimisation of [`2D-BFS/Diff_Physics.py`](../../2D-BFS/Diff_Physics.py)
on a GPU and times every stage of every epoch, with pre-flight checks, an ETA and a
health verdict.

## Contents

| File | What it is |
|------|-----------|
| `run_gpu_timed.py` | the script — run this (fully documented at the top); uses the `gpu_*.py` modules of `2D-BFS-GPU/` |
| `example_output_RTXA2000.txt` | full console output of a 10-epoch run on an RTX A2000 |
| `example_timing_RTXA2000.png` | its timing figure |


## Requirements

* NVIDIA GPU with ≥ 2 GB free memory, driver supporting CUDA 12 (525 or newer); ≥ 3 GB free RAM.
* The `fenics_gpu` environment: FEniCS **2019.1**, dolfin-adjoint **2019.1**, PyTorch (CUDA),
  `nvidia-cudss-cu12` (see [`../README.md`](../README.md#requirements)).
  Other envs (e.g. `fenicsenv`: dolfin 2018 + newer dolfin-adjoint) do not work — the script
  checks this and stops with a message.
* The data folder `Results_BFS/` (`mesh.xdmf`, `velocity_field.xdmf`, `C_4000.vtk`) in
  `2D-BFS-GPU/` or `2D-BFS/`, or given with `--data` / `BFS_DATA` (see `../bfs_data.py`).

## Run

```bash
conda activate fenics_gpu
cd 2D-BFS-GPU/gpu_timed_run
python run_gpu_timed.py                   # 10 epochs, N = 4000      (~1.2 min on an RTX A2000)
python run_gpu_timed.py --epochs 200      # the full run             (~19 min)
python run_gpu_timed.py --N 200 --epochs 3   # smoke test            (~30 s)
```

Options: `--epochs`, `--N`, `--data`, `--out`, `--cpu-epoch-s` (see the script header).

## Reading the output

Per epoch:

| Column | Meaning |
|--------|---------|
| `loss J` | objective at the start of the epoch |
| `Q`, `err` | inlet flow rate after the Adam step, error vs ground truth (Q_true = 2.667) |
| `NS s` / `ADE s` / `grad s` | Navier–Stokes solve / 4000 concentration steps / gradient |
| `epoch s` | whole epoch, including the Adam update |
| `ADE ms/step` | cost of one concentration time step — the simplest speed number (~0.52 ms on an A2000) |
| `avg s`, `ETA` | running average (epoch 1 excluded) and time left |
| `status` | `ok`, or `SLOW (k x the usual pace)` if an epoch takes > 1.5× the median |

`SLOW` means something else is competing — another program on the GPU, a busy CPU, or the
machine swapping (check `free -g`). Epoch 1 is always a bit slower (one-time warm-up).

The summary gives the total time, per-stage averages, peak GPU memory, an estimate for 200
epochs, the cost of the same run with the original CPU code, and `health: OK` when epoch
times are steady and the loss is finite.

## Measured on the workstation (RTX A2000 12 GB, N = 4000, 10 epochs)

| | Time |
|---|---|
| whole run (imports + set-up + 10 epochs) | **1 min 13 s** |
| set-up | 12 s |
| per epoch (steady) | **5.7 s** = NS 0.6 s + ADE 2.1 s + gradient 3.0 s |
| one ADE time step | 0.52 ms |
| peak GPU memory | 1.2 GB |
| 200 epochs (estimate) | **≈ 19 min** |
| same 10 epochs, original CPU code | ≈ 2 h 31 min (908 s/epoch, measured) → **≈ 157× faster** |

Note: with only 10 epochs the original learning-rate schedule halves the rate every
epoch, so Q stops near 0.91 (true 2.67); use `--epochs 200` for a real reconstruction.

## Outputs (`results_<E>ep_N<N>/`)

`history.json` (every epoch, written as it runs), `summary.json`, `timing.png`,
`inlet_control.h5` (final control, dataset `InletControl`).
