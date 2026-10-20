"""
run_gpu_timed.py
================

Run the 2-D backward-facing-step (BFS) inverse problem on a GPU — the original
Adam optimisation loop of 2D-BFS/Diff_Physics.py — and time every part of it,
so you can see at a glance that the GPU is doing the work and the run is healthy.

Quick start
-----------
    source ~/envs/fenics_gpu/bin/activate       # FEniCS 2019.1 + dolfin-adjoint 2019.1 + PyTorch + cuDSS
    python run_gpu_timed.py                      # 10 epochs, N = 4000 time steps  (~1.5 min on an RTX A2000)
    python run_gpu_timed.py --epochs 200         # full run                        (~20 min)
    python run_gpu_timed.py --N 200 --epochs 3   # smoke test                      (~30 s)

Options
-------
    --epochs E        Adam epochs (default 10)
    --N N             advection-diffusion time steps per epoch (default 4000 = the real case)
    --data DIR        folder with mesh.xdmf, velocity_field.xdmf, C_4000.vtk
                      (default: Results_BFS/ in 2D-BFS-GPU/ or 2D-BFS/, or $BFS_DATA)
    --out DIR         results folder (default: results_<E>ep_N<N>/ next to this script)
    --cpu-epoch-s S   seconds per epoch of the ORIGINAL CPU code at N = 4000, only used for the
                      comparison line in the summary (default 908 s = measured on the RTX A2000
                      workstation, one core)

The problem (unchanged from Diff_Physics.py)
--------------------------------------------
Find the inlet velocity g(y) such that a steady Navier–Stokes flow (Taylor–Hood P2/P1) with
u = g on the inlet, carrying a dye c (CG1, Crank–Nicolson, c = 1 on the inlet, c(0) = 0) for
N steps of dt = 1e-3, reproduces the observed concentration inside the band y in [0.3, 0.4]:

    J(g) = 1/2 ∫ mask (c_N - C_target)^2 dx  +  2e-6 ∫ |grad g|^2 dx

Each epoch = 1 Navier–Stokes solve + N concentration steps + the gradient dJ/dg
(dolfin-adjoint), followed by one Adam update (lr 1.0, halved 8 times over the run;
the two ends of the inlet are held at 0).

How the GPU is used (files in this folder)
------------------------------------------
    gpu_ns.py     Navier–Stokes: the convective terms are assembled cell-by-cell on the GPU,
                  Newton steps and the adjoint are solved with NVIDIA cuDSS (GPU sparse LU).
    gpu_ade.py    concentration: matrices built once per epoch on the GPU and factorised once;
                  each of the N forward and N adjoint steps is two GPU triangular solves.
    gpu_fem.py    finite-element tables and sparse-matrix bookkeeping shared by the two.
    gpu_cudss.py  small ctypes binding to cuDSS.
Both solvers are "custom blocks" of dolfin-adjoint, so dolfin-adjoint still assembles the
gradient; it equals the original CPU gradient (checked in ../validate_gpu.py).

What the script prints
----------------------
1. PRE-FLIGHT   environment versions, GPU name and free memory, free RAM, other GPU users.
                Stops early with a clear message if the environment or GPU is wrong.
2. one row per epoch:
      loss J            objective value at the start of the epoch
      Q, err            inlet flow rate after the Adam step, and its error vs the ground truth
      NS / ADE / grad   seconds for the Navier–Stokes solve, the N concentration steps, the gradient
      epoch s           wall time of the whole epoch (incl. the Adam update)
      ADE ms/step       concentration cost per time step — the best single "speed" number
      avg s, ETA        running average (epoch 1 excluded: it contains one-time warm-up) and
                        estimated time left
      status            "ok", or "SLOW (k x the usual pace)" when an epoch takes > 1.5x the median:
                        something else is using the GPU/CPU, or the machine is swapping
3. SUMMARY      total time, per-stage averages, peak GPU memory, estimate for 200 epochs,
                the same run with the original CPU code, and a health verdict.

Files written to --out
----------------------
    history.json       every epoch: J, Q, learning rate, stage times (updated as it runs)
    summary.json       the numbers of the SUMMARY block
    timing.png         time per epoch by stage + loss per epoch
    inlet_control.h5   the final inlet control (HDF5, dataset "InletControl")
"""
from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import time

# -----------------------------------------------------------------------------
# Command line. Thread limits must be set before numpy / torch / PETSc load: the
# CPU side of this program is serial, extra BLAS/OpenMP threads only compete.
# -----------------------------------------------------------------------------
p = argparse.ArgumentParser(description="Timed GPU run of the 2-D BFS inverse problem (Adam).")
p.add_argument("--epochs", type=int, default=10, help="Adam epochs (default 10)")
p.add_argument("--N", type=int, default=4000, help="time steps per epoch (default 4000 = real case)")
p.add_argument("--data", default=None, help="data folder (default: auto-detect)")
p.add_argument("--out", default=None, help="results folder")
p.add_argument("--cpu-epoch-s", type=float, default=908.0,
               help="original CPU code, seconds/epoch at N=4000 (for the comparison line)")
args = p.parse_args()
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # the gpu_*.py modules and bfs_data.py of 2D-BFS-GPU/


def find_data():
    """Data folder: --data, else see bfs_data.py (BFS_DATA, Results_BFS/ next to the scripts or in ../2D-BFS/)."""
    from bfs_data import find_data_dir
    return find_data_dir(args.data)


DATA = find_data()
OUT = os.path.abspath(args.out or os.path.join(HERE, f"results_{args.epochs}ep_N{args.N}"))
os.makedirs(OUT, exist_ok=True)


def hms(s):
    """Seconds -> '1h02m03s' / '4m05s'."""
    s = int(round(s))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


def rule(ch="=", n=100):
    print(ch * n)


# =============================================================================
# 1) PRE-FLIGHT: refuse to run in a wrong environment, report the machine state
# =============================================================================
rule()
print("PRE-FLIGHT")
import dolfin
import pyadjoint
env_ok = dolfin.__version__.startswith("2019.1") and pyadjoint.__version__.startswith("2019.1")
print(f"   FEniCS {dolfin.__version__} | dolfin-adjoint {pyadjoint.__version__}   "
      + ("OK" if env_ok else "<-- WRONG ENVIRONMENT: need 2019.1 for both "
                             "(source ~/envs/fenics_gpu/bin/activate)"))
if not env_ok:
    sys.exit(1)

import numpy as np
import torch
torch.set_num_threads(1)
if not torch.cuda.is_available():
    sys.exit("   No CUDA GPU visible to PyTorch - stopping.")
free_b, total_b = torch.cuda.mem_get_info()
print(f"   GPU: {torch.cuda.get_device_name(0)} | free {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GB   "
      + ("OK" if free_b > 2 * 2**30 else "<-- under 2 GB free: close other GPU programs"))
try:
    meminfo = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    avail_gb = meminfo["MemAvailable"] / 2**20
    print(f"   host RAM available: {avail_gb:.1f} GB   "
          + ("OK" if avail_gb > 3 else "<-- low: close programs (swapping slows everything down)"))
except Exception:
    pass
try:
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
    others = [a.strip() for a in apps.splitlines() if a.strip() and not a.startswith(str(os.getpid()))]
    print(f"   other processes on the GPU: {len(others)}" + (" (they share it)" if others else ""))
    for a in others:
        print("      " + a)
except Exception:
    pass
print(f"   data: {DATA}")
print(f"   run: {args.epochs} Adam epochs x N = {args.N} time steps  ->  {OUT}")

# =============================================================================
# 2) SET-UP: same data, objective and initial guess as Diff_Physics.py
# =============================================================================
t0 = time.time()
from fenics import *
from fenics_adjoint import *
from gpu_ns import GPUSteadyNS, ns_solve_gpu
from gpu_ade import GPUAdvectionDiffusion, ade_solve_gpu
set_log_level(LogLevel.WARNING)

# geometry / boundaries (physical coordinates of the mesh)
x_min_geo, x_max_geo = -0.5, 4.0
y1_min, y1_max = 0.3, 0.4                      # observation band
INLET = f"near(x[0], {x_min_geo}, DOLFIN_EPS)"
OUTLET = f"near(x[0], {x_max_geo}, DOLFIN_EPS)"
WALLS = f"on_boundary && !(near(x[0], {x_min_geo}, DOLFIN_EPS) || near(x[0], {x_max_geo}, DOLFIN_EPS))"
# physics
nu = 0.04 / 1.06                               # mu / rho
D, dt = 1.0e-3, 1.0e-3                         # diffusivity, time step
gamma_reg = 2e-6                               # regulariser weight

mesh = Mesh()
with XDMFFile(os.path.join(DATA, "mesh.xdmf")) as f:
    f.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)                                     # velocity / control
W = FunctionSpace(mesh, MixedElement([V.ufl_element(),
                                      FiniteElement("CG", mesh.ufl_cell(), 1)]))  # (u, p)
S = FunctionSpace(mesh, "CG", 1)                                           # concentration


def load_target():
    """Observed concentration (step 4000) from C_4000.vtk, mapped to the CG1 dofs by nearest point."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.spatial import cKDTree
    r = vtk.vtkGenericDataObjectReader()
    r.SetFileName(os.path.join(DATA, "C_4000.vtk"))
    r.Update()
    o = r.GetOutput()
    nn = cKDTree(vtk_to_numpy(o.GetPoints().GetData())[:, :2]).query(
        S.tabulate_dof_coordinates().reshape(-1, 2))[1]
    C = Function(S)
    C.vector().set_local(vtk_to_numpy(o.GetPointData().GetArray("C")).ravel()[nn])
    C.vector().apply("insert")
    return C


C_target = load_target()
mask = Expression("x[1] >= a && x[1] <= b ? 1.0 : 0.0", a=y1_min, b=y1_max, degree=0)

# ground truth, only to report the flow-rate error
u_gt = Function(V)
with XDMFFile(os.path.join(DATA, "velocity_field.xdmf")) as f:
    f.read_checkpoint(u_gt, "velocity", 0)
_bnd = MeshFunction("size_t", mesh, 1, 0)
CompiledSubDomain(INLET).mark(_bnd, 1)
ds_in = Measure("ds", domain=mesh, subdomain_data=_bnd)(1)
Q_true = assemble(u_gt[0] * ds_in)

# control: plug flow u_x = 1, with the two inlet end points held at 0 (as in the original)
g = Function(V, name="InletControl")
g.assign(project(Constant((1.0, 0.0)), V))
_idofs = np.array(list(DirichletBC(V, Constant((0, 0)), INLET).get_boundary_values().keys()))
_y = V.tabulate_dof_coordinates().reshape(-1, 2)[_idofs, 1]
_tol = 0.01 * mesh.hmin()
EDGE = _idofs[(np.abs(_y - _y.min()) <= _tol) | (np.abs(_y - _y.max()) <= _tol)]
_gv = g.vector().get_local()
_gv[EDGE] = 0.0
g.vector().set_local(_gv)
g.vector().apply("insert")

# GPU operators (mesh-only work, done once; they check themselves against FEniCS assembly)
ns_op = GPUSteadyNS(W, V, nu=nu, inlet=INLET, walls=WALLS, outlet=OUTLET, verbose=False)
ade_op = GPUAdvectionDiffusion(S, W, D=D, dt=dt, inlet=INLET, verbose=False)
t_setup = time.time() - t0
print(f"   set-up (mesh, data, GPU operators, GPU-vs-FEniCS assembly check): {t_setup:.1f} s")


# =============================================================================
# 3) THE ADAM LOOP, timed stage by stage
# =============================================================================
def sync():
    """Wait for the GPU, so the stage timers measure finished work."""
    torch.cuda.synchronize()


def lr_for_epoch(ep):
    """Original schedule: lr 1.0, halved at 8 evenly spaced milestones over the run."""
    step = max(1, args.epochs // 8)
    return 1.0 * 0.5 ** sum(1 for k in range(1, 9) if ep >= step * k)


ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
m = np.zeros(V.dim())                          # Adam first moment
v = np.zeros(V.dim())                          # Adam second moment
history = []
torch.cuda.reset_peak_memory_stats()

rule()
print(f"{'epoch':>5} | {'loss J':>12} | {'Q':>7} {'err':>6} | {'NS s':>6} {'ADE s':>6} {'grad s':>6} "
      f"{'epoch s':>7} | {'ADE ms/step':>11} | {'avg s':>6} | {'ETA':>8} | status")
rule("-")
t_loop = time.time()
for ep in range(1, args.epochs + 1):
    t_ep0 = time.time()
    get_working_tape().clear_tape()

    sync(); t = time.time()                    # (a) steady Navier-Stokes, GPU
    w = ns_solve_gpu(ns_op, g)
    sync(); t_ns = time.time() - t

    t = time.time()                            # (b) N concentration steps + loss, GPU
    c = ade_solve_gpu(ade_op, w, args.N)
    J_data = assemble(0.5 * mask * (c - C_target) ** 2 * dx)
    J = J_data + assemble(Constant(gamma_reg) * inner(grad(g), grad(g)) * dx)
    sync(); t_ade = time.time() - t

    t = time.time()                            # (c) gradient: dolfin-adjoint + GPU adjoints
    dJ = ReducedFunctional(J, Control(g)).derivative().vector().get_local()
    sync(); t_grad = time.time() - t

    lr = lr_for_epoch(ep)                      # (d) Adam update, CPU (cheap)
    m = ADAM_B1 * m + (1 - ADAM_B1) * dJ
    v = ADAM_B2 * v + (1 - ADAM_B2) * dJ ** 2
    g_new = g.vector().get_local() - lr * (m / (1 - ADAM_B1 ** ep)) / (np.sqrt(v / (1 - ADAM_B2 ** ep)) + ADAM_EPS)
    g_new[EDGE] = 0.0
    g.vector().set_local(g_new)
    g.vector().apply("insert")
    Q = assemble(g[0] * ds_in)
    t_ep = time.time() - t_ep0

    history.append(dict(epoch=ep, J=float(J), Q=float(Q), lr=lr, t_ns=t_ns, t_ade=t_ade,
                        t_grad=t_grad, t_epoch=t_ep))
    steady = [h["t_epoch"] for h in history[1:]] or [t_ep]   # epoch 1 contains one-time warm-up
    avg, med = float(np.mean(steady)), float(np.median(steady))
    eta = avg * (args.epochs - ep)
    if not np.isfinite(J):
        status = "ERROR: loss is not finite"
    elif ep == 1:
        status = "ok (includes one-time warm-up)"
    elif ep > 2 and t_ep > 1.5 * med:
        status = f"SLOW ({t_ep / med:.1f}x the usual pace: GPU/CPU busy or swapping?)"
    else:
        status = "ok"
    print(f"{ep:5d} | {float(J):12.6e} | {Q:7.4f} {100 * abs(Q - Q_true) / Q_true:5.1f}% | "
          f"{t_ns:6.2f} {t_ade:6.2f} {t_grad:6.2f} {t_ep:7.2f} | {1e3 * t_ade / args.N:11.3f} | "
          f"{avg:6.2f} | {hms(eta):>8} | {status}")
    json.dump(dict(history=history, Q_true=Q_true, N=args.N, epochs=args.epochs),
              open(os.path.join(OUT, "history.json"), "w"), indent=1)
    if ep == 1 and args.epochs > 1:
        print(f"      -> first estimate for all {args.epochs} epochs: ~{hms(t_ep * args.epochs)} "
              f"(refined from epoch 2 on)")
t_loop = time.time() - t_loop

with HDF5File(mesh.mpi_comm(), os.path.join(OUT, "inlet_control.h5"), "w") as f:
    f.write(g, "InletControl")

# =============================================================================
# 4) SUMMARY (+ summary.json, timing.png)
# =============================================================================
arr = lambda k: np.array([h[k] for h in history])
t_ns_a, t_ade_a, t_grad_a, t_ep_a = arr("t_ns"), arr("t_ade"), arr("t_grad"), arr("t_epoch")
st = slice(1, None) if len(history) > 1 else slice(None)          # steady epochs
per_ep = float(t_ep_a[st].mean())
total = time.time() - T_START
est200 = t_setup + t_ep_a[0] + per_ep * 199
spread = float(t_ep_a[st].max() / t_ep_a[st].min())
healthy = spread < 1.5 and np.isfinite(history[-1]["J"])

rule()
print(f"SUMMARY  ({torch.cuda.get_device_name(0)}, N = {args.N}, {args.epochs} epochs)")
print(f"   total wall time               {hms(total):>9}   ({total:.1f} s, incl. imports + set-up {t_setup:.1f} s)")
print(f"   optimisation loop             {hms(t_loop):>9}   ({t_loop:.1f} s)")
print(f"   per epoch (steady, excl. #1)  {per_ep:9.2f} s   (min {t_ep_a[st].min():.2f}, max {t_ep_a[st].max():.2f})")
print(f"      Navier-Stokes solve        {t_ns_a[st].mean():9.2f} s")
print(f"      ADE forward ({args.N} steps)   {t_ade_a[st].mean():9.2f} s   ({1e3 * t_ade_a[st].mean() / args.N:.3f} ms per step)")
print(f"      gradient (adjoints)        {t_grad_a[st].mean():9.2f} s")
print(f"   peak GPU memory (torch)       {torch.cuda.max_memory_allocated() / 2**20:9.0f} MB")
print(f"   estimate for 200 epochs       {hms(est200):>9}")
cpu_total = None
if args.N == 4000:
    cpu_total = args.cpu_epoch_s * args.epochs
    print(f"   same {args.epochs} epochs, original CPU code: ~{hms(cpu_total)} ({args.cpu_epoch_s:.0f} s/epoch, "
          f"measured)  ->  ~{cpu_total / t_loop:.0f}x faster on the GPU")
print(f"   health: " + ("OK - steady epoch times, finite loss" if healthy else
                        f"CHECK - epoch times vary x{spread:.1f} or loss not finite"))
print(f"   final: J = {history[-1]['J']:.6e}, Q = {history[-1]['Q']:.4f} (true {Q_true:.4f})")
print(f"   results: {OUT}")
rule()

json.dump(dict(gpu=torch.cuda.get_device_name(0), N=args.N, epochs=args.epochs, total_s=total,
               setup_s=t_setup, loop_s=t_loop, epoch1_s=float(t_ep_a[0]), per_epoch_s=per_ep,
               ns_s=float(t_ns_a[st].mean()), ade_s=float(t_ade_a[st].mean()),
               grad_s=float(t_grad_a[st].mean()), ade_ms_per_step=1e3 * float(t_ade_a[st].mean()) / args.N,
               peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20, estimate_200_epochs_s=est200,
               cpu_same_run_s=cpu_total, healthy=bool(healthy), final_J=history[-1]["J"],
               final_Q=history[-1]["Q"], Q_true=Q_true),
          open(os.path.join(OUT, "summary.json"), "w"), indent=1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = arr("epoch")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(ep, t_ns_a, color="#8c8b86", label="Navier-Stokes", edgecolor="white", linewidth=0.5)
    ax[0].bar(ep, t_ade_a, bottom=t_ns_a, color="#2a78d6", label=f"ADE ({args.N} steps)",
              edgecolor="white", linewidth=0.5)
    ax[0].bar(ep, t_grad_a, bottom=t_ns_a + t_ade_a, color="#eb6834", label="gradient",
              edgecolor="white", linewidth=0.5)
    ax[0].set(title=f"Time per epoch on {torch.cuda.get_device_name(0)}", xlabel="epoch", ylabel="seconds")
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax[1].semilogy(ep, arr("J"), color="#2a78d6", marker="o", markersize=4)
    ax[1].set(title="Loss J", xlabel="epoch")
    for a in ax:
        a.set_axisbelow(True)
        a.grid(True, color="#e4e3df", linewidth=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "timing.png"), dpi=150)
except Exception as e:                           # the figure is optional
    print("   (timing.png not written:", e, ")")
