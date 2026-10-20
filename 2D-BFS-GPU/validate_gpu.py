"""
validate_gpu.py — independent check that the GPU solver gives the same answers as
the original CPU code, and how much faster it is. Run it yourself:

    python validate_gpu.py                                # quick: N=200, whole-domain mask (~5 min)
    python validate_gpu.py --mask band --N 4000 --epochs 2   # the real setting (CPU ~12 min/epoch)
    python validate_gpu.py --no-cpu --mask band --N 4000 --epochs 10   # GPU cost only

Why the quick default observes the WHOLE domain: with the real observation band
y in [0.3, 0.4], the dye needs many time steps to reach the band. At N = 200 it has
not, so the data term would not depend on the inlet at all and only the
regulariser (identical code in both pipelines) would be compared. The script
therefore splits the gradient into data and regulariser parts, compares the data
(physics) part on its own, and fails if that part is negligible.

Needs: the fenics_gpu environment (FEniCS 2019.1, dolfin-adjoint 2019.1, torch with
CUDA, nvidia-cudss-cu12), an NVIDIA GPU, and Results_BFS/ next to
this file. From the terminal, e.g.:

    conda create -n fenics_gpu -c conda-forge python=3.9 fenics=2019.1.0 "numpy<1.24" scipy matplotlib vtk
    conda activate fenics_gpu
    pip install --no-deps git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    pip install nvidia-cudss-cu12

What it does (all on the same data, objective and initial guess as Diff_Physics.py):

  CPU pipeline = the ORIGINAL code path, written out below:
                 FEniCS solve(F == 0) for Navier–Stokes, a LinearVariationalSolver
                 (GMRES + ILU) for each ADE time step, dolfin-adjoint tapes it all.
                 It does not import any of the gpu_*.py files.
  GPU pipeline = gpu_ns.py + gpu_ade.py (the code under test).

  Test 1  one evaluation at the initial guess: J, flow field, c_N, gradient, timings;
          the gradient is split into data (physics) and regulariser parts
  Test 2  finite differences of the DATA misfit with the CPU pipeline ONLY:
          [J_data(g+eh) - J_data(g-eh)] / 2e  vs  <GPU data gradient, h>
          — checks the GPU physics gradient without trusting any adjoint
  Test 3  the original Adam loop for --epochs epochs with BOTH pipelines, side by side
"""
from __future__ import print_function

import argparse
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=200, help="ADE time steps (the real case uses 4000)")
p.add_argument("--epochs", type=int, default=3, help="Adam epochs for test 3")
p.add_argument("--no-cpu", action="store_true", help="skip the CPU pipeline (GPU timing only)")
p.add_argument("--mask", choices=["full", "band"], default="full",
               help="observe the whole domain (quick check) or only y in [0.3, 0.4] (real case)")
p.add_argument("--threads", type=int, default=1,
               help="CPU threads for BLAS/OpenMP (default 1: the original solver is serial, "
                    "and extra threads only compete with each other)")
p.add_argument("--data", default=None, help="data folder (default: see bfs_data.py)")
args = p.parse_args()
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bfs_data import find_data_dir
args.data = find_data_dir(args.data)

# must be set before numpy / torch / PETSc are imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(args.threads)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import dolfin
import pyadjoint
if not (dolfin.__version__.startswith("2019.1") and pyadjoint.__version__.startswith("2019.1")):
    sys.exit(f"Wrong environment: dolfin {dolfin.__version__}, pyadjoint {pyadjoint.__version__}.\n"
             "This needs FEniCS 2019.1 + dolfin-adjoint 2019.1 (the fenics_gpu env, see the top of\n"
             "this file). Mixed versions (e.g. dolfin 2018 + dolfin-adjoint 2025) fail inside UFL.")
from fenics import *
from fenics_adjoint import *
import torch
torch.set_num_threads(args.threads)

set_log_level(LogLevel.WARNING)
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU visible to torch - this check needs a GPU.")

# =============================================================================
# Problem set-up — identical to 2D-BFS/Diff_Physics.py
# =============================================================================
N = args.N
x_min_geo, x_max_geo = -0.5, 4.0
y1_min, y1_max = 0.3, 0.4
INLET = f"near(x[0], {x_min_geo}, DOLFIN_EPS)"
OUTLET = f"near(x[0], {x_max_geo}, DOLFIN_EPS)"
WALLS = f"on_boundary && !(near(x[0], {x_min_geo}, DOLFIN_EPS) || near(x[0], {x_max_geo}, DOLFIN_EPS))"
rho, mu = Constant(1.06), Constant(0.04)
nu = mu / rho
D = Constant(1.0e-3)
dt = Constant(1.0e-3)
gamma_reg = Constant(2e-6)

mesh = Mesh()
with XDMFFile(os.path.join(args.data, "mesh.xdmf")) as f:
    f.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
S = FunctionSpace(mesh, "CG", 1)

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
_r = vtk.vtkGenericDataObjectReader()
_r.SetFileName(os.path.join(args.data, "C_4000.vtk"))
_r.Update()
_o = _r.GetOutput()
C_target = Function(S)
C_target.vector().set_local(vtk_to_numpy(_o.GetPointData().GetArray("C")).ravel()[
    cKDTree(vtk_to_numpy(_o.GetPoints().GetData())[:, :2]).query(S.tabulate_dof_coordinates().reshape(-1, 2))[1]])
C_target.vector().apply("insert")
mask = (Constant(1.0) if args.mask == "full" else
        Expression("x[1] >= y1min && x[1] <= y1max ? 1.0 : 0.0", y1min=y1_min, y1max=y1_max, degree=0))

u_gt = Function(V)
with XDMFFile(os.path.join(args.data, "velocity_field.xdmf")) as f:
    f.read_checkpoint(u_gt, "velocity", 0)
_bnd = MeshFunction("size_t", mesh, 1, 0)
CompiledSubDomain(INLET).mark(_bnd, 1)
ds_in = Measure("ds", domain=mesh, subdomain_data=_bnd)(1)
Q_true = assemble(u_gt[0] * ds_in)


def initial_control():
    """(1, 0) everywhere, inlet end points clamped to 0 — as in the original."""
    g = Function(V, name="InletControl")
    g.assign(project(Constant((1.0, 0.0)), V))
    v = g.vector().get_local()
    v[EDGE] = 0.0
    g.vector().set_local(v)
    g.vector().apply("insert")
    return g


_dofs = np.array(list(DirichletBC(V, Constant((0.0, 0.0)), INLET).get_boundary_values().keys()))
_y = V.tabulate_dof_coordinates().reshape(-1, 2)[_dofs, 1]
_tol = 0.01 * mesh.hmin()
EDGE = np.unique(np.concatenate([_dofs[np.abs(_y - _y.min()) <= _tol], _dofs[np.abs(_y - _y.max()) <= _tol]]))
INLET_DOFS = _dofs


# =============================================================================
# CPU pipeline — the original code path (copied from Diff_Physics.py)
# =============================================================================
def cpu_ns(g):
    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    bc_in = DirichletBC(W.sub(0), g, INLET)
    bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), WALLS)
    bc_out = DirichletBC(W.sub(1), Constant(0.0), OUTLET)
    F_ns = (inner(dot(grad(u), u), v) * dx + nu * inner(grad(u), grad(v)) * dx
            - div(v) * p * dx + q * div(u) * dx)
    solve(F_ns == 0, w, [bc_in, bc_walls, bc_out])
    return w


def cpu_ade(w, solver="gmres"):
    u_flow = split(w)[0]
    c_n = interpolate(Constant(0.0), S)
    c = Function(S)
    phi, c_trial = TestFunction(S), TrialFunction(S)
    c_mid = 0.5 * (c_trial + c_n)
    F_ade = (phi * (c_trial - c_n) / dt * dx + phi * dot(u_flow, grad(c_mid)) * dx
             + D * dot(grad(phi), grad(c_mid)) * dx)
    s = LinearVariationalSolver(LinearVariationalProblem(
        lhs(F_ade), rhs(F_ade), c, DirichletBC(S, Constant(1.0), INLET)))
    s.parameters["linear_solver"] = solver
    if solver == "gmres":
        s.parameters["preconditioner"] = "ilu"
    for _ in range(N):
        s.solve()
        c_n.assign(c)
    return c


# =============================================================================
# GPU pipeline — the code under test
# =============================================================================
from gpu_ns import GPUSteadyNS, ns_solve_gpu
from gpu_ade import GPUAdvectionDiffusion, ade_solve_gpu

print("=" * 88)
print(f"GPU: {torch.cuda.get_device_name(0)} | N = {N} time steps | mask = {args.mask} | Q_true = {Q_true:.6f}")
print(f"mesh: {mesh.num_cells()} cells | NS dofs {W.dim()} | ADE dofs {S.dim()} | control dofs {V.dim()}")
print("=" * 88)
t0 = time.time()
ns_op = GPUSteadyNS(W, V, nu=float(nu), inlet=INLET, walls=WALLS, outlet=OUTLET)
ade_op = GPUAdvectionDiffusion(S, W, D=float(D), dt=float(dt), inlet=INLET)
t_setup = time.time() - t0
print(f"GPU setup (one-time): {t_setup:.1f} s")


def sync():
    torch.cuda.synchronize()


def evaluate(pipe, g, grad=True, ade_solver="gmres"):
    """J(g) and dJ/dg with the chosen pipeline; times the forward and the gradient."""
    get_working_tape().clear_tape()
    sync()
    t0 = time.time()
    if pipe == "cpu":
        w = cpu_ns(g)
        t_ns = time.time() - t0
        c = cpu_ade(w, ade_solver)
    else:
        w = ns_solve_gpu(ns_op, g)
        sync()
        t_ns = time.time() - t0
        c = ade_solve_gpu(ade_op, w, N)
    J_data = assemble(0.5 * mask * (c - C_target) ** 2 * dx)
    J_reg = assemble(gamma_reg * inner(grad_(g), grad_(g)) * dx)
    J = J_data + J_reg
    sync()
    t_fwd = time.time() - t0
    out = dict(J=float(J), J_data=float(J_data), w=w.vector().get_local().copy(),
               c=c.vector().get_local().copy(), t_fwd=t_fwd, t_ns=t_ns, t_ade=t_fwd - t_ns,
               g=None, t_grad=0.0)
    if grad:
        t0 = time.time()
        out["g"] = ReducedFunctional(J, Control(g)).derivative().vector().get_local().copy()
        sync()
        out["t_grad"] = time.time() - t0
    return out


grad_ = grad          # keep UFL's grad reachable inside evaluate()


def reg_gradient(g):
    """Exact gradient of the regulariser alone (so the physics part can be isolated)."""
    with stop_annotating():
        return assemble(derivative(gamma_reg * inner(grad_(g), grad_(g)) * dx, g)).get_local()
rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
checks = []


def check(name, ok, detail):
    checks.append((name, ok, detail))
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# =============================================================================
# Test 1 — one evaluation at the initial guess
# =============================================================================
print("\nTEST 1  one evaluation (J + gradient) at the initial guess")
g0 = initial_control()
torch.cuda.reset_peak_memory_stats()
gpu = evaluate("gpu", g0)                     # warm-up (first call compiles kernels)
gpu = evaluate("gpu", g0)
gpu_mem = torch.cuda.max_memory_allocated() / 2 ** 20
print(f"   GPU: J = {gpu['J']:.10e}   NS {gpu['t_ns']:7.2f} s   ADE {gpu['t_ade']:7.2f} s   gradient {gpu['t_grad']:7.2f} s"
      f"   (GPU memory used by torch: {gpu_mem:.0f} MB)")
if not args.no_cpu:
    cpu = evaluate("cpu", g0)
    print(f"   CPU: J = {cpu['J']:.10e}   NS {cpu['t_ns']:7.2f} s   ADE {cpu['t_ade']:7.2f} s   gradient {cpu['t_grad']:7.2f} s")
    print(f"   speed-up (forward + gradient): {(cpu['t_fwd'] + cpu['t_grad']) / (gpu['t_fwd'] + gpu['t_grad']):.1f}x")
    check("loss J", abs(gpu["J"] - cpu["J"]) / abs(cpu["J"]) < 1e-3,
          f"rel. diff {abs(gpu['J'] - cpu['J']) / abs(cpu['J']):.2e}")
    check("Navier-Stokes solution", rel(gpu["w"], cpu["w"]) < 1e-8, f"rel. diff {rel(gpu['w'], cpu['w']):.2e}")
    check("concentration c_N", rel(gpu["c"], cpu["c"]) < 1e-3, f"rel. diff {rel(gpu['c'], cpu['c']):.2e}")
    g_reg = reg_gradient(g0)
    gd_gpu, gd_cpu = gpu["g"] - g_reg, cpu["g"] - g_reg          # physics (data-term) parts
    share = np.linalg.norm(gd_cpu) / np.linalg.norm(cpu["g"])
    print(f"   gradient norm: total {np.linalg.norm(cpu['g']):.3e} = data part {np.linalg.norm(gd_cpu):.3e}"
          f" + regulariser part {np.linalg.norm(g_reg):.3e}")
    check("data term actually drives the gradient (test is not trivial)", share > 0.01,
          f"data part = {100 * share:.1f} % of the gradient norm")
    check("DATA (physics) gradient", cos(gd_gpu, gd_cpu) > 0.9999,
          f"rel. diff {rel(gd_gpu, gd_cpu):.2e}, cosine {cos(gd_gpu, gd_cpu):.10f}")
    check("total gradient", cos(gpu["g"], cpu["g"]) > 0.9999,
          f"rel. diff {rel(gpu['g'], cpu['g']):.2e}, cosine {cos(gpu['g'], cpu['g']):.10f}")
    check("gradient signs (what Adam uses)", np.mean(np.sign(gpu["g"]) == np.sign(cpu["g"])) > 0.999,
          f"{100 * np.mean(np.sign(gpu['g']) == np.sign(cpu['g'])):.3f} % identical")

# =============================================================================
# Test 2 — finite differences with the CPU pipeline only
# =============================================================================
if not args.no_cpu:
    print("\nTEST 2  finite-difference slope of J_data (CPU forward solves only, exact LU) vs GPU data gradient")
    rng = np.random.RandomState(0)
    h = np.zeros(V.dim())
    h[INLET_DOFS] = rng.uniform(-1.0, 1.0, INLET_DOFS.size)   # perturb the inlet velocities
    h[EDGE] = 0.0
    for eps in (1e-2, 1e-3):
        Js = []
        for sgn in (+1, -1):
            gp = initial_control()
            gp.vector().set_local(gp.vector().get_local() + sgn * eps * h)
            gp.vector().apply("insert")
            with stop_annotating():
                Js.append(evaluate("cpu", gp, grad=False, ade_solver="lu")["J_data"])
        fd = (Js[0] - Js[1]) / (2 * eps)
        ad = float(gd_gpu @ h)
        print(f"   eps = {eps:.0e}:  finite difference {fd:+.10e}   GPU data gradient . h {ad:+.10e}"
              f"   rel. diff {abs(fd - ad) / abs(fd):.2e}")
    check("GPU physics gradient vs CPU finite differences", abs(fd - ad) / abs(fd) < 1e-4,
          f"rel. diff {abs(fd - ad) / abs(fd):.2e} (eps = 1e-3)")

# =============================================================================
# Test 3 — the original Adam loop, both pipelines side by side
# =============================================================================
print(f"\nTEST 3  original Adam loop, {args.epochs} epochs")
lr0, n_decay, gam = 1.0, 8, 0.5
b1, b2, eps_adam = 0.9, 0.999, 1e-8


def lr_for_epoch(ep, epochs):
    step = max(1, epochs // n_decay)
    return lr0 * gam ** sum(1 for m in (step * k for k in range(1, n_decay + 1)) if ep >= m)


def adam_run(pipe):
    g = initial_control()
    m = np.zeros(V.dim())
    v = np.zeros(V.dim())
    hist = []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        r = evaluate(pipe, g)
        lr = lr_for_epoch(ep, args.epochs)
        m = b1 * m + (1 - b1) * r["g"]
        v = b2 * v + (1 - b2) * r["g"] ** 2
        new = g.vector().get_local() - lr * (m / (1 - b1 ** ep)) / (np.sqrt(v / (1 - b2 ** ep)) + eps_adam)
        new[EDGE] = 0.0
        g.vector().set_local(new)
        g.vector().apply("insert")
        Qp = assemble(g[0] * ds_in)
        hist.append((ep, r["J"], np.linalg.norm(r["g"]), Qp, r["t_fwd"] + r["t_grad"]))
    return hist, time.time() - t_start


runs = {"gpu": adam_run("gpu")}
if not args.no_cpu:
    runs["cpu"] = adam_run("cpu")
print(f"   {'epoch':>5} | {'J (GPU)':>14} {'J (CPU)':>14} | {'Q after step (GPU)':>18} {'(CPU)':>9} | "
      f"{'s (GPU)':>8} {'s (CPU)':>8}")
for i in range(args.epochs):
    eg, Jg, _, Qg, tg = runs["gpu"][0][i]
    line = f"   {eg:5d} | {Jg:14.8e} "
    if "cpu" in runs:
        _, Jc, _, Qc, tc = runs["cpu"][0][i]
        line += f"{Jc:14.8e} | {Qg:18.6f} {Qc:9.6f} | {tg:8.2f} {tc:8.2f}"
    else:
        line += f"{'-':>14} | {Qg:18.6f} {'-':>9} | {tg:8.2f} {'-':>8}"
    print(line)
tg_tot = runs["gpu"][1]
print(f"   total for {args.epochs} epochs:  GPU {tg_tot:.1f} s", end="")
if "cpu" in runs:
    tc_tot = runs["cpu"][1]
    print(f"   CPU {tc_tot:.1f} s   ->  {tc_tot / tg_tot:.1f}x faster")
    Jg_last, Jc_last = runs["gpu"][0][-1][1], runs["cpu"][0][-1][1]
    Qg_last, Qc_last = runs["gpu"][0][-1][3], runs["cpu"][0][-1][3]
    check("Adam trajectory: last-epoch loss", abs(Jg_last - Jc_last) / abs(Jc_last) < 1e-2,
          f"rel. diff {abs(Jg_last - Jc_last) / abs(Jc_last):.2e}")
    check("Adam trajectory: flow rate after last step", abs(Qg_last - Qc_last) / abs(Qc_last) < 1e-2,
          f"GPU {Qg_last:.6f} vs CPU {Qc_last:.6f}")
else:
    print()

# =============================================================================
print("\n" + "=" * 88)
print(f"TIMING SUMMARY (N = {N}, {torch.cuda.get_device_name(0)}, CPU threads = {args.threads})")
print(f"   {'':34} {'CPU (original)':>16} {'GPU':>10} {'speed-up':>9}")
for label, k in (("Navier-Stokes solve", "t_ns"), ("ADE forward (N steps)", "t_ade"),
                 ("gradient (all adjoints)", "t_grad")):
    if not args.no_cpu:
        print(f"   {label:34} {cpu[k]:14.2f} s {gpu[k]:8.2f} s {cpu[k] / gpu[k]:8.1f}x")
    else:
        print(f"   {label:34} {'-':>16} {gpu[k]:8.2f} s")
tot_g = gpu["t_fwd"] + gpu["t_grad"]
if not args.no_cpu:
    tot_c = cpu["t_fwd"] + cpu["t_grad"]
    print(f"   {'one epoch (forward + gradient)':34} {tot_c:14.2f} s {tot_g:8.2f} s {tot_c / tot_g:8.1f}x")
    print(f"   {'Adam, %d epochs (test 3)' % args.epochs:34} {runs['cpu'][1]:14.1f} s {runs['gpu'][1]:8.1f} s "
          f"{runs['cpu'][1] / runs['gpu'][1]:8.1f}x")
else:
    print(f"   {'one epoch (forward + gradient)':34} {'-':>16} {tot_g:8.2f} s")
    print(f"   {'Adam, %d epochs (test 3)' % args.epochs:34} {'-':>16} {runs['gpu'][1]:8.1f} s")
print(f"   GPU one-time setup: {t_setup:.1f} s   |   GPU memory used by torch: {gpu_mem:.0f} MB")
print("=" * 88)
if checks:
    n_ok = sum(ok for _, ok, _ in checks)
    print(f"RESULT: {n_ok}/{len(checks)} checks passed" + ("" if n_ok == len(checks) else "  <-- see FAIL lines"))
    print("Small CPU/GPU differences (up to ~1e-3 at N = 4000) come from the original's GMRES tolerance\n"
          "in the per-step ADE solve; the GPU solves every step exactly (LU).")
else:
    print("GPU-only run (--no-cpu): timings above, no comparisons.")
print("=" * 88)
