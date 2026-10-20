"""
Diff_Physics_GPU.py — inlet-velocity reconstruction for the 2-D backward-facing
step, from a single concentration snapshot, with every PDE solve on the GPU.

Problem
-------
Find the inlet velocity profile g(y) such that

    steady Navier–Stokes (Taylor–Hood P2/P1)     u, p   with  u = g on the inlet
    advection–diffusion (CG1, Crank–Nicolson)    c(t)   with  c = 1 on the inlet, c(0) = 0

reproduces the observed concentration C_target = c(t = FINAL_IDX·dt) inside
the observation band y ∈ [y1_min, y1_max]:

    J(g) = ½ ∫ mask (c_N − C_target)² dx  +  γ ∫ |∇g|² dx

This is the same objective, discretisation and data as ../2D-BFS/Diff_Physics.py.

How the GPU is used
-------------------
dolfin-adjoint (pyadjoint) still records the objective and computes dJ/dg, but
the two PDE solves are custom pyadjoint Blocks that run on the GPU:

  gpu_ns.py   steady NS: convective residual/Jacobian assembled per cell on the
              GPU, Newton steps + adjoint solved with cuDSS (GPU sparse LU).
  gpu_ade.py  N Crank–Nicolson steps: matrices built once per evaluation on the
              GPU, factorised once with cuDSS; forward and adjoint are 2 GPU
              triangular solves per step.

The gradient equals the pure-FEniCS/dolfin-adjoint gradient to ~1e-14
(verify_gradient.py). At the final control of a 200-epoch run the GPU and the
original CPU pipeline agree to 1e-11 (NS) and 6e-4 (gradient; the CPU's GMRES
tolerance), with identical gradient signs.

Optimiser
---------
OPTIMIZER=lbfgs (default)  L-BFGS-B via pyadjoint.minimize. Recovers the inlet
                           flow rate to < 1 % on this case.
OPTIMIZER=adam             the original hand-written Adam loop (lr 1.0, 8 step
                           decays of ×0.5). Kept for comparison: Adam normalises
                           every dof separately, so the ~150k control dofs that
                           only the regulariser sees drift by ~lr per step; it
                           stalls at J ≈ 4.5e-4 with a 17 % flow-rate error,
                           while the true inlet has J ≈ 8e-5.

Usage (on a GPU machine, env as in the README's Requirements)
-------------------------------------------------------------
    python Diff_Physics_GPU.py                          # L-BFGS-B, 300 iterations
    OPTIMIZER=adam ITERS=200 python Diff_Physics_GPU.py # original Adam loop
    ITERS=5 FINAL_IDX=200 python Diff_Physics_GPU.py    # quick smoke test

Outputs (in OUT_DIR, printed at start)
--------------------------------------
    history.json / history.npy   per-iteration J, J_data, J_reg, |dJ/dg| (Adam), Q, times
    inlet_control.h5             optimised control g            (HDF5, "InletControl")
    velocity.h5, pressure.h5     NS solution for the optimised g (HDF5, "velocity"/"pressure")
    concentration.h5             c at the final step             (HDF5, "C")
    *.pvd / *.vtu                the same fields for ParaView
    summary.txt, summary.png     flow-rate error, loss curve, inlet profile
"""
from __future__ import print_function

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import pause_annotation

from gpu_ns import GPUSteadyNS, ns_solve_gpu
from gpu_ade import GPUAdvectionDiffusion, ade_solve_gpu

t_script = time.time()
set_log_level(LogLevel.WARNING)

# =============================================================================
# 1) CONFIGURATION            (env vars override the defaults in CAPITALS)
# =============================================================================
from bfs_data import find_data_dir
DATA_DIR = find_data_dir()                               # see bfs_data.py
OPTIMIZER = os.environ.get("OPTIMIZER", "lbfgs")        # "lbfgs" | "adam"
ITERS = int(os.environ.get("ITERS", 300 if OPTIMIZER == "lbfgs" else 200))
FINAL_IDX = int(os.environ.get("FINAL_IDX", 4000))       # observed time step
DEVICE = os.environ.get("DEVICE", "cuda")

# physics (identical to the forward CFD that generated the data)
rho, mu = 1.06, 0.04
nu = mu / rho
D = 1.0e-3                    # diffusivity
dt = 1.0e-3                   # ADE time step
gamma_reg = 2e-6              # weight of the gradient regulariser on g

# observation window: data misfit only for y1_min <= y <= y1_max
# (second band: set y2_min, y2_max; None disables it). Physical mesh y.
y1_min, y1_max = 0.3, 0.4
y2_min, y2_max = None, None

# geometry: inlet at x = x_min_geo, outlet at x = x_max_geo, walls elsewhere
x_min_geo, x_max_geo = -0.5, 4.0
INLET = f"near(x[0], {x_min_geo}, DOLFIN_EPS)"
OUTLET = f"near(x[0], {x_max_geo}, DOLFIN_EPS)"
WALLS = f"on_boundary && !({INLET} || {OUTLET})"

# Adam settings (OPTIMIZER=adam only) — as in the original script
adam_lr, adam_decay_times, adam_decay_gamma = 1.0, 8, 0.5
adam_b1, adam_b2, adam_eps = 0.9, 0.999, 1e-8

band = f"y[{y1_min},{y1_max}]" + (f"_y[{y2_min},{y2_max}]" if y2_min is not None else "")
OUT_DIR = os.environ.get(
    "OUT_DIR", f"Inverse_Results_GPU_{OPTIMIZER}_T{FINAL_IDX}_{band}_gamma{gamma_reg:.0e}_it{ITERS}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[config] optimizer={OPTIMIZER} iters={ITERS} N={FINAL_IDX} device={DEVICE} -> {OUT_DIR}")

# =============================================================================
# 2) MESH, SPACES, DATA
# =============================================================================
mesh = Mesh()
with XDMFFile(os.path.join(DATA_DIR, "mesh.xdmf")) as f:
    f.read(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)                                  # velocity / control
W = FunctionSpace(mesh, MixedElement([V.ufl_element(),
                                      FiniteElement("CG", mesh.ufl_cell(), 1)]))  # (u, p)
S = FunctionSpace(mesh, "CG", 1)                                        # concentration


def load_target_concentration():
    """C at step FINAL_IDX from C_4000.vtk, mapped onto the CG1 dofs by nearest point.
    (The VTK stores the CG1 field in ParaView's cell-expanded form.)"""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.spatial import cKDTree
    path = os.path.join(DATA_DIR, "C_4000.vtk")
    r = vtk.vtkGenericDataObjectReader()
    r.SetFileName(path)
    r.Update()
    o = r.GetOutput()
    xy = vtk_to_numpy(o.GetPoints().GetData())[:, :2]
    vals = vtk_to_numpy(o.GetPointData().GetArray("C")).ravel()
    nn = cKDTree(xy).query(S.tabulate_dof_coordinates().reshape(-1, 2))[1]
    C = Function(S, name="C_target")
    C.vector().set_local(vals[nn])
    C.vector().apply("insert")
    return C


C_target = load_target_concentration()

u_gt = Function(V, name="velocity_gt")          # ground truth, for diagnostics only
with XDMFFile(os.path.join(DATA_DIR, "velocity_field.xdmf")) as f:
    f.read_checkpoint(u_gt, "velocity", 0)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
CompiledSubDomain(INLET).mark(boundaries, 1)
ds_in = Measure("ds", domain=mesh, subdomain_data=boundaries)(1)
Q_true = assemble(u_gt[0] * ds_in)

cond = "(x[1] >= y1min && x[1] <= y1max)"
kw = dict(y1min=y1_min, y1max=y1_max)
if y2_min is not None:
    cond += " || (x[1] >= y2min && x[1] <= y2max)"
    kw.update(y2min=y2_min, y2max=y2_max)
mask = Expression(f"({cond}) ? 1.0 : 0.0", degree=0, **kw)

n_band = int(np.sum([assemble(mask * TestFunction(S) * dx).get_local() > 0]))
print(f"[data] cells={mesh.num_cells()} | NS dofs={W.dim()} | ADE dofs={S.dim()} | "
      f"C_target in [{C_target.vector().min():.3f}, {C_target.vector().max():.3f}] | "
      f"band dofs={n_band} | Q_true={Q_true:.6f}")

# =============================================================================
# 3) CONTROL: inlet velocity g (a P2 field; only its inlet values drive the flow)
# =============================================================================
g = Function(V, name="InletControl")
g.assign(project(Constant((1.0, 0.0)), V))       # initial guess: plug flow u_x = 1


def inlet_edge_dofs():
    """Control dofs at the two ends of the inlet (wall corners) — held at 0."""
    dofs = np.array(list(DirichletBC(V, Constant((0, 0)), INLET).get_boundary_values().keys()))
    y = V.tabulate_dof_coordinates().reshape(-1, 2)[dofs, 1]
    tol = 0.01 * mesh.hmin()
    return dofs[(np.abs(y - y.min()) <= tol) | (np.abs(y - y.max()) <= tol)]


edge = inlet_edge_dofs()
gv = g.vector().get_local()
gv[edge] = 0.0
g.vector().set_local(gv)
g.vector().apply("insert")

# =============================================================================
# 4) GPU OPERATORS (mesh-only setup, done once; includes GPU-vs-FEniCS checks)
# =============================================================================
ns_op = GPUSteadyNS(W, V, nu=nu, inlet=INLET, walls=WALLS, outlet=OUTLET, device=DEVICE)
ade_op = GPUAdvectionDiffusion(S, W, D=D, dt=dt, inlet=INLET, device=DEVICE)


def objective(control):
    """Record J(control) on the pyadjoint tape; returns (J, J_data, J_reg, w, c)."""
    w = ns_solve_gpu(ns_op, control)                    # steady NS on the GPU
    c = ade_solve_gpu(ade_op, w, FINAL_IDX)             # N ADE steps on the GPU
    J_data = assemble(0.5 * mask * (c - C_target) ** 2 * dx)
    J_reg = assemble(Constant(gamma_reg) * inner(grad(control), grad(control)) * dx)
    return J_data + J_reg, J_data, J_reg, w, c


def flow_rate(control):
    return assemble(control[0] * ds_in)


history = []
HEADER = (f"{'it':>4} | {'J':>12} | {'J_data':>10} | {'J_reg':>10} | {'|dJ/dg|':>9} | "
          f"{'Q':>8} | {'Q err':>7} | {'s/it':>5}")


def log(it, J, Jd, Jr, gnorm, control, t_it):
    Q = flow_rate(control)
    history.append(dict(it=it, J=float(J), J_data=float(Jd), J_reg=float(Jr),
                        grad_norm=float(gnorm), Q=float(Q),
                        Q_rel_err=float(abs(Q - Q_true) / abs(Q_true)), time_s=float(t_it)))
    h = history[-1]
    print(f"{it:4d} | {h['J']:12.6e} | {h['J_data']:10.3e} | {h['J_reg']:10.3e} | "
          f"{gnorm:9.2e} | {Q:8.4f} | {100 * h['Q_rel_err']:6.2f}% | {t_it:5.1f}", flush=True)


# =============================================================================
# 5a) OPTIMISER: L-BFGS-B (default)
# =============================================================================
def run_lbfgs():
    J, J_data, J_reg, _, _ = objective(g)                # tape once; minimize() replays it
    rf = ReducedFunctional(J, Control(g))
    lo, hi = Function(V), Function(V)                   # edge clamp as bounds
    lov, hiv = np.full(V.dim(), -1e3), np.full(V.dim(), 1e3)
    lov[edge] = hiv[edge] = 0.0
    lo.vector().set_local(lov); lo.vector().apply("insert")
    hi.vector().set_local(hiv); hi.vector().apply("insert")

    t_last = [time.time()]
    probe = Function(V)

    def callback(xk):
        # scipy hands us the accepted iterate; re-evaluate it to log the terms
        probe.vector().set_local(xk)
        probe.vector().apply("insert")
        Jv = rf(probe)
        now = time.time()
        log(len(history) + 1, Jv, J_data.block_variable.checkpoint,
            J_reg.block_variable.checkpoint, float("nan"), probe, now - t_last[0])
        t_last[0] = now

    print(HEADER)
    g_opt = minimize(rf, method="L-BFGS-B", bounds=([lo], [hi]), callback=callback,
                     options={"maxiter": ITERS, "maxfun": 3 * ITERS, "maxcor": 20,
                              # the gradient is a mesh-weighted dual vector (entries
                              # ~1e-6), so scipy's default gtol=1e-5 stops far too early
                              "gtol": 1e-14, "ftol": 1e-16, "disp": False})
    g.assign(g_opt)


# =============================================================================
# 5b) OPTIMISER: Adam (the original loop, for comparison)
# =============================================================================
def run_adam():
    m = np.zeros(V.dim())
    v = np.zeros(V.dim())
    step = max(1, ITERS // adam_decay_times)
    print(HEADER)
    for it in range(1, ITERS + 1):
        t0 = time.time()
        get_working_tape().clear_tape()
        J, J_data, J_reg, _, _ = objective(g)
        dJ = ReducedFunctional(J, Control(g)).derivative().vector().get_local()
        lr = adam_lr * adam_decay_gamma ** sum(it >= step * k for k in range(1, adam_decay_times + 1))
        m = adam_b1 * m + (1 - adam_b1) * dJ
        v = adam_b2 * v + (1 - adam_b2) * dJ ** 2
        gv = g.vector().get_local() - lr * (m / (1 - adam_b1 ** it)) / (np.sqrt(v / (1 - adam_b2 ** it)) + adam_eps)
        gv[edge] = 0.0
        # logged J is at the control *before* this update (as in the original script)
        log(it, J, J_data, J_reg, np.linalg.norm(dJ), g, time.time() - t0)
        g.vector().set_local(gv)
        g.vector().apply("insert")


t_opt = time.time()
run_lbfgs() if OPTIMIZER == "lbfgs" else run_adam()
t_opt = time.time() - t_opt

# =============================================================================
# 6) FINAL STATE FOR THE OPTIMISED CONTROL + SAVE
# =============================================================================
get_working_tape().clear_tape()
pause_annotation()                   # nothing below needs a gradient
J, J_data, J_reg, w, c = objective(g)
u_opt, p_opt = w.split(deepcopy=True)
u_opt.rename("velocity", "velocity")
p_opt.rename("pressure", "pressure")
c.rename("C", "C")
Q_opt = flow_rate(g)
rel_u = np.sqrt(assemble(inner(u_opt - u_gt, u_opt - u_gt) * dx) / assemble(inner(u_gt, u_gt) * dx))


def save(fn, name, stem):
    """HDF5 (for reloading) + PVD (for ParaView), each to its own file."""
    with HDF5File(mesh.mpi_comm(), os.path.join(OUT_DIR, stem + ".h5"), "w") as f:
        f.write(fn, name)
    File(os.path.join(OUT_DIR, stem + ".pvd")) << fn


save(g, "InletControl", "inlet_control")
save(u_opt, "velocity", "velocity")
save(p_opt, "pressure", "pressure")
save(c, "C", "concentration")
json.dump(dict(history=history, Q_true=Q_true, Q_opt=Q_opt, optimizer=OPTIMIZER,
               final_idx=FINAL_IDX, gamma=gamma_reg), open(os.path.join(OUT_DIR, "history.json"), "w"),
          indent=1)
np.save(os.path.join(OUT_DIR, "history.npy"),
        np.array([[h[k] for k in ("it", "J", "J_data", "J_reg", "grad_norm", "Q", "Q_rel_err", "time_s")]
                  for h in history]))

summary = [
    f"optimizer            : {OPTIMIZER} ({len(history)} iterations)",
    f"final J              : {float(J):.6e}   (data {float(J_data):.3e}, reg {float(J_reg):.3e})",
    f"Q_true               : {Q_true:.6f}",
    f"Q_opt                : {Q_opt:.6f}",
    f"flow-rate rel. error : {abs(Q_opt - Q_true) / abs(Q_true):.4e}",
    f"velocity rel. L2 err : {rel_u:.4e}   (whole domain, vs ground truth)",
    f"optimisation time    : {t_opt:.1f} s   ({t_opt / max(1, len(history)):.2f} s/iteration)",
    f"total script time    : {time.time() - t_script:.1f} s",
]
open(os.path.join(OUT_DIR, "summary.txt"), "w").write("\n".join(summary) + "\n")
print("\n" + "=" * 72 + "\n" + "\n".join(summary) + "\n" + "=" * 72)

# =============================================================================
# 7) SUMMARY PLOT: loss history + inlet profile
# =============================================================================
H = np.array([[h["it"], h["J"], h["Q_rel_err"]] for h in history])
ys = np.linspace(0.5, 1.0, 300)
prof = lambda fn: np.array([fn((x_min_geo, y))[0] for y in ys])
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].semilogy(H[:, 0], H[:, 1], color="#2a78d6")
ax[0].set(title="Objective J", xlabel="iteration")
ax[1].plot(H[:, 0], 100 * H[:, 2], color="#2a78d6")
ax[1].set(title="Inlet flow-rate error [%]", xlabel="iteration", ylim=(0, None))
ax[2].plot(prof(u_gt), ys, "--", color="#0b0b0b", label="ground truth")
ax[2].plot(prof(g), ys, color="#2a78d6", label=f"reconstructed ({OPTIMIZER})")
ax[2].set(title="Inlet profile u_x(y)", xlabel="u_x", ylabel="y")
ax[2].legend(frameon=False)
for a in ax:
    a.grid(True, color="#e4e3df")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "summary.png"), dpi=150)
print(f"saved results to {OUT_DIR}")
