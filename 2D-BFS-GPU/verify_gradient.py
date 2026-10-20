"""
Correctness check for the GPU blocks (gpu_ns.py, gpu_ade.py).

Builds the same J(u_inlet) for a small number of time steps N with
  fenics  : original FEniCS NS + per-step ADE loop, all taped by pyadjoint (LU solves)
  gpu_ade : FEniCS NS (taped)      + GPU ADE block
  gpu_all : GPU NS block           + GPU ADE block
and compares J, c^N and dJ/du_inlet against `fenics`, then runs a pyadjoint
Taylor test on `gpu_all` (expected convergence rate 2).

    N=20 python verify_gradient.py
"""
from __future__ import print_function
import os
import time
import numpy as np
from fenics import *
from fenics_adjoint import *
from gpu_ade import GPUAdvectionDiffusion, ade_solve_gpu
from gpu_ns import GPUSteadyNS, ns_solve_gpu

set_log_level(LogLevel.WARNING)
N = int(os.environ.get("N", 20))
device = os.environ.get("ADE_DEVICE", "cuda")
from bfs_data import find_data_dir
forward_dir = find_data_dir()
x_min_geo, x_max_geo = -0.5, 4.0
inlet = f"near(x[0], {x_min_geo}, DOLFIN_EPS)"
walls = f"on_boundary && !(near(x[0], {x_min_geo}, DOLFIN_EPS) || near(x[0], {x_max_geo}, DOLFIN_EPS))"
outlet = f"near(x[0], {x_max_geo}, DOLFIN_EPS)"

mesh = Mesh()
with XDMFFile(os.path.join(forward_dir, "mesh.xdmf")) as f:
    f.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
S = FunctionSpace(mesh, "CG", 1)
rho, mu = Constant(1.06), Constant(0.04)
nu = mu / rho
D, dt = Constant(1.0e-3), Constant(1.0e-3)
# full-domain mask by default: with small N the dye has not reached the
# y in [0.3, 0.4] band yet, which would make J and dJ/du trivially ~0
mask = (Constant(1.0) if os.environ.get("MASK", "full") == "full" else
        Expression("x[1] >= 0.3 && x[1] <= 0.4 ? 1.0 : 0.0", degree=0))

u_ctrl = Function(V, name="InletControl")
u_ctrl.assign(project(Constant((1.0, 0.0)), V))


def ns_solve(u_in):
    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    bcs = [DirichletBC(W.sub(0), u_in, inlet),
           DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls),
           DirichletBC(W.sub(1), Constant(0.0), outlet)]
    F = (inner(dot(grad(u), u), v) * dx + nu * inner(grad(u), grad(v)) * dx
         - div(v) * p * dx + q * div(u) * dx)
    solve(F == 0, w, bcs)
    return w


def ade_fenics(w):
    u_flow = split(w)[0]
    c_n = interpolate(Constant(0.0), S)
    c = Function(S)
    phi, ct = TestFunction(S), TrialFunction(S)
    c_mid = 0.5 * (ct + c_n)
    F = (phi * (ct - c_n) / dt * dx + phi * dot(u_flow, grad(c_mid)) * dx
         + D * dot(grad(phi), grad(c_mid)) * dx)
    prob = LinearVariationalProblem(lhs(F), rhs(F), c, DirichletBC(S, Constant(1.0), inlet))
    solver = LinearVariationalSolver(prob)
    solver.parameters["linear_solver"] = "lu"   # exact solves for the reference
    for _ in range(N):
        solver.solve()
        c_n.assign(c)
    return c


# target from a parabolic inlet (not annotated)
with stop_annotating():
    u_par = project(Expression(("6.0*x[1]*(1.0-x[1])", "0.0"), degree=2), V)
    C_target = ade_fenics(ns_solve(u_par))

op_ade = GPUAdvectionDiffusion(S, W, D=float(D), dt=float(dt), inlet=inlet, device=device)
op_ns = GPUSteadyNS(W, V, nu=float(nu), inlet=inlet, walls=walls, outlet=outlet, device=device)
gamma_reg = Constant(2e-6)


def build_J(backend):
    get_working_tape().clear_tape()
    t0 = time.time()
    w = ns_solve_gpu(op_ns, u_ctrl) if backend == "gpu_all" else ns_solve(u_ctrl)
    t_ns = time.time() - t0
    t0 = time.time()
    c = ade_fenics(w) if backend == "fenics" else ade_solve_gpu(op_ade, w, N)
    t_fwd = time.time() - t0
    J = assemble(0.5 * mask * (c - C_target) ** 2 * dx
                 + gamma_reg * inner(grad(u_ctrl), grad(u_ctrl)) * dx)
    rf = ReducedFunctional(J, Control(u_ctrl))
    t0 = time.time()
    g = rf.derivative().vector().get_local()
    t_adj = time.time() - t0
    return dict(J=float(J), c=c.vector().get_local().copy(), w=w.vector().get_local().copy(),
                g=g, rf=rf, t_ns=t_ns, t_fwd=t_fwd, t_adj=t_adj)


rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
res = {b: build_J(b) for b in ("fenics", "gpu_ade", "gpu_all")}
ref = res["fenics"]

print("=" * 86)
print(f"N = {N} steps | device = {op_ade.device} | |dJ/du_ref| = {np.linalg.norm(ref['g']):.3e}")
print(f"{'backend':<9} | {'J rel':>8} | {'w rel':>8} | {'c^N rel':>8} | {'dJ/du rel':>9} | "
      f"{'NS s':>6} | {'ADE s':>6} | {'grad s':>6}")
for b, r in res.items():
    print(f"{b:<9} | {abs(r['J']-ref['J'])/abs(ref['J']):8.1e} | {rel(r['w'], ref['w']):8.1e} | "
          f"{rel(r['c'], ref['c']):8.1e} | {rel(r['g'], ref['g']):9.1e} | "
          f"{r['t_ns']:6.2f} | {r['t_fwd']:6.2f} | {r['t_adj']:6.2f}")
print("gpu_ns  stats:", op_ns.last_stats)
print("gpu_ade stats:", op_ade.last_stats)
print("=" * 86)

h = Function(V)
with stop_annotating():
    h.vector().set_local(1e-2 * np.random.RandomState(0).rand(V.dim()))
    h.vector().apply("insert")
rate = taylor_test(res["gpu_all"]["rf"], u_ctrl, h)
print(f"Taylor test convergence rate (all-GPU path): {rate:.3f}  (expect ~2)")
ok = (max(rel(res[b]["g"], ref["g"]) for b in ("gpu_ade", "gpu_all")) < 1e-6 and rate > 1.9)
print("VERIFY:", "PASS" if ok else "FAIL")
