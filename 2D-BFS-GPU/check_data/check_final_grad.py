"""
Check the 200-epoch GPU result at its FINAL inlet control (N = 4000, real mask):

  MODE=cpu : original pipeline (FEniCS NS + per-step GMRES/ILU ADE, dolfin-adjoint)
  MODE=gpu : all-GPU pipeline (gpu_ns + gpu_ade blocks)

Each mode saves J_data, J_reg, the NS solution and dJ/du to check_data/out/<mode>_*.npy.
MODE=gpu also evaluates the objective at the TRUE inlet profile (two extensions)
and runs a Taylor test at the final control.

    RUN=<run folder> MODE=cpu|gpu python check_data/check_final_grad.py
"""
import glob
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from fenics import *
from fenics_adjoint import *

set_log_level(LogLevel.WARNING)
MODE = os.environ["MODE"]
RUN = os.environ["RUN"]
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
from bfs_data import find_data_dir
fd = find_data_dir()
x_min_geo, x_max_geo = -0.5, 4.0
inlet = f"near(x[0], {x_min_geo}, DOLFIN_EPS)"
walls = f"on_boundary && !(near(x[0], {x_min_geo}, DOLFIN_EPS) || near(x[0], {x_max_geo}, DOLFIN_EPS))"
outlet = f"near(x[0], {x_max_geo}, DOLFIN_EPS)"
N = 4000

mesh = Mesh()
with XDMFFile(os.path.join(fd, "mesh.xdmf")) as f:
    f.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
S = FunctionSpace(mesh, "CG", 1)
nu = Constant(0.04) / Constant(1.06)
D, dt = Constant(1.0e-3), Constant(1.0e-3)
gamma_reg = Constant(2e-6)
mask = Expression("x[1] >= 0.3 && x[1] <= 0.4 ? 1.0 : 0.0", degree=0)

# target, exactly as in the run script
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
r = vtk.vtkGenericDataObjectReader()
r.SetFileName(os.path.join(fd, "C_4000.vtk"))
r.Update()
o = r.GetOutput()
C_target = Function(S)
C_target.vector().set_local(vtk_to_numpy(o.GetPointData().GetArray("C")).ravel()[
    cKDTree(vtk_to_numpy(o.GetPoints().GetData())[:, :2]).query(S.tabulate_dof_coordinates().reshape(-1, 2))[1]])
C_target.vector().apply("insert")

res_dir = glob.glob(os.path.join(RUN, "Inverse_Results_*ep200"))[0]
u_fin = Function(V, name="InletControl")
with HDF5File(mesh.mpi_comm(), os.path.join(res_dir, "optimized_inlet_control.h5"), "r") as f:
    f.read(u_fin, "InletControl")


def ns_fenics(u_in):
    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    bcs = [DirichletBC(W.sub(0), u_in, inlet), DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls),
           DirichletBC(W.sub(1), Constant(0.0), outlet)]
    solve((inner(dot(grad(u), u), v) + nu * inner(grad(u), grad(v)) - div(v) * p + q * div(u)) * dx == 0,
          w, bcs)
    return w


def ade_fenics(w):   # the original loop, including GMRES + ILU
    c_n = interpolate(Constant(0.0), S)
    c = Function(S)
    phi, ct = TestFunction(S), TrialFunction(S)
    c_mid = 0.5 * (ct + c_n)
    F = (phi * (ct - c_n) / dt * dx + phi * dot(split(w)[0], grad(c_mid)) * dx
         + D * dot(grad(phi), grad(c_mid)) * dx)
    s = LinearVariationalSolver(LinearVariationalProblem(lhs(F), rhs(F), c, DirichletBC(S, Constant(1.0), inlet)))
    s.parameters["linear_solver"] = "gmres"
    s.parameters["preconditioner"] = "ilu"
    for _ in range(N):
        s.solve()
        c_n.assign(c)
    return c


if MODE == "gpu":
    from gpu_ns import GPUSteadyNS, ns_solve_gpu
    from gpu_ade import GPUAdvectionDiffusion, ade_solve_gpu
    op_ns = GPUSteadyNS(W, V, float(nu), inlet, walls, outlet)
    op_ade = GPUAdvectionDiffusion(S, W, float(D), float(dt), inlet)
    ns, ade = (lambda u: ns_solve_gpu(op_ns, u)), (lambda w: ade_solve_gpu(op_ade, w, N))
else:
    ns, ade = ns_fenics, ade_fenics


def evaluate(ctrl, tag, grad_too=True):
    get_working_tape().clear_tape()
    t0 = time.time()
    w = ns(ctrl)
    c = ade(w)
    Jd = assemble(0.5 * mask * (c - C_target) ** 2 * dx)
    Jr = assemble(gamma_reg * inner(grad(ctrl), grad(ctrl)) * dx)
    J = Jd + Jr
    out = dict(J=float(J), J_data=float(Jd), J_reg=float(Jr),
               Q=float(assemble(ctrl[0] * ds(domain=mesh, subdomain_data=_bnd)(1))))
    if grad_too:
        rf = ReducedFunctional(J, Control(ctrl))
        g = rf.derivative().vector().get_local()
        np.save(os.path.join(OUT, f"{MODE}_{tag}_grad.npy"), g)
        out["rf"] = rf
    np.save(os.path.join(OUT, f"{MODE}_{tag}_w.npy"), w.vector().get_local())
    np.save(os.path.join(OUT, f"{MODE}_{tag}_c.npy"), c.vector().get_local())
    print(f"[{MODE}] {tag:<14} J={out['J']:.6e}  J_data={out['J_data']:.6e}  "
          f"J_reg={out['J_reg']:.6e}  Q={out['Q']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return out


class _Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_min_geo, DOLFIN_EPS)


_bnd = MeshFunction("size_t", mesh, 1, 0)
_Inlet().mark(_bnd, 1)

res = evaluate(u_fin, "final")

if MODE == "gpu":
    # the objective at the TRUE inlet profile, with two interior extensions of the control
    u_gt = Function(V)
    with XDMFFile(os.path.join(fd, "velocity_field.xdmf")) as f:
        f.read_checkpoint(u_gt, "velocity", 0)
    with stop_annotating():
        # (a) the ground-truth velocity field itself as the control
        c_a = Function(V)
        c_a.assign(u_gt)
        # (b) min-energy (harmonic) extension of the true inlet profile: smallest possible J_reg
        uh, vh = TrialFunction(V), TestFunction(V)
        c_b = Function(V)
        solve(inner(grad(uh), grad(vh)) * dx == inner(Constant((0, 0)), vh) * dx, c_b,
              DirichletBC(V, u_gt, inlet))
    evaluate(c_a, "true_field", grad_too=False)
    evaluate(c_b, "true_harmonic", grad_too=False)
    h = Function(V)
    with stop_annotating():
        h.vector().set_local(1e-2 * np.random.RandomState(0).rand(V.dim()))
        h.vector().apply("insert")
    get_working_tape().clear_tape()
    res = evaluate(u_fin, "final_taylor")
    print("Taylor rate at final control (GPU):", taylor_test(res["rf"], u_fin, h))
