"""GPU version of 2D-BFS/Diff_Physics.py, written with DOLFINx 0.10 and cuDOLFINx (patched,
see cudolfinx/) and its adjoint module cudolfinx.adjoint (pyadjoint tape, GPU assembly and
NVIDIA cuDSS solves).

Same configuration as the CPU script:
    steady Navier-Stokes (Taylor-Hood P2/P1, Newton), inlet velocity g as the control
    4000 Crank-Nicolson advection-diffusion steps (P1), dt = 1e-3, D = 1e-3
    J = 1/2 int mask (c_N - C_target)^2 dx + gamma int |grad g|^2 dx,   mask: y in [0.3, 0.4]
    200 Adam epochs, lr 1.0 halved 8 times, inlet corner dofs clamped to 0

    python3 Diff_Physics_GPU.py [--epochs 200] [--steps 4000] [--data DIR] [--out DIR]
"""
import argparse, json, os, sys, time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl
from dolfinx import fem, io, mesh
from scipy.spatial import cKDTree
from cudolfinx import adjoint as ad

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--data", default=None, help="folder with mesh.xdmf, velocity_field.h5, C_4000.vtk")
p.add_argument("--out", default=None)
p.add_argument("--ns-ksp", default="cudss", choices=["cudss", "mumps"], help="linear solver inside Newton")
p.add_argument("--no-fused", action="store_true", help="record the ADE loop step by step instead of as one GPU block")
p.add_argument("--taylor", action="store_true", help="Taylor test of the gradient in epoch 1")
args = p.parse_args()
comm = MPI.COMM_WORLD
T0 = time.perf_counter()


def find_data(explicit):
    files = ("mesh.xdmf", "velocity_field.h5", "C_4000.vtk")
    cands = [explicit] if explicit else []
    if os.environ.get("BFS_DATA"):
        cands.append(os.environ["BFS_DATA"])
    cands += [os.path.join(HERE, "Results_BFS"), os.path.join(HERE, "..", "2D-BFS", "Results_BFS")]
    for c in cands:
        if c and all(os.path.exists(os.path.join(c, f)) for f in files):
            return os.path.abspath(c)
    sys.exit("BFS data not found: put Results_BFS/ (from the data bundle) in 2D-BFS-GPU/ or 2D-BFS/, "
             "or pass --data / set BFS_DATA")


DATA = find_data(args.data)

# ----------------------------------------------------------------------------- configuration
epochs = args.epochs
base_learning_rate, lr_decay_times, lr_decay_gamma = 1.0, 8, 0.5
GAMMA_REG = 2e-6
final_idx = args.steps
y1_min, y1_max = 0.3, 0.4
x_min_geo, x_max_geo = -0.5, 4.0
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
out_dir = args.out or os.path.join(HERE, f"Inverse_Results_GPU_T{final_idx}_y[{y1_min},{y1_max}]_adam_lr{base_learning_rate:.1e}_"
                                         f"decay{lr_decay_gamma:.2f}x{lr_decay_times}_gamma{GAMMA_REG:.1e}_ep{epochs}")
os.makedirs(out_dir, exist_ok=True)


def lr_for_epoch(epoch):
    step_size = max(1, epochs // lr_decay_times)
    milestones = [step_size * k for k in range(1, lr_decay_times + 1)]
    return base_learning_rate * (lr_decay_gamma ** sum(1 for m in milestones if epoch >= m))


# ----------------------------------------------------------------------------- mesh and spaces
with io.XDMFFile(comm, os.path.join(DATA, "mesh.xdmf"), "r") as xf:
    msh = xf.read_mesh(name="mesh")
gdim, tdim = msh.geometry.dim, msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
W = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,)),
                                                    basix.ufl.element("Lagrange", msh.basix_cell(), 1)]))
S = fem.functionspace(msh, ("Lagrange", 1))
V0, _ = W.sub(0).collapse()
DG0 = fem.functionspace(msh, ("DG", 0))
nu = fem.Constant(msh, PETSc.ScalarType(0.04 / 1.06))
D = fem.Constant(msh, PETSc.ScalarType(1.0e-3))
dt = fem.Constant(msh, PETSc.ScalarType(1.0e-3))
gamma_reg = fem.Constant(msh, PETSc.ScalarType(GAMMA_REG))


# ----------------------------------------------------------------------------- data
def read_vtk_points_scalar(path, name="C"):
    """Points and point scalar `name` of a legacy VTK file (ASCII or binary)."""
    data = open(path, "rb").read()
    binary = b"\nBINARY" in data[:200]

    def after(tag, start=0):
        i = data.index(tag, start)
        return i, data.index(b"\n", i) + 1

    i, p0 = after(b"POINTS ")
    header = data[i:p0].split()
    npts, dtp = int(header[1]), {"float": ">f4", "double": ">f8"}[header[2].decode()]
    pts = (np.frombuffer(data, dtype=dtp, count=3 * npts, offset=p0).reshape(npts, 3).astype(np.float64) if binary
           else np.array(data[p0:].split()[:3 * npts], dtype=np.float64).reshape(npts, 3))
    i, _ = after(b"POINT_DATA")
    i, _ = after(("SCALARS %s" % name).encode(), i)
    dtv = {"float": ">f4", "double": ">f8"}[data[i:data.index(b"\n", i)].split()[2].decode()]
    i, p1 = after(b"LOOKUP_TABLE", i)
    vals = (np.frombuffer(data, dtype=dtv, count=npts, offset=p1).astype(np.float64) if binary
            else np.array(data[p1:].split()[:npts], dtype=np.float64))
    return pts[:, :2], vals


def read_legacy_velocity(path):
    """Ground-truth velocity from the FEniCS 2019 checkpoint velocity_field.h5 onto V0, matched by
    dof coordinates (a 2-component CG2 field: 12 dofs per triangle, component-major, vertices
    then edge midpoints opposite vertices 0, 1, 2)."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f["velocity/velocity_0"]
        vec, cd, xcd = (np.array(g[k]).ravel() for k in ("vector", "cell_dofs", "x_cell_dofs"))
        top, geo = np.array(g["mesh/topology"]), np.array(g["mesh/geometry"])
    coords, comp = np.zeros((vec.size, 2)), np.zeros(vec.size, dtype=int)
    v = geo[top]                                                    # (cells, 3, 2)
    pts = np.stack([v[:, 0], v[:, 1], v[:, 2], 0.5 * (v[:, 1] + v[:, 2]), 0.5 * (v[:, 0] + v[:, 2]),
                    0.5 * (v[:, 0] + v[:, 1])], axis=1)              # (cells, 6, 2)
    dofs = np.stack([cd[xcd[c]:xcd[c + 1]] for c in range(top.shape[0])])   # (cells, 12)
    for k in range(2):
        coords[dofs[:, 6 * k:6 * k + 6].ravel()] = pts.reshape(-1, 2)
        comp[dofs[:, 6 * k:6 * k + 6].ravel()] = k
    u = fem.Function(V0)
    arr = u.x.array.reshape(-1, 2)
    xy = V0.tabulate_dof_coordinates()[:, :2]
    for k in range(2):
        sel = np.where(comp == k)[0]
        dist, idx = cKDTree(coords[sel]).query(xy, k=1)
        assert dist.max() < 1e-8, "velocity_field.h5 does not match mesh.xdmf"
        arr[:, k] = vec[sel][idx]
    u.x.array[:] = arr.ravel()
    return u


vtk_xy, vtk_C = read_vtk_points_scalar(os.path.join(DATA, "C_4000.vtk"))
C_target = fem.Function(S, name="C_target")
C_target.x.array[:] = vtk_C[cKDTree(vtk_xy).query(S.tabulate_dof_coordinates()[:, :2], k=1)[1]]
mask = fem.Function(DG0, name="mask")                 # Expression(degree=0) of the CPU script: cell midpoints
mask.interpolate(lambda x: np.where((x[1] >= y1_min) & (x[1] <= y1_max), 1.0, 0.0))
u_gt = read_legacy_velocity(os.path.join(DATA, "velocity_field.h5"))

inlet_f = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], x_min_geo))
outlet_f = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], x_max_geo))
walls_f = mesh.locate_entities_boundary(msh, fdim, lambda x: ~(np.isclose(x[0], x_min_geo) | np.isclose(x[0], x_max_geo)))
ds_in = ufl.Measure("ds", domain=msh, subdomain_data=mesh.meshtags(msh, fdim, inlet_f, np.full(len(inlet_f), 1, dtype=np.int32)))(1)


def flow_rate(u):
    return comm.allreduce(fem.assemble_scalar(fem.form(u[0] * ds_in)), op=MPI.SUM)


Q_true = flow_rate(u_gt)
inlet_dofs = fem.locate_dofs_topological((W.sub(0), V0), fdim, inlet_f)
walls_dofs = fem.locate_dofs_topological((W.sub(0), V0), fdim, walls_f)
outlet_dofs = fem.locate_dofs_topological(W.sub(1), fdim, outlet_f)

# ----------------------------------------------------------------------------- control
g = ad.Function(V0, name="InletControl")
g.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
inlet_dofs_V0 = np.asarray(inlet_dofs[1])
yv = V0.tabulate_dof_coordinates()[inlet_dofs_V0 // gdim, 1]
hmin = comm.allreduce(msh.h(tdim, np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)).min(), op=MPI.MIN)
edge = inlet_dofs_V0[(np.abs(yv - yv.min()) <= 0.01 * hmin) | (np.abs(yv - yv.max()) <= 0.01 * hmin)]
g.x.array[edge] = 0.0

bc_in = fem.dirichletbc(g, inlet_dofs, W.sub(0))
bc_walls = fem.dirichletbc(fem.Function(V0), walls_dofs, W.sub(0))
bc_out = fem.dirichletbc(PETSc.ScalarType(0.0), outlet_dofs, W.sub(1))
bc_c = fem.dirichletbc(PETSc.ScalarType(1.0), fem.locate_dofs_topological(S, fdim, inlet_f), S)
NS_KSP = ({"ksp_type": "preonly", "pc_type": "cudss"} if args.ns_ksp == "cudss"
          else {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
ADE_KSP = {"ksp_type": "preonly", "pc_type": "cudss"}

# ----------------------------------------------------------------------------- forms and problems (built once)
w = ad.Function(W, name="w")
u, pr = ufl.split(w)
v, q = ufl.TestFunctions(W)
F_ns = (ufl.inner(ufl.dot(ufl.grad(u), u), v) + nu * ufl.inner(ufl.grad(u), ufl.grad(v))
        - ufl.div(v) * pr + q * ufl.div(u)) * ufl.dx
ns_problems = {relax: ad.NonlinearProblem(F_ns, w, [bc_in, bc_walls, bc_out], petsc_options=NS_KSP, adjoint_petsc_options=NS_KSP,
                                          newton_options={"rtol": 1e-10, "atol": 1e-10, "relaxation": relax, "max_it": maxit})
               for relax, maxit in ((1.0, 50), (0.5, 100), (0.25, 200))}
c_n, c = ad.Function(S, name="c_n"), ad.Function(S, name="c")
phi, ct = ufl.TestFunction(S), ufl.TrialFunction(S)
cm = 0.5 * (ct + c_n)
F_ade = (phi * (ct - c_n) / dt + phi * ufl.dot(ufl.split(w)[0], ufl.grad(cm)) + D * ufl.dot(ufl.grad(phi), ufl.grad(cm))) * ufl.dx
ade = ad.LinearProblem(ufl.lhs(F_ade), ufl.rhs(F_ade), c, [bc_c], petsc_options=ADE_KSP, adjoint_petsc_options=ADE_KSP,
                       constant_matrix=True)
J_data_form = 0.5 * mask * (c - C_target) ** 2 * ufl.dx
J_reg_form = gamma_reg * ufl.inner(ufl.grad(g), ufl.grad(g)) * ufl.dx
zero_S = np.zeros_like(c.x.array)
w_prev = np.zeros_like(w.x.array)

m_adam, v_adam, adam_t = np.zeros(g.x.array.size), np.zeros(g.x.array.size), 0
history = []
print(f"data {DATA} | cells {msh.topology.index_map(tdim).size_global} | N = {final_idx} | epochs {epochs} | Q_true = {Q_true:.6f}")
print(f"{'Epoch':<6} | {'LR':<10} | {'Loss':<12} | {'GradNorm':<10} | {'Q_pred':<10} | {'RelErr(Q)':<9} | {'NS(s)':<6} | {'ADE(s)':<6} | {'grad(s)':<6}")

# ----------------------------------------------------------------------------- optimisation loop
for epoch in range(1, epochs + 1):
    t_ep = time.perf_counter()
    ad.get_working_tape().clear_tape()
    learning_rate = lr_for_epoch(epoch)

    # steady Navier-Stokes: Newton from the previous epoch's flow, damped retries if needed
    t_ns = time.perf_counter()
    for relax in (1.0, 0.5, 0.25):
        w.x.array[:] = w_prev
        try:
            ns = ns_problems[relax]
            ns.solve()
            break
        except RuntimeError:
            ad.get_working_tape().clear_tape()
    else:
        raise RuntimeError("Navier-Stokes did not converge")
    w_prev[:] = w.x.array
    t_ns = time.perf_counter() - t_ns

    # advection-diffusion from c = 0
    t_ade = time.perf_counter()
    ad.assign(c_n, zero_S)
    if args.no_fused:
        for n in range(final_idx):
            ade.solve()
            ad.assign(c_n, c)
    else:
        ad.linear_time_loop(ade, c_n, final_idx)      # one tape block, states on the GPU
    t_ade = time.perf_counter() - t_ade

    # loss and gradient
    t_g = time.perf_counter()
    J_data, J_reg = ad.assemble_scalar(J_data_form), ad.assemble_scalar(J_reg_form)
    J = J_data + J_reg
    rf = ad.ReducedFunctional(J, ad.Control(g))
    grad = rf.derivative().x.array.copy()
    t_g = time.perf_counter() - t_g
    if args.taylor and epoch == 1:
        h = ad.Function(V0); h.x.array[:] = 0.05 * np.random.RandomState(0).rand(h.x.array.size); h.x.array[edge] = 0.0
        print("Taylor test convergence rate: %.3f" % ad.taylor_test(rf, g, h), flush=True)
        w.x.array[:] = w_prev

    # Adam update + inlet corner clamp
    adam_t += 1
    m_adam = beta1 * m_adam + (1.0 - beta1) * grad
    v_adam = beta2 * v_adam + (1.0 - beta2) * grad ** 2
    g.x.array[:] -= learning_rate * (m_adam / (1.0 - beta1 ** adam_t)) / (np.sqrt(v_adam / (1.0 - beta2 ** adam_t)) + eps_adam)
    g.x.array[edge] = 0.0

    Q_pred = flow_rate(g)
    rel_err_Q = abs(Q_pred - Q_true) / abs(Q_true)
    history.append(dict(it=epoch, J=float(J), J_data=float(J_data), J_reg=float(J_reg), grad_norm=float(np.linalg.norm(grad)),
                        Q=Q_pred, Q_rel_err=rel_err_Q, lr=learning_rate, time_s=time.perf_counter() - t_ep,
                        t_ns=t_ns, t_ade=t_ade, t_grad=t_g))
    print(f"{epoch:<6} | {learning_rate:<10.3e} | {float(J):.6e} | {np.linalg.norm(grad):.3e} | {Q_pred:.6f} | {rel_err_Q:.2e} | "
          f"{t_ns:<6.1f} | {t_ade:<6.1f} | {t_g:<6.1f}", flush=True)
    if epoch % 10 == 0 or epoch == epochs:
        json.dump(dict(history=history, Q_true=Q_true, final_idx=final_idx, gamma=GAMMA_REG, epochs=epochs),
                  open(os.path.join(out_dir, "history.json"), "w"), indent=1)

# ----------------------------------------------------------------------------- results
u_final = fem.Function(V0, name="velocity")
u_final.interpolate(w.sub(0))
Q_pred_final = flow_rate(g)
with open(os.path.join(out_dir, "final_flow_rate_error.txt"), "w") as f:
    f.write(f"Q_pred_final : {Q_pred_final:.6e}\nQ_true       : {Q_true:.6e}\n"
            f"Rel_Error    : {abs(Q_pred_final - Q_true) / abs(Q_true):.6e}\n")
np.save(os.path.join(out_dir, "loss_history.npy"), np.array([h["J"] for h in history]))
with io.XDMFFile(comm, os.path.join(out_dir, "optimized_fields.xdmf"), "w") as xf:
    xf.write_mesh(msh)
    c.name = "C"
    xf.write_function(c, 0.0)
    P1v = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    for src, name in ((u_final, "velocity"), (g, "InletControl")):
        f1 = fem.Function(P1v, name=name)
        f1.interpolate(src)
        xf.write_function(f1, 0.0)
tt = time.perf_counter() - T0
print(f"Q_pred_final = {Q_pred_final:.6f} (Q_true = {Q_true:.6f}) | results in {out_dir} | total {tt / 60:.1f} min")
