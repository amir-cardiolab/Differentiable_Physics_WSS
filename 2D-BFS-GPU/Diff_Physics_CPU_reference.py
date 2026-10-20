from __future__ import print_function
import os
import time
import numpy as np

from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt

# =============================================================================
# SCRIPT TIMER
# =============================================================================
script_start_time = time.time()

# =============================================================================
# 1) CONFIG
# =============================================================================
set_log_level(LogLevel.WARNING)
parameters["std_out_all_processes"] = False

forward_dir = "Results_BFS"

epochs = 200

# --- ADAM / LR SCHEDULE (NEW) ---
base_learning_rate = 1.0
lr_decay_times = 8
lr_decay_gamma = 0.5

# --- REG ---
gamma_reg = Constant(2e-6)
final_idx = 4000

# ----------------------- y-OBSERVATION WINDOW (MASK REGIONS) -----------------
# The data-misfit term J_data is only evaluated inside one or two horizontal
# bands in y (see the `mask` Expression in section 4). Everything outside is
# ignored by the inverse problem.
#   Region 1 : y1_min <= y <= y1_max                      (always active)
#   Region 2 : y2_min <= y <= y2_max                      (optional)
#              set y2_min = y2_max = None to disable Region 2
# Limits are in PHYSICAL mesh y. Keep this window consistent with the one used
# by the PINN (PINN.py).
y1_min, y1_max = 0.3, 0.4       # Region 1 [y_min, y_max]
y2_min, y2_max = None, None     # Region 2 [y_min, y_max]  (None = disabled)

progress_every_percent = 10

# Geometry x-limits for BC identification
x_min_geo = -0.5
x_max_geo = 4.0

# =============================================================================
# 1.5) OUTPUT DIRECTORY TAGGING
# =============================================================================
def mask_tag(_y1_min, _y1_max, _y2_min, _y2_max):
    if _y2_min is None or _y2_max is None:
        return f"y[{_y1_min},{_y1_max}]_y[null]"
    return f"y[{_y1_min},{_y1_max}]_y[{_y2_min},{_y2_max}]"

mask_str = mask_tag(y1_min, y1_max, y2_min, y2_max)

inverse_dir = (
    f"Inverse_Results_T{final_idx}_"
    f"{mask_str}_"
    f"adam_lr{base_learning_rate:.1e}_"
    f"decay{lr_decay_gamma:.2f}x{lr_decay_times}_"
    f"gamma{float(gamma_reg):.1e}_"
    f"ep{epochs}"
)
os.makedirs(inverse_dir, exist_ok=True)

# =============================================================================
# 2) MESH & SPACES
# =============================================================================
mesh = Mesh()
with XDMFFile(os.path.join(forward_dir, "mesh.xdmf")) as f:
    f.read(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))
S = FunctionSpace(mesh, "CG", 1)

# Physics
rho, mu = Constant(1.06), Constant(0.04)
nu = mu / rho
D = Constant(1.0e-3)
dt = Constant(1.0e-3)

try:
    print("MPI size:", MPI.comm_world.size)
except Exception:
    pass

# =============================================================================
# 3) LOAD TARGET CONCENTRATION @ STEP + MASK DIAGNOSTICS
# =============================================================================
C_target = Function(S)

# ── Load the forward-step `final_idx` concentration snapshot ──────────────────
# This single snapshot used to be read from the full time-series
# concentration.xdmf/.h5. It was exported once to C_4000.vtk, so the multi-GB
# time-series no longer ships with the data bundle. The VTK stores the CG1 field
# in ParaView's cell-expanded form, so map its point values onto the CG1 DOFs of
# S by nearest-neighbour coordinate match (order/duplicate independent).
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

_c_vtk_path = os.path.join(forward_dir, "C_4000.vtk")
_reader = vtk.vtkGenericDataObjectReader()
_reader.SetFileName(_c_vtk_path)
_reader.Update()
_vtk_obj = _reader.GetOutput()
if _vtk_obj is None or _vtk_obj.GetNumberOfPoints() == 0:
    raise RuntimeError(f"Could not read concentration snapshot: {_c_vtk_path}")
_c_arr = _vtk_obj.GetPointData().GetArray("C")
if _c_arr is None:
    raise RuntimeError(f"Scalar array 'C' not found in {_c_vtk_path}")
_vtk_xy = vtk_to_numpy(_vtk_obj.GetPoints().GetData())[:, :2]
_vtk_C  = vtk_to_numpy(_c_arr).ravel()

_dof_xy = S.tabulate_dof_coordinates().reshape(-1, 2)
_nn = cKDTree(_vtk_xy).query(_dof_xy, k=1)[1]
C_target.vector().set_local(_vtk_C[_nn])
C_target.vector().apply("insert")
print(f"Loaded C_target from {_c_vtk_path} "
      f"({_vtk_xy.shape[0]} VTK pts -> {_dof_xy.shape[0]} CG1 DOFs)")

coords = mesh.coordinates()

if y2_min is None or y2_max is None:
    mask_idx = [i for i, x in enumerate(coords) if y1_min <= x[1] <= y1_max]
else:
    mask_idx = [
        i for i, x in enumerate(coords)
        if (y1_min <= x[1] <= y1_max) or (y2_min <= x[1] <= y2_max)
    ]

c_vals = C_target.vector().get_local()
masked_vals = c_vals[mask_idx] if len(mask_idx) > 0 else np.array([])

print("-" * 85)
print(f"TARGET DIAGNOSTICS @ step {final_idx}")
print(f"  Global max C            : {c_vals.max():.6e}")
print(f"  Global min C            : {c_vals.min():.6e}")
print(f"  Points in mask          : {len(mask_idx)}")
if masked_vals.size > 0:
    print(f"  Mask max C              : {masked_vals.max():.6e}")
    print(f"  Mask min C              : {masked_vals.min():.6e}")
else:
    print("  WARNING: mask contains no mesh vertices.")
print("-" * 85)

# Ground truth velocity
u_gt = Function(V)
with XDMFFile(os.path.join(forward_dir, "velocity_field.xdmf")) as vf:
    vf.read_checkpoint(u_gt, "velocity", 0)

# =============================================================================
# FLOW RATE DIAGNOSTIC SETUP (ONCE)  (FIXED ds)
# =============================================================================
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_min_geo, DOLFIN_EPS)

Inlet().mark(boundaries, 1)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
ds_in = ds(1)

Q_true = assemble(u_gt[0] * ds_in)

print("\n" + "-" * 85)
print(f"[REFERENCE] Ground-truth inlet flow rate Q_true = {Q_true:.6e}")
print("-" * 85 + "\n")

# =============================================================================
# 4) MASK & CONTROL VARIABLE
# =============================================================================
if y2_min is None or y2_max is None:
    mask = Expression(
        "x[1] >= y1min && x[1] <= y1max ? 1.0 : 0.0",
        y1min=y1_min,
        y1max=y1_max,
        degree=0
    )
else:
    mask = Expression(
        "( (x[1] >= y1min && x[1] <= y1max) || "
        "  (x[1] >= y2min && x[1] <= y2max) ) ? 1.0 : 0.0",
        y1min=y1_min,
        y1max=y1_max,
        y2min=y2_min,
        y2max=y2_max,
        degree=0
    )

u_inlet_control = Function(V, name="InletControl")
u_inlet_control.assign(project(Constant((1.0, 0.0)), V))

# =============================================================================
# 4.5) SIMPLE INLET EDGE DOFs (GLOBAL MIN/MAX y ON THE INLET)
# =============================================================================
def inlet_edge_dofs_minmax(V_space, mesh_obj, x_inlet):
    inlet_expr = f"near(x[0], {x_inlet}, DOLFIN_EPS)"
    bc_tmp = DirichletBC(V_space, Constant((0.0, 0.0)), inlet_expr)

    inlet_dofs = np.array(list(bc_tmp.get_boundary_values().keys()), dtype=np.int64)
    if inlet_dofs.size == 0:
        raise RuntimeError("No inlet DOFs found. Check x_min_geo and inlet definition.")

    dof_coords = V_space.tabulate_dof_coordinates().reshape((-1, mesh_obj.geometry().dim()))
    y = dof_coords[inlet_dofs, 1]

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    h = mesh_obj.hmin()
    edge_tol = 0.01 * h

    dofs_low = inlet_dofs[np.abs(y - y_min) <= edge_tol]
    dofs_high = inlet_dofs[np.abs(y - y_max) <= edge_tol]

    edge_dofs = np.unique(np.concatenate([dofs_low, dofs_high]))

    print(
        f"[INLET EDGE CLAMP MINMAX] y∈[{y_min:.6f},{y_max:.6f}] | "
        f"edge_dofs={len(edge_dofs)} | hmin={h:.3e}"
    )

    return edge_dofs

inlet_edge_dofs = inlet_edge_dofs_minmax(V, mesh, x_min_geo)

# Clamp initial control at inlet edges
u0 = u_inlet_control.vector().get_local()
u0[inlet_edge_dofs] = 0.0
u_inlet_control.vector().set_local(u0)
u_inlet_control.vector().apply("insert")

# =============================================================================
# 5) HELPERS
# =============================================================================
def print_ade_progress(step, n_steps, t0, last_pct):
    pct = int(100.0 * float(step + 1) / float(n_steps))
    target = (pct // progress_every_percent) * progress_every_percent
    if target > last_pct and target % progress_every_percent == 0:
        elapsed = time.time() - t0
        print(f"    ADE progress: {target:3d}% | elapsed {elapsed:.1f}s")
        return target
    return last_pct

def inlet_y_range_from_mesh(mesh_obj, x_inlet, tol=1e-10):
    xy = mesh_obj.coordinates()
    ys = xy[np.isclose(xy[:, 0], x_inlet, atol=tol), 1]
    if ys.size == 0:
        return np.nan, np.nan
    return float(ys.min()), float(ys.max())

def safe_profile_samples(func, x_inlet, y_samples):
    out = []
    for y in y_samples:
        try:
            val = func((float(x_inlet), float(y)))
            out.append(np.hypot(val[0], val[1]))
        except RuntimeError:
            out.append(np.nan)
    return np.array(out)

# =============================================================================
# 5.5) LR SCHEDULE (NEW)
# =============================================================================
def lr_for_epoch(epoch):
    step_size = max(1, epochs // lr_decay_times)
    milestones = [step_size * k for k in range(1, lr_decay_times + 1)]
    n_decays = sum(1 for m in milestones if epoch >= m)
    return base_learning_rate * (lr_decay_gamma ** n_decays)

# =============================================================================
# 5.6) ADAM STATE (NEW)
# =============================================================================
beta1 = 0.9
beta2 = 0.999
eps_adam = 1e-8

m_adam = u_inlet_control.vector().copy()
v_adam = u_inlet_control.vector().copy()
m_adam.zero()
v_adam.zero()
adam_t = 0

# =============================================================================
# 6) OPTIMIZATION LOOP
# =============================================================================
loss_history = []

print(
    f"{'Epoch':<6} | {'LR':<10} | {'Loss':<12} | {'GradNorm':<12} | {'Vmax':<9} | "
    f"{'Q_pred':<12} | {'Q_true':<12} | {'RelErr(Q)':<10} | "
    f"{'NS(s)':<7} | {'ADE(s)':<7}"
)
print("-" * 130)

for epoch in range(1, epochs + 1):
    get_working_tape().clear_tape()

    learning_rate = lr_for_epoch(epoch)

    # -------------------------------------------------------------------------
    # A) STEADY NAVIER–STOKES
    # -------------------------------------------------------------------------
    t_ns = time.time()

    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)

    bc_in = DirichletBC(W.sub(0), u_inlet_control, f"near(x[0], {x_min_geo}, DOLFIN_EPS)")
    bc_walls = DirichletBC(
        W.sub(0),
        Constant((0.0, 0.0)),
        f"on_boundary && !(near(x[0], {x_min_geo}, DOLFIN_EPS) || near(x[0], {x_max_geo}, DOLFIN_EPS))"
    )
    bc_out = DirichletBC(W.sub(1), Constant(0.0), f"near(x[0], {x_max_geo}, DOLFIN_EPS)")

    F_ns = (
        inner(dot(grad(u), u), v) * dx
        + nu * inner(grad(u), grad(v)) * dx
        - div(v) * p * dx
        + q * div(u) * dx
    )

    solve(F_ns == 0, w, [bc_in, bc_walls, bc_out])
    u_flow = split(w)[0]
    u_actual = w.split()[0]
    ns_time = time.time() - t_ns

    # -------------------------------------------------------------------------
    # B) ADVECTION–DIFFUSION
    # -------------------------------------------------------------------------
    t_ade = time.time()

    c_n = interpolate(Constant(0.0), S)
    c = Function(S)
    phi = TestFunction(S)
    c_trial = TrialFunction(S)
    c_mid = 0.5 * (c_trial + c_n)

    F_ade = (
        phi * (c_trial - c_n) / dt * dx
        + phi * dot(u_flow, grad(c_mid)) * dx
        + D * dot(grad(phi), grad(c_mid)) * dx
    )

    a_ade, L_ade = lhs(F_ade), rhs(F_ade)
    bc_c = DirichletBC(S, Constant(1.0), f"near(x[0], {x_min_geo}, DOLFIN_EPS)")

    ade_problem = LinearVariationalProblem(a_ade, L_ade, c, bc_c)
    ade_solver = LinearVariationalSolver(ade_problem)

    prm = ade_solver.parameters
    prm["linear_solver"] = "gmres"
    prm["preconditioner"] = "ilu"

    print(f"  Epoch {epoch}: starting ADE time loop (N={final_idx})")
    last_pct = 0
    for n in range(final_idx):
        ade_solver.solve()
        c_n.assign(c)
        last_pct = print_ade_progress(n, final_idx, t_ade, last_pct)

    ade_time = time.time() - t_ade

    # -------------------------------------------------------------------------
    # C) LOSS & GRADIENT
    # -------------------------------------------------------------------------
    J = assemble(
        0.5 * mask * (c - C_target) ** 2 * dx
        + gamma_reg * inner(grad(u_inlet_control), grad(u_inlet_control)) * dx
    )

    rf = ReducedFunctional(J, Control(u_inlet_control))
    grad_u = rf.derivative()
    loss_history.append(float(J))

    # -------------------------------------------------------------------------
    # D) UPDATE (ADAM) + HARD CLAMP at inlet edges
    # -------------------------------------------------------------------------
    adam_t += 1

    g = grad_u.vector().get_local()

    m_old = m_adam.get_local()
    v_old = v_adam.get_local()

    m_new = beta1 * m_old + (1.0 - beta1) * g
    v_new = beta2 * v_old + (1.0 - beta2) * (g ** 2)

    m_hat = m_new / (1.0 - beta1 ** adam_t)
    v_hat = v_new / (1.0 - beta2 ** adam_t)

    u_new = u_inlet_control.vector().get_local()
    u_new = u_new - learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)

    # Hard clamp at inlet endpoints
    u_new[inlet_edge_dofs] = 0.0

    u_inlet_control.vector().set_local(u_new)
    u_inlet_control.vector().apply("insert")

    m_adam.set_local(m_new)
    m_adam.apply("insert")

    v_adam.set_local(v_new)
    v_adam.apply("insert")

    # -------------------------------------------------------------------------
    # E) LOGGING
    # -------------------------------------------------------------------------
    v_max = u_inlet_control.vector().norm("linf")
    g_norm = grad_u.vector().norm("l2")

    Q_pred = assemble(u_inlet_control[0] * ds_in)
    rel_err_Q = abs(Q_pred - Q_true) / (abs(Q_true) + 1e-14)

    print(
        f"{epoch:<6} | {learning_rate:<10.3e} | {float(J):.6e} | {g_norm:.3e} | {v_max:.4f} | "
        f"{Q_pred:.6e} | {Q_true:.6e} | {rel_err_Q:.2e} | "
        f"{ns_time:.1f} | {ade_time:.1f}"
    )

# =============================================================================
# 6.5) FINAL FLOW RATE ERROR (PRINT + SAVE)
# =============================================================================
Q_pred_final = assemble(u_inlet_control[0] * ds_in)
rel_l2_err_Q = abs(Q_pred_final - Q_true) / (abs(Q_true) + 1e-14)

print("\n" + "-" * 85)
print("[FINAL FLOW RATE ERROR]")
print(f"  Q_pred_final : {Q_pred_final:.6e}")
print(f"  Q_true       : {Q_true:.6e}")
print(f"  Rel L2 Error : {rel_l2_err_Q:.6e}")
print("-" * 85)

flow_err_file = os.path.join(inverse_dir, "final_flow_rate_error.txt")
with open(flow_err_file, "w") as f:
    f.write("FINAL FLOW RATE ERROR\n")
    f.write(f"Q_pred_final : {Q_pred_final:.6e}\n")
    f.write(f"Q_true       : {Q_true:.6e}\n")
    f.write(f"Rel_L2_Error : {rel_l2_err_Q:.6e}\n")

print(f"Flow-rate error saved to: {flow_err_file}")

# =============================================================================
# 6.6) SAVE LOSS HISTORY (.NPY)
# =============================================================================
loss_npy_path = os.path.join(inverse_dir, "loss_history.npy")
np.save(loss_npy_path, np.array(loss_history))
print(f"Loss history saved to: {loss_npy_path}")

# =============================================================================
# 7) SAVE RESULTS
# =============================================================================
u_copy = project(u_actual, V)

with XDMFFile(os.path.join(inverse_dir, "optimized_final_state.xdmf")) as xf:
    xf.write_checkpoint(c, "C", 0, XDMFFile.Encoding.HDF5, False)

with XDMFFile(os.path.join(inverse_dir, "optimized_inlet_control.xdmf")) as xf:
    xf.write_checkpoint(u_inlet_control, "InletControl", 0, XDMFFile.Encoding.HDF5, False)

File(os.path.join(inverse_dir, "optimized_inlet_control.xml")) << u_inlet_control

with HDF5File(mesh.mpi_comm(), os.path.join(inverse_dir, "optimized_inlet_control.h5"), "w") as h5f:
    h5f.write(u_inlet_control, "InletControl")

with XDMFFile(os.path.join(inverse_dir, "optimized_velocity_field.xdmf")) as xf:
    xf.write_checkpoint(u_copy, "velocity", 0, XDMFFile.Encoding.HDF5, False)

File(os.path.join(inverse_dir, "optimized_velocity_field.xml")) << u_copy

vtk_file = File(os.path.join(inverse_dir, "optimized_velocity_field.pvd"))
vtk_file << u_copy

with HDF5File(mesh.mpi_comm(), os.path.join(inverse_dir, "optimized_velocity_field.h5"), "w") as h5f:
    h5f.write(u_copy, "velocity")

print(f"Results saved to {inverse_dir} in Checkpoint, XML, VTK, and HDF5 formats.")
print("  - Velocity field saved as: optimized_velocity_field.xdmf/.h5/.xml/.pvd")

# =============================================================================
# 8) SAVE PLOTS ONLY (NO SHOW)
# =============================================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(range(1, epochs + 1), loss_history, "r-o")
plt.title(f"Loss at Step {final_idx}")
plt.xlabel("Epoch")
plt.ylabel("J")
plt.grid(True, which="both")

plt.subplot(1, 2, 2)
y0, y1 = inlet_y_range_from_mesh(mesh, x_min_geo)
y_vals = np.linspace(y0, y1, 200) if np.isfinite(y0) and np.isfinite(y1) else np.linspace(0.0, 1.0, 200)

u_opt = safe_profile_samples(u_inlet_control, x_min_geo, y_vals)
u_true = safe_profile_samples(u_gt, x_min_geo, y_vals)

valid = np.isfinite(u_opt) & np.isfinite(u_true)
plt.plot(u_true[valid], y_vals[valid], "k--", label="Ground Truth")
plt.plot(u_opt[valid], y_vals[valid], "b-", label="Optimized")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(inverse_dir, "optimization_summary.png"), dpi=200)
plt.close("all")

# =============================================================================
# 9) TOTAL TIME
# =============================================================================
total_time = time.time() - script_start_time
h = int(total_time // 3600)
m = int((total_time % 3600) // 60)
s = total_time % 60

print("\n" + "=" * 85)
print(f"TOTAL SCRIPT RUNTIME: {h:02d}h {m:02d}m {s:06.2f}s")
print("=" * 85)
