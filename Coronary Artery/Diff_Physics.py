#!/usr/bin/env python3
# coding: utf-8
"""
Inverse IPCS — Inlet Profile Reconstruction
TWO-OUTLET VERSION  (tags 3 & 4, both p=0)
Compatible with the two-outlet forward script (FEniCS 2019).
KEY FEATURES  (matching the updated single-outlet inverse style):
  - ONE-TIME NS ramp/warm-start (N_SETTLE steps) done BEFORE optimisation.
  - During optimisation: N_NS_PER_EPOCH IPCS steps per epoch (default=1),
    then advection-diffusion, then inlet-control update.
  - MASK: loads Mask.vtk, matches Q-DOFs via KD-tree, restricts J_data to
    the masked region (dx_mask).
  - PULSED concentration BC matching forward exactly
    (period=0.2 s, open in [0,0.05)∪[0.1,0.15) → 1.0, else 0.0).
  - Per-epoch velocity + concentration saves.
  - Inlet profile 3-D surface visualisation (predicted vs ground-truth).
FORWARD SCRIPT COMPATIBILITY:
  - Two outlet tags 3 and 4 (both p=0)
  - Observation files in Forward_CFD/
  - T_ADV=0.2, N_ADV_STEPS=200, ADV_WRITE_EVERY=1  → DT_ADV=1e-3
  - N_RAMP=100, N_POST_RAMP_STEPS=100              → N_SETTLE=200
  - Concentration BC = PULSED (period=0.2 s)
SKIP_NS_INV MODE:
  - True  → load cached inverse ramp checkpoint from OUT_DIR.
             Requires a prior SKIP_NS_INV=False run of the inverse.
  - False → run ramp from scratch; cache saved for future re-use.
LOSS:
  J_data = 0.5 * integral_MASK ( (c_pred - c_obs)^2 ) dx_mask
  J_reg  = 0.5 * GAMMA_REG * integral_{inlet} |u_ctrl|^2 ds
  J      = J_data + J_reg
"""
from __future__ import print_function
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from fenics import *
from fenics_adjoint import *
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"
# ── Monkey-patch fenics_adjoint adjoint solver → iterative ───────────────────
import fenics_adjoint.types.compat as _fa_compat
import dolfin as _dolfin
def _iterative_linalg_solve(A, x, b, *args, **kwargs):
    solver = _dolfin.KrylovSolver(A, "gmres", "ilu")
    solver.parameters["relative_tolerance"]      = 1e-8
    solver.parameters["absolute_tolerance"]      = 1e-10
    solver.parameters["maximum_iterations"]      = 2000
    solver.parameters["error_on_nonconvergence"] = False
    solver.solve(x, b)
_fa_compat.linalg_solve = _iterative_linalg_solve
# ==============================================================================
# MPI
# ==============================================================================
comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)
def log(msg=""):
    if rank == 0:
        print(msg, flush=True)
def banner(title):
    if rank == 0:
        w = 90
        print("\n" + "="*w + f"\n  {title}\n" + "="*w + "\n", flush=True)
def section(title):
    if rank == 0:
        print("\n" + "-"*70 + f"\n  >> {title}\n" + "-"*70, flush=True)
def timer_str(t0):
    return f"{time.time()-t0:.2f}s"
def vec_stats(label, vec, indent="  "):
    if rank == 0:
        v = np.asarray(vec).flatten()
        print(f"{indent}{label:35s}: min={v.min():+.6e}  max={v.max():+.6e}  "
              f"mean={v.mean():+.6e}  std={v.std():.4e}  "
              f"nnz={int((np.abs(v)>1e-14).sum())}/{len(v)}", flush=True)
# ==============================================================================
# MASK VTK LOADER
# ==============================================================================
def load_mask_coords_vtk(vtk_path):
    """
    Read Mask.vtk and return point coordinates + optional scalar array.
    Tries vtk Python bindings first; falls back to a minimal ASCII parser.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(vtk_path)
        reader.Update()
        data   = reader.GetOutput()
        coords = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64)
        pd     = data.GetPointData()
        scalars = (vtk_to_numpy(pd.GetArray(0)).astype(np.float64).ravel()
                   if pd.GetNumberOfArrays() > 0 else None)
        if scalars is not None and len(scalars) != len(coords):
            scalars = None
        return coords, scalars
    except ImportError:
        pass
    # Minimal ASCII fallback
    coords = []; scalars = []; n_pts = 0; state = None
    with open(vtk_path, "r") as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        tok = lines[i].strip().split()
        if not tok:
            i += 1; continue
        kw = tok[0].upper()
        if kw == "POINTS":
            n_pts = int(tok[1]); state = "points"; i += 1; continue
        if kw == "SCALARS":
            state = None; i += 1
            while i < len(lines) and not lines[i].strip(): i += 1
            if i < len(lines) and lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                i += 1
            state = "scalars"; continue
        if state == "points":
            coords.extend(float(v) for v in tok)
            if len(coords) >= n_pts * 3: state = None
        elif state == "scalars":
            scalars.extend(float(v) for v in tok)
            if len(scalars) >= n_pts: state = None
        i += 1
    pts = np.array(coords, dtype=np.float64).reshape(-1, 3)
    scl = np.array(scalars, dtype=np.float64).ravel() if scalars else None
    if scl is not None and len(scl) != len(pts): scl = None
    return pts, scl
# ==============================================================================
# USER SETTINGS
# ==============================================================================
banner("USER SETTINGS — Two-Outlet Inverse (ONE-TIME RAMP, MASKED OBJ, PULSED BC)")
MESH_XML  = "Coronary_Mesh/Volume_mesh.xml"
FACET_XML = "Coronary_Mesh/BCnodeFacets.xml"
MASK_VTK  = "Coronary_Mask.vtk"
INLET_ID   = 1
WALL_ID    = 2
OUTLET_IDS = [3, 4]       # ← TWO outlets, matching forward
MU        = 0.04
RHO       = 1.06
DIFF_COEF = 0.005
U_INIT     = 25          # initial guess for inlet magnitude
U_MAX_TRUE = 53.0          # ground-truth reference (monitoring + visualisation)
DT_NS      = 1e-3          # must match forward DT
# ── NS settle — matches forward N_RAMP=100 + N_POST_RAMP_STEPS=100 ────────────
N_RAMP_STEPS = 100   #100
N_POST_STEPS = 10 #100
N_SETTLE     = N_RAMP_STEPS + N_POST_STEPS   # = 200
N_NS_PER_EPOCH = 10 # 100         # IPCS steps per optimisation epoch
# ── Advection-diffusion — must match forward exactly ─────────────────────────
T_ADV           = 0.05
N_ADV_OBS       = 500
DT_ADV          = T_ADV / N_ADV_OBS   # = 1e-3
ADV_WRITE_EVERY = 500
ADV_PRINT_EVERY = 500
# ── Pulsed BC — identical to forward ConcentrationBC ─────────────────────────
def _conc_bc_value(t_now):
    """
    period = 0.2 s:
      open  when local_time in [0.00, 0.05) or [0.10, 0.15) → 1.0
      closed otherwise                                        → 0.0
    """
    period     = 0.4
    local_time = t_now % period
    if (0.0 <= local_time < 0.1) or (0.2 <= local_time < 0.3):
        return 1.0
    return 0.0
# ── Observation files (forward-CFD solution used as observations) ────────────
OBS_DIR       = "Forward_CFD"
OBS_XDMF      = os.path.join(OBS_DIR, "concentration.xdmf")
OBS_H5        = os.path.join(OBS_DIR, "concentration.h5")
OBS_FUNC_NAME = "concentration"
# ── Auto-detect OBS_TIME_INDEX ────────────────────────────────────────────────
def xdmf_time_values(xdmf_path):
    import re
    with open(xdmf_path, "r") as f:
        xml = f.read()
    return re.findall(r'<Time Value="([^"]+)"', xml)
_obs_times = []
if rank == 0:
    try:
        _obs_times = xdmf_time_values(OBS_XDMF)
        log(f"XDMF snapshots found: {len(_obs_times)}")
        if len(_obs_times) > 0:
            log(f"  first time = {_obs_times[0]} | last time = {_obs_times[-1]}")
    except Exception as e:
        log(f"XDMF header parse warning: {e}")
_obs_n = comm.bcast(len(_obs_times) if rank == 0 else None, root=0)

if _obs_n == (N_ADV_OBS + 1):
    OBS_TIME_INDEX = N_ADV_OBS -1

elif _obs_n == N_ADV_OBS:
    OBS_TIME_INDEX = N_ADV_OBS - 1

else:
    OBS_TIME_INDEX = max(_obs_n -1 ,0)


# OBS_TIME_INDEX = 50

# OBS_TIME_INDEX = 200

log(f"OBS_TIME_INDEX auto-set to {OBS_TIME_INDEX} (based on {_obs_n} snapshots)")
# ── SKIP_NS_INV ───────────────────────────────────────────────────────────────
SKIP_NS_INV = False
# ── Optimisation ──────────────────────────────────────────────────────────────
EPOCHS    = 300
LR        = 2
OPTIMIZER = "adam"           # "gd" or "adam"
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
LR_DECAY       = 0.5
LR_DECAY_STEPS = 6
LR_DECAY_EVERY = max(EPOCHS // LR_DECAY_STEPS, 1)
LR_current     = LR
lr_history     = []
GAMMA_REG = 1e-8
# ── Mask tolerance ────────────────────────────────────────────────────────────
MASK_TOL = 1e-6   # increase (e.g. 1e-4) if mask/mesh were built independently
# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================
BASE_DIR = "Inverse_Results"
os.makedirs(BASE_DIR, exist_ok=True)
RUN_TAG = (
    f"{OPTIMIZER.upper()}"
    f"_LR{LR:.0e}"
    f"_decay{LR_DECAY:.2f}"
    f"_reg{GAMMA_REG:.0e}"
    f"_ep{EPOCHS}"
    f"_Uinit{U_INIT:.0f}"
)
OUT_DIR = os.path.join(BASE_DIR, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
EPOCH_VEL_DIR  = os.path.join(OUT_DIR, "Epoch_Velocity")
EPOCH_CONC_DIR = os.path.join(OUT_DIR, "Epoch_Concentration")
FINAL_VEL_DIR  = os.path.join(OUT_DIR, "Final_Velocity")
FINAL_CONC_DIR = os.path.join(OUT_DIR, "Final_Concentration")
for d in [EPOCH_VEL_DIR, EPOCH_CONC_DIR, FINAL_VEL_DIR, FINAL_CONC_DIR]:
    os.makedirs(d, exist_ok=True)
INV_RAMP_U_XDMF = os.path.join(OUT_DIR, "ramp_settled_u.xdmf")
INV_RAMP_P_XDMF = os.path.join(OUT_DIR, "ramp_settled_p.xdmf")
log(f"RUN_TAG = {RUN_TAG}")
log(f"OUT_DIR = {OUT_DIR}")
# ==============================================================================
# FEniCS SETTINGS
# ==============================================================================
parameters["ghost_mode"]                               = "shared_facet"
parameters["form_compiler"]["optimize"]                = True
parameters["form_compiler"]["quadrature_degree"]       = 2
set_log_level(LogLevel.WARNING)
parameters["linear_algebra_backend"]                   = "PETSc"
parameters["krylov_solver"]["relative_tolerance"]      = 1e-8
parameters["krylov_solver"]["absolute_tolerance"]      = 1e-10
parameters["krylov_solver"]["maximum_iterations"]      = 2000
parameters["krylov_solver"]["monitor_convergence"]     = False
parameters["krylov_solver"]["error_on_nonconvergence"] = False
_sp_gmres  = {"linear_solver": "gmres", "preconditioner": "ilu"}
_sp_cg_amg = {"linear_solver": "cg",    "preconditioner": "hypre_amg"}
_sp_cg_jac = {"linear_solver": "cg",    "preconditioner": "jacobi"}
# ==============================================================================
# MESH
# ==============================================================================
banner("MESH")
mesh   = Mesh(MESH_XML)
facets = MeshFunction("size_t", mesh, FACET_XML)
gdim   = mesh.geometry().dim()
coords_all = mesh.coordinates()
log(f"Cells={mesh.num_cells()}  Vertices={mesh.num_vertices()}  dim={gdim}")
log(f"Facet IDs: {np.unique(facets.array())}")
nrm   = FacetNormal(mesh)
ds    = Measure("ds", domain=mesh, subdomain_data=facets)
ds_in = ds(INLET_ID)
def _outlet_ds():
    """Combined outlet measure for tags 3 and 4."""
    r = ds(OUTLET_IDS[0])
    for oid in OUTLET_IDS[1:]:
        r = r + ds(oid)
    return r
# ==============================================================================
# FUNCTION SPACES
# ==============================================================================
banner("FUNCTION SPACES")
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
u_tr = TrialFunction(V);  v_te = TestFunction(V)
p_tr = TrialFunction(Q);  q_te = TestFunction(Q)
log(f"V.dim={V.dim()}  Q.dim={Q.dim()}")
# ==============================================================================
# LOAD AND PROCESS MASK
# ==============================================================================
banner(f"LOADING SPATIAL MASK  ({MASK_VTK})")
if not os.path.isfile(MASK_VTK):
    raise FileNotFoundError(
        f"Mask file not found: {MASK_VTK}\n"
        f"Place {MASK_VTK} in the working directory."
    )
mask_coords, mask_scalars = load_mask_coords_vtk(MASK_VTK)
n_mask_pts = len(mask_coords)
log(f"  Mask points loaded : {n_mask_pts}")
log(f"  Mask x : min={mask_coords[:,0].min():+.6e}  max={mask_coords[:,0].max():+.6e}")
log(f"  Mask y : min={mask_coords[:,1].min():+.6e}  max={mask_coords[:,1].max():+.6e}")
log(f"  Mask z : min={mask_coords[:,2].min():+.6e}  max={mask_coords[:,2].max():+.6e}")
if mask_scalars is not None:
    log(f"  Mask scalar : min={mask_scalars.min():+.6e}  "
        f"max={mask_scalars.max():+.6e}  mean={mask_scalars.mean():+.6e}")
else:
    log("  Mask scalar : (none found in VTK file)")
# Match Q-DOFs to mask points via KD-tree
section(f"Matching Q-DOFs to mask points  (MASK_TOL={MASK_TOL:.2e})")
from scipy.spatial import cKDTree
Q_coords  = Q.tabulate_dof_coordinates()     # (Q.dim(), gdim)
mask_tree = cKDTree(mask_coords)
dists, _  = mask_tree.query(Q_coords, k=1, workers=-1)
inside    = dists < MASK_TOL
mask_dofs = np.where(inside)[0].astype(np.int64)
log(f"  Q DOFs matched : {len(mask_dofs)} / {Q.dim()}")
log(f"  Distance stats : min={dists.min():.4e}  "
    f"max={dists.max():.4e}  median={np.median(dists):.4e}")
if len(mask_dofs) == 0:
    suggested = float(np.percentile(dists, 1)) * 10
    raise RuntimeError(
        f"No Q-DOFs matched mask with MASK_TOL={MASK_TOL:.2e}.\n"
        f"Smallest distance: {dists.min():.4e}\n"
        f"Suggested MASK_TOL: {suggested:.2e}\n"
        "Increase MASK_TOL and re-run."
    )
masked_coords_dofs = Q_coords[mask_dofs]
# Build masked cell measure: a cell is masked if ALL its vertex DOFs are matched
masked_dof_set = set(mask_dofs.tolist())
cell_marker    = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
for cell in cells(mesh):
    if all(int(v) in masked_dof_set for v in cell.entities(0)):
        cell_marker[cell.index()] = 1
n_masked_cells = int((cell_marker.array() == 1).sum())
log(f"  Cells fully inside mask : {n_masked_cells} / {mesh.num_cells()}")
dx_mask  = Measure("dx", domain=mesh, subdomain_data=cell_marker, subdomain_id=1)
_vol_mask = float(assemble(Constant(1.0) * dx_mask))
log(f"  Volume of masked region : {_vol_mask:.6e}")
if _vol_mask < 1e-30:
    raise RuntimeError(
        f"Masked integration domain has zero volume (MASK_TOL={MASK_TOL:.2e}).\n"
        f"{len(mask_dofs)} DOFs matched but no complete cell has all vertices matched.\n"
        f"Try MASK_TOL={MASK_TOL*10:.2e}."
    )
# ==============================================================================
# INLET GEOMETRY (PCA — identical to forward)
# ==============================================================================
banner("INLET GEOMETRY (PCA)")
inlet_facet_ids = np.where(facets.array() == INLET_ID)[0]
if inlet_facet_ids.size == 0:
    raise RuntimeError("No inlet facets found. Check INLET_ID and facet tags.")
_vids = []
for fid in inlet_facet_ids:
    _vids.extend(Facet(mesh, int(fid)).entities(0))
_vids        = np.unique(np.array(_vids, dtype=np.int64))
coords_in    = coords_all[_vids]
inlet_center = coords_in.mean(axis=0)
X_in         = coords_in - inlet_center
eigvals, eigvecs = np.linalg.eigh(X_in.T @ X_in)
inlet_normal = eigvecs[:, np.argmin(eigvals)]
inlet_normal /= np.linalg.norm(inlet_normal)
if np.dot(inlet_normal, coords_all.mean(axis=0) - inlet_center) < 0:
    inlet_normal *= -1
inlet_radius = float(np.max(np.linalg.norm(coords_in - inlet_center, axis=1)))
inlet_area   = float(assemble(Constant(1.0) * ds_in))
log(f"center={inlet_center}  radius={inlet_radius:.6f}")
log(f"normal={inlet_normal}  area={inlet_area:.6e}")
class ConstantInlet(UserExpression):
    def __init__(self, normal, magnitude, **kwargs):
        super().__init__(**kwargs)
        self.n   = np.array(normal, dtype=np.float64)
        self.mag = float(magnitude)
    def value_shape(self):
        return (gdim,)
    def eval(self, values, x):
        values[:] = self.mag * self.n
class ParabolicInletPCA(UserExpression):
    """Used for the ground-truth reference profile in visualisation."""
    def __init__(self, center, radius, normal, Umax, **kwargs):
        super().__init__(**kwargs)
        self.c    = np.array(center, dtype=np.float64)
        self.R    = float(radius)
        self.n    = np.array(normal, dtype=np.float64)
        self.Umax = float(Umax)
    def value_shape(self):
        return (gdim,)
    def eval(self, values, x):
        d    = np.array(x, dtype=np.float64) - self.c
        rvec = d - np.dot(d, self.n) * self.n
        r    = np.linalg.norm(rvec)
        if r <= self.R:
            values[:] = self.Umax * (1.0 - (r / self.R) ** 2) * self.n
        else:
            values[:] = 0.0
# ==============================================================================
# INLET CONTROL  (initial guess = constant profile at U_INIT)
# ==============================================================================
banner(f"INLET CONTROL  (U_INIT={U_INIT})")
u_inlet_ctrl = Function(V, name="InletControl")
_init_expr   = ConstantInlet(inlet_normal, U_INIT, degree=1)
_Ap = assemble(inner(TrialFunction(V), TestFunction(V)) * dx)
_bp = assemble(inner(_init_expr, TestFunction(V)) * dx)
_bc_proj = DirichletBC(V, _init_expr, facets, INLET_ID)
_bc_proj.apply(_Ap, _bp)
_sp_proj = KrylovSolver("cg", "jacobi")
_sp_proj.parameters["relative_tolerance"] = 1e-8
_sp_proj.parameters["maximum_iterations"] = 2000
_sp_proj.parameters["monitor_convergence"] = False
_sp_proj.solve(_Ap, u_inlet_ctrl.vector(), _bp)
log(f"||u_ctrl||_inf = {u_inlet_ctrl.vector().norm('linf'):.4e}")
_bc_in_tmp  = DirichletBC(V, Constant(tuple([0.0]*gdim)), facets, INLET_ID)
inlet_dofs  = np.array(list(_bc_in_tmp.get_boundary_values().keys()), dtype=np.int64)
log(f"Inlet DOFs: {len(inlet_dofs)}")
_bc_wall_tmp   = DirichletBC(V, Constant(tuple([0.0]*gdim)), facets, WALL_ID)
_inlet_dof_set = set(inlet_dofs.tolist())
_wall_dof_set  = set(_bc_wall_tmp.get_boundary_values().keys())
inlet_edge_dofs = np.array(sorted(_inlet_dof_set & _wall_dof_set), dtype=np.int64)
log(f"Inlet-wall edge DOFs (hard no-slip): {len(inlet_edge_dofs)}")
# ==============================================================================
# REFERENCE FLOW RATE (monitoring only)
# ==============================================================================
banner(f"REFERENCE FLOW RATE  (U_MAX_TRUE={U_MAX_TRUE} — monitoring only)")
_ref_expr = ParabolicInletPCA(inlet_center, inlet_radius, inlet_normal, U_MAX_TRUE, degree=2)
_u_ref = Function(V)
_Ar = assemble(inner(TrialFunction(V), TestFunction(V)) * dx)
_br = assemble(inner(_ref_expr, TestFunction(V)) * dx)
DirichletBC(V, _ref_expr, facets, INLET_ID).apply(_Ar, _br)
_sr = KrylovSolver("cg", "jacobi")
_sr.parameters["relative_tolerance"] = 1e-8
_sr.parameters["maximum_iterations"] = 2000
_sr.parameters["monitor_convergence"] = False
_sr.solve(_Ar, _u_ref.vector(), _br)
Q_ref = abs(float(assemble(dot(_u_ref, nrm) * ds_in)))
log(f"Q_ref (monitor) = {Q_ref:.6e}")
# ==============================================================================
# LOAD OBSERVED CONCENTRATION
# ==============================================================================
banner("LOADING OBSERVED CONCENTRATION")
if not os.path.isfile(OBS_XDMF):
    raise FileNotFoundError(f"Missing observation file: {OBS_XDMF}")
c_obs_data = Function(Q, name="ObsData")
if rank == 0:
    print(f"[OBS READ] file={OBS_XDMF}")
    print(f"[OBS READ] field={OBS_FUNC_NAME}, index={OBS_TIME_INDEX}")
with XDMFFile(comm, OBS_XDMF) as xf:
    xf.read_checkpoint(c_obs_data, OBS_FUNC_NAME, OBS_TIME_INDEX)
OBS_C_MIN  = float(c_obs_data.vector().min())
OBS_C_MAX  = float(c_obs_data.vector().max())
OBS_C_MEAN = float(c_obs_data.vector().sum()) / Q.dim()
log(f"c_obs (full domain) — min={OBS_C_MIN:.4e}  max={OBS_C_MAX:.4e}  mean={OBS_C_MEAN:.4e}")
_c_obs_masked    = c_obs_data.vector().get_local()[mask_dofs]
OBS_MASK_C_MIN   = float(_c_obs_masked.min())
OBS_MASK_C_MAX   = float(_c_obs_masked.max())
OBS_MASK_C_MEAN  = float(_c_obs_masked.mean())
log(f"c_obs (masked)      — min={OBS_MASK_C_MIN:.4e}  "
    f"max={OBS_MASK_C_MAX:.4e}  mean={OBS_MASK_C_MEAN:.4e}")
section("MASK DIAGNOSTIC — coordinates and concentration at matched DOFs")
log(f"  Number of matched DOFs : {len(mask_dofs)}")
log(f"  x range : [{masked_coords_dofs[:,0].min():+.6e}, {masked_coords_dofs[:,0].max():+.6e}]")
log(f"  y range : [{masked_coords_dofs[:,1].min():+.6e}, {masked_coords_dofs[:,1].max():+.6e}]")
log(f"  z range : [{masked_coords_dofs[:,2].min():+.6e}, {masked_coords_dofs[:,2].max():+.6e}]")
log(f"  c_obs range (mask) : [{OBS_MASK_C_MIN:+.6e}, {OBS_MASK_C_MAX:+.6e}]")
# ==============================================================================
# LOAD OBSERVED (REFERENCE) VELOCITY  (for final comparison)
# ==============================================================================
banner("LOADING OBSERVED (REFERENCE) VELOCITY")
OBS_VEL_XDMF = os.path.join(OBS_DIR, "velocity_final.xdmf")
u_obs_data   = Function(V, name="ReferenceVelocity")
if not os.path.isfile(OBS_VEL_XDMF):
    raise FileNotFoundError(f"Missing reference velocity file: {OBS_VEL_XDMF}")
with XDMFFile(comm, OBS_VEL_XDMF) as xf:
    xf.read_checkpoint(u_obs_data, "velocity", 0)
log(f"u_obs loaded — ||u||_inf = {u_obs_data.vector().norm('linf'):.4e}")
# ==============================================================================
# ONE-TIME NS WARM-START  (N_SETTLE steps, then cached)
# ==============================================================================
banner("ONE-TIME NS WARM-START  (TWO-OUTLET RAMP)")
u_nm1_ns = Function(V); u_nm1_ns.vector().zero()
u_n_ns   = Function(V); u_n_ns.vector().zero()
p_n_ns   = Function(Q); p_n_ns.vector().zero()
def _load_inv_ramp():
    with XDMFFile(comm, INV_RAMP_U_XDMF) as xf:
        xf.read_checkpoint(u_n_ns, "velocity", 0)
    with XDMFFile(comm, INV_RAMP_P_XDMF) as xf:
        xf.read_checkpoint(p_n_ns, "pressure", 0)
    u_nm1_ns.assign(u_n_ns)
if SKIP_NS_INV:
    section("Loading cached inverse ramp checkpoint  (SKIP_NS_INV=True)")
    for _path in (INV_RAMP_U_XDMF, INV_RAMP_P_XDMF):
        if not os.path.isfile(_path):
            raise RuntimeError(
                f"Inverse ramp checkpoint not found: {_path}\n"
                "Run once with SKIP_NS_INV=False to generate it."
            )
    _load_inv_ramp()
    log(f"Loaded u_n / u_nm1 ← {INV_RAMP_U_XDMF}")
    log(f"Loaded p_          ← {INV_RAMP_P_XDMF}")
else:
    if os.path.isfile(INV_RAMP_U_XDMF) and os.path.isfile(INV_RAMP_P_XDMF):
        section("Loading cached inverse ramp checkpoint")
        _load_inv_ramp()
        log(f"Loaded from {INV_RAMP_U_XDMF}")
    else:
        section(f"Running ONE-TIME inverse NS ramp from zero  ({N_SETTLE} steps)")
        _ramp_expr = ConstantInlet(inlet_normal, 0.0, degree=1)
        bcu_r = [
            DirichletBC(V, _ramp_expr, facets, INLET_ID),
            DirichletBC(V, Constant(tuple([0.0]*gdim)), facets, WALL_ID),
        ]
        # Both outlet tags → p=0
        bcp_r = [DirichletBC(Q, Constant(0.0), facets, oid) for oid in OUTLET_IDS]
        set_working_tape(Tape())
        Mv_r = assemble(inner(u_tr, v_te)*dx);             Mv_r.ident_zeros()
        K_r  = assemble(inner(grad(u_tr), grad(v_te))*dx); K_r.ident_zeros()
        A2_r = assemble(inner(grad(p_tr), grad(q_te))*dx)
        A3_r = assemble(RHO*inner(u_tr, v_te)*dx);         A3_r.ident_zeros()
        Ms_r = assemble(inner(p_tr, q_te)*dx);             Ms_r.ident_zeros()
        for bc in bcp_r:
            bc.apply(A2_r)
            bc.apply(Ms_r)
        A2_r.ident_zeros()
        u_sr  = Function(V); phi_r = Function(Q); p_nr = Function(Q)
        u_nwr = Function(V); d_r   = Function(Q)
        b1_r  = Vector(); Mv_r.init_vector(b1_r, 0)
        b2_r  = Vector(); A2_r.init_vector(b2_r, 0)
        b3_r  = Vector(); A3_r.init_vector(b3_r, 0)
        bd_r  = Vector(); Ms_r.init_vector(bd_r, 0)
        t2_r  = Vector(); Mv_r.init_vector(t2_r, 0)
        C_r   = PETScMatrix()
        assemble(inner(dot(u_n_ns, nabla_grad(u_tr)), v_te)*dx, tensor=C_r)
        A1_r  = PETScMatrix()
        assemble(inner(u_tr, v_te)*dx, tensor=A1_r)
        def _ks(m, p_):
            s = KrylovSolver(m, p_)
            s.parameters["relative_tolerance"] = 1e-8
            s.parameters["maximum_iterations"] = 2000
            s.parameters["monitor_convergence"] = False
            return s
        sol_ur = _ks("gmres", "ilu")
        sol_pr = _ks("cg", "hypre_amg")
        sol_cr = _ks("cg", "jacobi")
        sol_dr = _ks("cg", "jacobi")
        odr = _outlet_ds()
        t0r = time.time()
        for step in range(N_SETTLE):
            if step < N_RAMP_STEPS:
                Um = U_INIT * (step + 1) / N_RAMP_STEPS
            else:
                Um = U_INIT
            _ramp_expr.mag = Um
            uc_r = Function(V)
            uc_r.vector().axpy(2.0, u_n_ns.vector())
            uc_r.vector().axpy(-1.0, u_nm1_ns.vector())
            assemble(inner(dot(uc_r, nabla_grad(u_tr)), v_te)*dx, tensor=C_r)
            A1_r.zero(); b1_r.zero()
            if step == 0:
                ar = float(RHO / DT_NS)
                A1_r.axpy(ar, Mv_r, True)
                A1_r.axpy(float(MU), K_r, True)
                A1_r.axpy(1.0, C_r, True)
                Mv_r.mult(u_n_ns.vector(), b1_r)
                b1_r *= ar
            else:
                ar = float(3.0 * RHO / (2.0 * DT_NS))
                A1_r.axpy(ar, Mv_r, True)
                A1_r.axpy(float(MU), K_r, True)
                A1_r.axpy(1.0, C_r, True)
                t2_r.zero()
                t2_r.axpy(4.0, u_n_ns.vector())
                t2_r.axpy(-1.0, u_nm1_ns.vector())
                Mv_r.mult(t2_r, b1_r)
                b1_r *= float(RHO / (2.0 * DT_NS))
            # Natural pressure BC on BOTH outlets
            b1_r.axpy(1.0, assemble(
                p_n_ns*div(v_te)*dx - dot(p_n_ns*nrm, v_te)*odr
            ))
            for bc in bcu_r:
                bc.apply(A1_r, b1_r)
            sol_ur.solve(A1_r, u_sr.vector(), b1_r)
            b2_r.zero()
            if step == 0:
                assemble(-(Constant(RHO/DT_NS))*div(u_sr)*q_te*dx, tensor=b2_r)
                cr = float(DT_NS)
            else:
                assemble(-(Constant(3.0*RHO/(2.0*DT_NS)))*div(u_sr)*q_te*dx, tensor=b2_r)
                cr = float(2.0*DT_NS/3.0)
            for bc in bcp_r:
                bc.apply(b2_r)
            sol_pr.solve(A2_r, phi_r.vector(), b2_r)
            p_nr.vector().zero()
            p_nr.vector().axpy(1.0, p_n_ns.vector())
            p_nr.vector().axpy(1.0, phi_r.vector())
            bd_r.zero()
            assemble(div(u_sr)*q_te*dx, tensor=bd_r)
            for bc in bcp_r:
                bc.apply(bd_r)
            sol_dr.solve(Ms_r, d_r.vector(), bd_r)
            p_nr.vector().axpy(-float(MU), d_r.vector())
            for bc in bcp_r:
                bc.apply(p_nr.vector())
            b3_r.zero()
            assemble(
                RHO*inner(u_sr, v_te)*dx - cr*inner(grad(phi_r), v_te)*dx,
                tensor=b3_r,
            )
            for bc in bcu_r:
                bc.apply(A3_r, b3_r)
            sol_cr.solve(A3_r, u_nwr.vector(), b3_r)
            u_nm1_ns.assign(u_n_ns)
            u_n_ns.assign(u_nwr)
            p_n_ns.assign(p_nr)
            if rank == 0 and step % 10 == 0:
                phase  = "RAMP" if step < N_RAMP_STEPS else "POST"
                Q_in   = float(assemble(dot(u_nwr, nrm)*ds_in))
                Q_o3   = float(assemble(dot(u_nwr, nrm)*ds(3)))
                Q_o4   = float(assemble(dot(u_nwr, nrm)*ds(4)))
                div_l2 = float(np.sqrt(assemble((div(u_nwr)**2)*dx)))
                log(f"  [{phase} {step:3d}/{N_SETTLE}] Umax={Um:.4f}  "
                    f"Qin={Q_in:+.4e}  Qo3={Q_o3:+.4e}  Qo4={Q_o4:+.4e}  "
                    f"div_L2={div_l2:.3e}")
        log(f"One-time ramp done in {timer_str(t0r)}")
        with XDMFFile(comm, INV_RAMP_U_XDMF) as xf:
            xf.write_checkpoint(u_n_ns, "velocity", 0, XDMFFile.Encoding.HDF5, False)
        with XDMFFile(comm, INV_RAMP_P_XDMF) as xf:
            xf.write_checkpoint(p_n_ns, "pressure", 0, XDMFFile.Encoding.HDF5, False)
        log(f"Inverse ramp checkpoint saved → {INV_RAMP_U_XDMF}")
# Store NS state as plain numpy arrays (passed into each epoch's forward)
u_nm1_state_vec = u_nm1_ns.vector().get_local().copy()
u_n_state_vec   = u_n_ns.vector().get_local().copy()
p_state_vec     = p_n_ns.vector().get_local().copy()
vec_stats("u_n (one-time ramp)",   u_n_state_vec)
vec_stats("u_nm1 (one-time ramp)", u_nm1_state_vec)
# ==============================================================================
# ADAM STATE
# ==============================================================================
m_adam = u_inlet_ctrl.vector().copy(); m_adam.zero()
v_adam = u_inlet_ctrl.vector().copy(); v_adam.zero()
adam_t = 0
# ==============================================================================
# FORWARD PASS PER EPOCH  (annotated for dolfin-adjoint)
# ==============================================================================
def forward_pass_epoch(u_ctrl, u_nm1_vec0, u_n_vec0, p_vec0,
                       annotate=True, verbose=False, label=""):
    """
    N_NS_PER_EPOCH BDF2-IPCS steps starting from the given NS state vectors,
    then N_ADV_OBS advection-diffusion steps with the pulsed concentration BC.
    Both outlet tags use p=0 BCs and the combined natural outflow term.
    Returns:
        c_new_adv      — final concentration (used in J)
        u_n            — velocity after NS steps (for per-epoch save)
        new_u_nm1_vec  — updated u_nm1 numpy array (carry forward)
        new_u_n_vec    — updated u_n   numpy array (carry forward)
        new_p_vec      — updated p     numpy array (carry forward)
    """
    rho_c = Constant(RHO)
    mu_c  = Constant(MU)
    odsl  = _outlet_ds()
    bcu_opt = [
        DirichletBC(V, u_ctrl, facets, INLET_ID),
        DirichletBC(V, Constant(tuple([0.0]*gdim)), facets, WALL_ID),
    ]
    bcp_opt = [DirichletBC(Q, Constant(0.0), facets, oid) for oid in OUTLET_IDS]
    # Initialise from provided state vectors
    u_nm1 = Function(V)
    u_nm1.vector().set_local(u_nm1_vec0); u_nm1.vector().apply("insert")
    u_n   = Function(V)
    u_n.vector().set_local(u_n_vec0);   u_n.vector().apply("insert")
    p_n   = Function(Q)
    p_n.vector().set_local(p_vec0);     p_n.vector().apply("insert")
    u_sl  = Function(V); u_nl = Function(V)
    phi_l = Function(Q); p_nl = Function(Q)
    if verbose:
        section(f"[{label}] NS update  (N_NS_PER_EPOCH={N_NS_PER_EPOCH}, two outlets)")
    for it in range(N_NS_PER_EPOCH):
        # Always use BDF2 (settled state already past step 0)
        uc_ufl = 2.0*u_n - u_nm1
        ac     = Constant(3.0*RHO / (2.0*DT_NS))
        ar     = Constant(RHO  / (2.0*DT_NS))
        ur_ufl = 4.0*u_n - u_nm1
        cf     = Constant(2.0*DT_NS / 3.0)
        dc     = Constant(-3.0*RHO / (2.0*DT_NS))
        # Step 1: tentative velocity
        a1 = (ac*inner(u_tr, v_te)*dx
              + mu_c*inner(grad(u_tr), grad(v_te))*dx
              + inner(dot(uc_ufl, nabla_grad(u_tr)), v_te)*dx)
        L1 = (ar*inner(ur_ufl, v_te)*dx
              + p_n*div(v_te)*dx
              - dot(p_n*nrm, v_te)*odsl)   # natural BC on BOTH outlets
        solve(a1 == L1, u_sl, bcu_opt, solver_parameters=_sp_gmres, annotate=annotate)
        # Step 2: pressure correction
        a2 = inner(grad(p_tr), grad(q_te))*dx
        L2 = dc*div(u_sl)*q_te*dx
        solve(a2 == L2, phi_l, bcp_opt, solver_parameters=_sp_cg_amg, annotate=annotate)
        # Rotational pressure update
        ap = inner(p_tr, q_te)*dx
        Lp = (p_n + phi_l - mu_c*div(u_sl))*q_te*dx
        solve(ap == Lp, p_nl, bcp_opt, solver_parameters=_sp_cg_jac, annotate=annotate)
        # Step 3: velocity correction
        a3 = rho_c*inner(u_tr, v_te)*dx
        L3 = rho_c*inner(u_sl, v_te)*dx - cf*inner(grad(phi_l), v_te)*dx
        solve(a3 == L3, u_nl, bcu_opt, solver_parameters=_sp_cg_jac, annotate=annotate)
        if verbose and rank == 0:
            Q_in = float(assemble(dot(u_nl, nrm)*ds_in))
            Q_o3 = float(assemble(dot(u_nl, nrm)*ds(3)))
            Q_o4 = float(assemble(dot(u_nl, nrm)*ds(4)))
            print(f"    NS-it {it+1}: Qin={Q_in:+.4e}  Qo3={Q_o3:+.4e}  "
                  f"Qo4={Q_o4:+.4e}  "
                  f"div_L2={float(np.sqrt(assemble((div(u_nl)**2)*dx))):.3e}  "
                  f"u_inf={float(u_nl.vector().norm('linf')):.3e}", flush=True)
        u_nm1.assign(u_n,  annotate=annotate)
        u_n.assign(u_nl,   annotate=annotate)
        p_n.assign(p_nl,   annotate=annotate)
    # ── Advection-diffusion (frozen velocity = u_n after NS steps) ────────────
    u_stdy    = u_n
    dtc       = Constant(DT_ADV)
    c_tr_     = TrialFunction(Q)
    w_te_     = TestFunction(Q)
    c_n_adv   = Function(Q)   # zero IC
    c_new_adv = Function(Q)
    a_adv = (w_te_*c_tr_*dx
             + dtc*w_te_*dot(u_stdy, grad(c_tr_))*dx
             + dtc*Constant(DIFF_COEF)*dot(grad(w_te_), grad(c_tr_))*dx)
    if verbose:
        section(f"[{label}] Adv-diff  ({N_ADV_OBS} steps, dt={DT_ADV:.3e}, pulsed BC)")
    for k in range(N_ADV_OBS):
        t_adv  = (k + 1) * DT_ADV
        bc_val = _conc_bc_value(t_adv)
        bc_c   = DirichletBC(Q, Constant(bc_val), facets, INLET_ID)
        solve(a_adv == w_te_*c_n_adv*dx, c_new_adv, bc_c,
              solver_parameters=_sp_gmres, annotate=annotate)
        c_n_adv.assign(c_new_adv, annotate=annotate)
        if verbose and rank == 0 and (k % ADV_PRINT_EVERY == 0 or k == N_ADV_OBS - 1):
            _cv = c_new_adv.vector().get_local()
            print(f"    ADV {k+1:4d}/{N_ADV_OBS}  t={t_adv:.4e}  bc={bc_val:.3f}  "
                  f"min={_cv.min():+.4e}  max={_cv.max():+.4e}  mean={_cv.mean():+.4e}",
                  flush=True)
    new_u_nm1_vec = u_nm1.vector().get_local().copy()
    new_u_n_vec   = u_n.vector().get_local().copy()
    new_p_vec     = p_n.vector().get_local().copy()
    return c_new_adv, u_n, new_u_nm1_vec, new_u_n_vec, new_p_vec
# ==============================================================================
# OPTIMISATION LOOP
# ==============================================================================
banner("OPTIMISATION LOOP  (MASKED OBJECTIVE, PULSED BC, TWO OUTLETS)")
log("J = J_data + J_reg")
log("  J_data = 0.5 * integral_MASK ( (c_pred - c_obs)^2 ) dx_mask")
log(f"  J_reg  = 0.5 * GAMMA_REG * integral_inlet |u_ctrl|^2 ds,  GAMMA_REG={GAMMA_REG:.3e}")
log(f"  Masked cells: {n_masked_cells} / {mesh.num_cells()}")
loss_history   = []
loss_data_hist = []
loss_reg_hist  = []
umax_history   = []
qpred_history  = []
grad_history   = []
_hdr = (f"{'Ep':>4} | {'LR':>10} | {'J_total':>13} | {'J_data':>13} | {'J_reg':>13} | "
        f"{'GradNorm':>10} | {'Q_pred':>12} | {'Q_ref':>12} | {'relErrQ':>9} | "
        f"{'U_pred':>9} | {'c_min':>10} | {'c_max':>10} | {'c_mean':>11} | {'dt(s)':>7}")
_hdiv = "=" * len(_hdr)
def _ph():
    if rank == 0:
        print(f"\n{_hdiv}\n{_hdr}\n{_hdiv}", flush=True)
_ph()
for epoch in range(1, EPOCHS + 1):
    t_ep     = time.time()
    _verbose = (epoch == 1) or (epoch % 10 == 0)
    # LR step-decay
    if epoch > 1 and epoch % LR_DECAY_EVERY == 0:
        LR_current = LR_current * LR_DECAY
        log(f"  *** LR decay at epoch {epoch}: LR = {LR_current:.6e} ***")
    lr_history.append(LR_current)
    get_working_tape().clear_tape()
    ctrl = Control(u_inlet_ctrl)
    # --------------------------------------------------------------
    # Enforce hard constraint BEFORE forward solve (safety clamp)
    # --------------------------------------------------------------
    u_tmp = u_inlet_ctrl.vector().get_local()
    if len(inlet_edge_dofs) > 0:
        u_tmp[inlet_edge_dofs] = 0.0
    u_inlet_ctrl.vector().set_local(u_tmp)
    u_inlet_ctrl.vector().apply("insert")
    c_pred, u_vel_epoch, u_nm1_new, u_n_new, p_new = forward_pass_epoch(
        u_inlet_ctrl,
        u_nm1_state_vec, u_n_state_vec, p_state_vec,
        annotate=True, verbose=_verbose, label=f"epoch {epoch}",
    )
    if _verbose:
        _ph()
    _cp         = c_pred.vector().get_local()
    c_pred_min  = float(_cp.min())
    c_pred_max  = float(_cp.max())
    c_pred_mean = float(_cp.mean())
    # Objective — J_data restricted to masked region
    gamma  = Constant(GAMMA_REG)
    J_data = assemble(0.5*(c_pred - c_obs_data)**2 * dx_mask)
    J_reg  = assemble(0.5 * gamma * dot(u_inlet_ctrl, u_inlet_ctrl) * ds_in)
    J      = J_data + J_reg
    loss_history.append(float(J))
    loss_data_hist.append(float(J_data))
    loss_reg_hist.append(float(J_reg))
    rf    = ReducedFunctional(J, ctrl)
    dJ    = rf.derivative()
    g_norm = float(dJ.vector().norm("l2"))
    grad_history.append(g_norm)
    g         = dJ.vector().get_local()
    u_old     = u_inlet_ctrl.vector().get_local()
    u_new_vec = u_old.copy()
    if OPTIMIZER.lower() == "gd":
        u_new_vec[inlet_dofs] -= LR_current * g[inlet_dofs]
    else:  # Adam
        adam_t += 1
        m_v = ADAM_B1 * m_adam.get_local() + (1 - ADAM_B1) * g
        v_v = ADAM_B2 * v_adam.get_local() + (1 - ADAM_B2) * (g ** 2)
        mh  = m_v / (1 - ADAM_B1 ** adam_t)
        vh  = v_v / (1 - ADAM_B2 ** adam_t)
        u_new_vec[inlet_dofs] -= (
            LR_current * mh[inlet_dofs] / (np.sqrt(vh[inlet_dofs]) + ADAM_EPS)
        )
        m_adam.set_local(m_v); m_adam.apply("insert")
        v_adam.set_local(v_v); v_adam.apply("insert")
    # Hard-enforce no-slip at inlet-wall edge
    u_new_vec[inlet_edge_dofs] = 0.0
    u_inlet_ctrl.vector().set_local(u_new_vec)
    u_inlet_ctrl.vector().apply("insert")
    # Carry NS state forward
    u_nm1_state_vec = u_nm1_new
    u_n_state_vec   = u_n_new
    p_state_vec     = p_new
    # Monitoring
    Q_pred    = abs(float(assemble(dot(u_inlet_ctrl, nrm) * ds_in)))
    rel_err_Q = abs(Q_pred - Q_ref) / (Q_ref + 1e-14)
    _uim = (np.linalg.norm(
        u_inlet_ctrl.vector().get_local()[inlet_dofs].reshape(-1, gdim), axis=1)
        if len(inlet_dofs) > 0 else np.array([0.0]))
    U_max_pred = float(_uim.max())
    umax_history.append(U_max_pred)
    qpred_history.append(Q_pred)
    ep_time = time.time() - t_ep
    log(f"{epoch:>4} | {LR_current:>10.3e} | {float(J):>13.6e} | {float(J_data):>13.6e} | "
        f"{float(J_reg):>13.6e} | {g_norm:>10.3e} | {Q_pred:>12.6e} | {Q_ref:>12.6e} | "
        f"{rel_err_Q:>9.3e} | {U_max_pred:>9.4f} | {c_pred_min:>10.4e} | "
        f"{c_pred_max:>10.4e} | {c_pred_mean:>11.4e} | {ep_time:>7.1f}")
    if epoch % 5 == 0:
        section(f"Epoch {epoch} diagnostics")
        log(f"  J_total          : {float(J):.8e}")
        log(f"  J_data (mask)    : {float(J_data):.8e}  ← integrated over masked region only")
        log(f"  J_reg            : {float(J_reg):.8e}  (gamma={GAMMA_REG:.3e})")
        log(f"  ||grad||_L2      : {g_norm:.4e}")
        log(f"  Q_pred/Q_ref     : {Q_pred:.6e}/{Q_ref:.6e}  relErr={rel_err_Q:.4e}")
        log(f"  U_max_pred       : {U_max_pred:.4f}")
        log(f"  c_pred min/max   : {c_pred_min:.4e}/{c_pred_max:.4e}")
        log(f"  c_obs (mask)     : {OBS_MASK_C_MIN:.4e}/{OBS_MASK_C_MAX:.4e}")
        vec_stats("u_ctrl", u_inlet_ctrl.vector().get_local())
        log(f"  epoch time       : {ep_time:.2f}s")
    # Per-epoch saves
    vel_file  = os.path.join(EPOCH_VEL_DIR,  f"velocity_epoch_{epoch:04d}.xdmf")
    conc_file = os.path.join(EPOCH_CONC_DIR, f"concentration_epoch_{epoch:04d}.xdmf")
    u_vel_epoch.rename("velocity",      "velocity")
    c_pred.rename     ("concentration", "concentration")
    with XDMFFile(comm, vel_file) as xf:
        xf.write_checkpoint(u_vel_epoch, "velocity",      0, XDMFFile.Encoding.HDF5, False)
    with XDMFFile(comm, conc_file) as xf:
        xf.write_checkpoint(c_pred,      "concentration", 0, XDMFFile.Encoding.HDF5, False)
    log(f"  [SAVE] {vel_file}")
    log(f"  [SAVE] {conc_file}")
    if epoch == EPOCHS:
        final_vel_path  = os.path.join(FINAL_VEL_DIR,  "final_velocity.xdmf")
        final_conc_path = os.path.join(FINAL_CONC_DIR, "final_concentration.xdmf")
        with XDMFFile(comm, final_vel_path) as xf:
            xf.write_checkpoint(u_vel_epoch, "velocity",      0, XDMFFile.Encoding.HDF5, False)
        with XDMFFile(comm, final_conc_path) as xf:
            xf.write_checkpoint(c_pred,      "concentration", 0, XDMFFile.Encoding.HDF5, False)
        log(f"  [FINAL SAVE] Velocity      → {final_vel_path}")
        log(f"  [FINAL SAVE] Concentration → {final_conc_path}")
# ==============================================================================
# SAVE RESULTS
# ==============================================================================
banner("SAVING RESULTS")
np.save(os.path.join(OUT_DIR, "loss_history.npy"),      np.array(loss_history))
np.save(os.path.join(OUT_DIR, "loss_data_history.npy"), np.array(loss_data_hist))
np.save(os.path.join(OUT_DIR, "loss_reg_history.npy"),  np.array(loss_reg_hist))
np.save(os.path.join(OUT_DIR, "umax_history.npy"),      np.array(umax_history))
np.save(os.path.join(OUT_DIR, "qpred_history.npy"),     np.array(qpred_history))
np.save(os.path.join(OUT_DIR, "grad_history.npy"),      np.array(grad_history))
np.save(os.path.join(OUT_DIR, "lr_history.npy"),        np.array(lr_history))
with XDMFFile(comm, os.path.join(OUT_DIR, "optimized_inlet_ctrl.xdmf")) as xf:
    xf.write_checkpoint(u_inlet_ctrl, "InletControl", 0, XDMFFile.Encoding.HDF5, False)
log(f"Optimized inlet → {OUT_DIR}/optimized_inlet_ctrl.xdmf")
section("Final un-annotated forward pass (using last warm-start NS state)")
c_final, u_final, _, _, _ = forward_pass_epoch(
    u_inlet_ctrl,
    u_nm1_state_vec, u_n_state_vec, p_state_vec,
    annotate=False, verbose=True, label="FINAL",
)
with XDMFFile(comm, os.path.join(OUT_DIR, "final_concentration.xdmf")) as xf:
    xf.write(c_final, 0.0)
u_final.rename("Velocity", "predicted")
File(os.path.join(OUT_DIR, "final_velocity_predicted.pvd")) << u_final
u_obs_data.rename("Velocity", "reference")
File(os.path.join(OUT_DIR, "final_velocity_reference.pvd")) << u_obs_data
log("Saved: final_velocity_predicted.pvd/.vtu")
log("Saved: final_velocity_reference.pvd/.vtu")
Q_final = abs(float(assemble(dot(u_inlet_ctrl, nrm) * ds_in)))
_ufm = (np.linalg.norm(
    u_inlet_ctrl.vector().get_local()[inlet_dofs].reshape(-1, gdim), axis=1)
    if len(inlet_dofs) > 0 else np.array([0.0]))
U_max_final  = float(_ufm.max())
_cf          = c_final.vector().get_local()
_cf_masked   = _cf[mask_dofs]
log(f"\n  c_final (masked) — min={_cf_masked.min():+.4e}  "
    f"max={_cf_masked.max():+.4e}  mean={_cf_masked.mean():+.4e}")
banner("FINAL RESULTS SUMMARY")
log(f"  {'Quantity':<32}  {'Predicted':>14}  {'Reference':>14}  {'Rel. Error':>12}")
log(f"  {'-'*32}  {'-'*14}  {'-'*14}  {'-'*12}")
log(f"  {'Q_inlet (monitor)':<32}  {Q_final:>14.6e}  {Q_ref:>14.6e}  "
    f"{abs(Q_final-Q_ref)/(Q_ref+1e-14):>12.4e}")
log(f"  {'U_max_inlet (monitor)':<32}  {U_max_final:>14.4f}  {U_MAX_TRUE:>14.4f}  "
    f"{abs(U_max_final-U_MAX_TRUE)/(U_MAX_TRUE+1e-14):>12.4e}")
log(f"  {'c_min (mask)':<32}  {_cf_masked.min():>14.4e}  {OBS_MASK_C_MIN:>14.4e}")
log(f"  {'c_max (mask)':<32}  {_cf_masked.max():>14.4e}  {OBS_MASK_C_MAX:>14.4e}")
log(f"  {'J_total initial':<32}  {loss_history[0]:>14.6e}")
log(f"  {'J_total final':<32}  {loss_history[-1]:>14.6e}")
log(f"  {'Reduction':<32}  {loss_history[0]/max(loss_history[-1], 1e-300):>14.2f}x")
log(f"  {'Best epoch':<32}  {int(np.argmin(loss_history))+1:>14d}")
log(f"  {'Masked cells in J_data':<32}  {n_masked_cells:>14d}")
log(f"  {'Total mesh cells':<32}  {mesh.num_cells():>14d}")
with open(os.path.join(OUT_DIR, "final_summary.txt"), "w") as f:
    f.write(
        f"SKIP_NS_INV:{SKIP_NS_INV}\n"
        f"N_NS_PER_EPOCH:{N_NS_PER_EPOCH}\n"
        f"GAMMA_REG:{GAMMA_REG:.6e}\n"
        f"OUTLET_IDS:{OUTLET_IDS}\n"
        f"MASK_VTK:{MASK_VTK}\n"
        f"MASK_N_PTS:{n_mask_pts}\n"
        f"MASK_N_CELLS:{n_masked_cells}\n"
        f"Q_pred:{Q_final:.6e}\nQ_ref:{Q_ref:.6e}\n"
        f"U_max_pred:{U_max_final:.6f}\nU_max_ref(monitor):{U_MAX_TRUE:.6f}\n"
        f"J_total_initial:{loss_history[0]:.6e}\nJ_total_final:{loss_history[-1]:.6e}\n"
        f"J_data_final:{loss_data_hist[-1]:.6e}\nJ_reg_final:{loss_reg_hist[-1]:.6e}\n"
        f"T_ADV:{T_ADV}\nN_ADV_OBS:{N_ADV_OBS}\nDT_ADV:{DT_ADV:.6e}\n"
        f"CONC_BC:pulsed_period0.2s\nOBS_TIME_INDEX:{OBS_TIME_INDEX}\n"
        f"N_SETTLE_ONE_TIME:{N_SETTLE}\nN_RAMP_STEPS:{N_RAMP_STEPS}\nN_POST_STEPS:{N_POST_STEPS}\n"
    )
# ==============================================================================
# INLET PROFILE VISUALISATION  (predicted vs ground-truth parabola)
# ==============================================================================
banner("INLET PROFILE VISUALISATION")

# ------------------------------------------------------------------------------
# Step 1 — Extract inlet node coordinates and velocity magnitudes
#
# Use V.tabulate_dof_coordinates() (indexed by GLOBAL DOF index, shape V.dim()×gdim)
# rather than the sub-space dofmap, which avoids the IndexError that arises when
# inlet_dofs values (up to V.dim()-1) are used to index into a sub-space array of
# length V.dim()/gdim.
# ------------------------------------------------------------------------------
section("Inlet profile: extracting node coordinates via global DOF map")

# V_dof_coords[i] is the spatial location of global DOF i  (shape: V.dim() × gdim)
V_dof_coords = V.tabulate_dof_coordinates().reshape(V.dim(), gdim)

# Coordinates for every global inlet DOF (includes all gdim components per node)
inlet_dof_coords_all = V_dof_coords[inlet_dofs]   # shape: (n_inlet_dofs, gdim)

# Unique physical nodes: round to avoid floating-point duplicates, take unique rows
_rounded = np.round(inlet_dof_coords_all, decimals=10)
_, _unique_idx = np.unique(_rounded, axis=0, return_index=True)
coords_inlet_nodes = inlet_dof_coords_all[np.sort(_unique_idx)]   # (n_nodes, gdim)

log(f"Inlet nodes for visualization : {len(coords_inlet_nodes)}")
log(f"Radius check (max dist from center): "
    f"{np.max(np.linalg.norm(coords_inlet_nodes - inlet_center, axis=1)):.6f}")

# ------------------------------------------------------------------------------
# Step 2 — Velocity magnitude at each unique inlet node
#
# For each unique node, find the gdim co-located global DOFs and read the
# velocity components directly from the flat u_ctrl vector.
# ------------------------------------------------------------------------------
u_full   = u_inlet_ctrl.vector().get_local()   # shape: (V.dim(),)
mag_pred = np.zeros(len(coords_inlet_nodes))

for ni, node_coord in enumerate(coords_inlet_nodes):
    dists_to_node     = np.linalg.norm(inlet_dof_coords_all - node_coord, axis=1)
    close_mask        = dists_to_node < 1e-14 * max(inlet_radius, 1.0)
    close_global_dofs = inlet_dofs[close_mask]

    if len(close_global_dofs) >= gdim:
        u_components = u_full[close_global_dofs[:gdim]]
    elif len(close_global_dofs) > 0:
        u_components = u_full[close_global_dofs]
    else:
        u_components = np.zeros(gdim)

    mag_pred[ni] = np.linalg.norm(u_components)

# ------------------------------------------------------------------------------
# Step 3 — Ground-truth parabolic profile at the same physical nodes
# ------------------------------------------------------------------------------
mag_true = np.zeros(len(coords_inlet_nodes))
for ni, x in enumerate(coords_inlet_nodes):
    d     = x - inlet_center
    r_vec = d - np.dot(d, inlet_normal) * inlet_normal
    r     = np.linalg.norm(r_vec)
    mag_true[ni] = U_MAX_TRUE * max(0.0, 1.0 - (r / inlet_radius) ** 2)

log(f"Predicted magnitude range   : [{mag_pred.min():+.6e}, {mag_pred.max():+.6e}]")
log(f"Ground-truth magnitude range: [{mag_true.min():+.6e}, {mag_true.max():+.6e}]")

# ------------------------------------------------------------------------------
# Step 4 — Local 2-D coordinate system on the inlet plane
# ------------------------------------------------------------------------------
n  = inlet_normal.copy()
a  = np.array([1.0, 0.0, 0.0])
if abs(np.dot(a, n)) > 0.9:
    a = np.array([0.0, 1.0, 0.0])
e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2)

local  = coords_inlet_nodes - inlet_center
x2d    = local @ e1
y2d    = local @ e2

# 90° in-plane rotation (matches forward convention)
x_rot = -y2d
y_rot =  x2d

# ------------------------------------------------------------------------------
# Step 5 — Interpolate onto a structured grid and mask outside the disk
# ------------------------------------------------------------------------------
GRID_RES = 200
xi = np.linspace(x_rot.min(), x_rot.max(), GRID_RES)
yi = np.linspace(y_rot.min(), y_rot.max(), GRID_RES)
Xi, Yi = np.meshgrid(xi, yi)

Zi_pred = griddata((x_rot, y_rot), mag_pred, (Xi, Yi), method="linear")
Zi_true = griddata((x_rot, y_rot), mag_true, (Xi, Yi), method="linear")

# NaN-mask outside the circular inlet disk
Ri = np.sqrt(Xi**2 + Yi**2)
Zi_pred[Ri > inlet_radius] = np.nan
Zi_true[Ri > inlet_radius] = np.nan

# ------------------------------------------------------------------------------
# Step 6 — Save surface arrays for later analysis
# ------------------------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "inlet_surface_X.npy"),             Xi)
np.save(os.path.join(OUT_DIR, "inlet_surface_Y.npy"),             Yi)
np.save(os.path.join(OUT_DIR, "inlet_profile_predicted.npy"),     Zi_pred)
np.save(os.path.join(OUT_DIR, "inlet_profile_ground_truth.npy"),  Zi_true)
np.save(os.path.join(OUT_DIR, "inlet_points_coords.npy"),         coords_inlet_nodes)
np.save(os.path.join(OUT_DIR, "inlet_points_mag_pred.npy"),       mag_pred)
np.save(os.path.join(OUT_DIR, "inlet_points_mag_true.npy"),       mag_true)
log("Saved inlet surface arrays (.npy)")
log("Saved raw inlet point data (.npy)")

# ------------------------------------------------------------------------------
# Step 7 — Shared axis limits
# ------------------------------------------------------------------------------
xmin, xmax = Xi.min(), Xi.max()
ymin, ymax = Yi.min(), Yi.max()
zmin = min(np.nanmin(Zi_pred), np.nanmin(Zi_true))
zmax = max(np.nanmax(Zi_pred), np.nanmax(Zi_true))
if abs(zmax - zmin) < 1e-14:
    zmax = zmin + 1e-12

# ------------------------------------------------------------------------------
# Step 8 — 3-D surface comparison plot (predicted vs ground-truth)
# ------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

fig  = plt.figure(figsize=(16, 7))
ax1  = fig.add_subplot(121, projection="3d")
ax2  = fig.add_subplot(122, projection="3d")

ax1.plot_surface(Xi, Yi, Zi_pred, cmap="jet", vmin=zmin, vmax=zmax,
                 linewidth=0, antialiased=True)
ax2.plot_surface(Xi, Yi, Zi_true, cmap="jet", vmin=zmin, vmax=zmax,
                 linewidth=0, antialiased=True)

ax1.set_title("Predicted Inlet Velocity Magnitude")
ax2.set_title("Ground Truth Inlet Velocity Magnitude")

for ax in [ax1, ax2]:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xlabel("Local inlet x")
    ax.set_ylabel("Local inlet y")
    ax.set_zlabel("|u|")

norm     = plt.Normalize(zmin, zmax)
mappable = cm.ScalarMappable(norm=norm, cmap="jet")
mappable.set_array([])
fig.colorbar(mappable, ax=ax1, shrink=0.6)
fig.colorbar(mappable, ax=ax2, shrink=0.6)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "inlet_surface_comparison.png")
plt.savefig(fig_path, dpi=300)
plt.close()
log(f"Inlet surface comparison → {fig_path}")

# ------------------------------------------------------------------------------
# Step 9 — Raw point scatter diagnostic (2-D)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sc1 = plt.scatter(x_rot, y_rot, c=mag_pred, cmap="jet", s=12)
plt.colorbar(sc1)
plt.axis("equal")
plt.title("Predicted inlet points")
plt.xlabel("Local x"); plt.ylabel("Local y")

plt.subplot(1, 2, 2)
sc2 = plt.scatter(x_rot, y_rot, c=mag_true, cmap="jet", s=12)
plt.colorbar(sc2)
plt.axis("equal")
plt.title("Ground-truth inlet points")
plt.xlabel("Local x"); plt.ylabel("Local y")

plt.tight_layout()
scatter_path = os.path.join(OUT_DIR, "inlet_point_scatter_comparison.png")
plt.savefig(scatter_path, dpi=300)
plt.close()
log(f"Inlet point scatter comparison → {scatter_path}")

# ------------------------------------------------------------------------------
# Step 10 — Quantitative inlet reconstruction error
# ------------------------------------------------------------------------------
valid  = ~np.isnan(Zi_pred) & ~np.isnan(Zi_true)
if np.any(valid) and np.linalg.norm(Zi_true[valid]) > 1e-14:
    rel_L2 = (np.linalg.norm(Zi_pred[valid] - Zi_true[valid])
              / np.linalg.norm(Zi_true[valid]))
else:
    rel_L2 = np.nan
log(f"Relative L2 error (inlet surface) = {rel_L2:.6e}")

# ==============================================================================
# OPTIMISATION HISTORY PLOT
# ==============================================================================
banner("PLOTTING OPTIMISATION HISTORY")
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle(
    f"Inverse IPCS — Two-Outlet | MASKED OBJ | ONE-TIME RAMP + {N_NS_PER_EPOCH} NS/epoch | "
    f"outlets={OUTLET_IDS} | gamma={GAMMA_REG:.1e} | {OPTIMIZER} LR={LR:.2e} | "
    f"N_adv={N_ADV_OBS} T_adv={T_ADV} BC=pulsed | mask_cells={n_masked_cells}/{mesh.num_cells()}",
    fontsize=8,
)
axes[0, 0].semilogy(range(1, EPOCHS+1), loss_history,   "r-o", ms=2, label="J_total")
axes[0, 0].semilogy(range(1, EPOCHS+1), loss_data_hist, "b--", lw=1.2, label="J_data (mask)")
axes[0, 0].semilogy(range(1, EPOCHS+1), loss_reg_hist,  "k-.", lw=1.2, label="J_reg")
for _di in range(1, LR_DECAY_STEPS+1):
    axes[0, 0].axvline(_di*LR_DECAY_EVERY, color="gray", ls=":", lw=0.8)
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Objective terms (log)"); axes[0, 0].grid(True, which="both")
axes[0, 0].legend()
axes[0, 1].plot(range(1, EPOCHS+1), umax_history, "b-o", ms=3, label="U_max (monitor)")
axes[0, 1].axhline(U_MAX_TRUE, color="k", ls="--", lw=1.5, label=f"ref={U_MAX_TRUE}")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("U_max")
axes[0, 1].set_title("Peak inlet velocity (monitor)"); axes[0, 1].legend()
axes[0, 1].grid(True)
axes[1, 0].semilogy(range(1, EPOCHS+1), grad_history, "g-s", ms=3)
axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("||dJ/du||_L2")
axes[1, 0].set_title("Gradient norm (log)"); axes[1, 0].grid(True, which="both")
axes[1, 1].plot(range(1, EPOCHS+1), qpred_history, "m-^", ms=2, label="Q_pred (monitor)")
axes[1, 1].axhline(Q_ref, color="k", ls="--", lw=1.5, label=f"Q_ref={Q_ref:.3e}")
axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Flow rate")
axes[1, 1].set_title("Inlet flow rate (monitor)"); axes[1, 1].legend()
axes[1, 1].grid(True)
axes[0, 2].semilogy(range(1, EPOCHS+1), lr_history, "c-D", ms=2)
for _di in range(1, LR_DECAY_STEPS+1):
    axes[0, 2].axvline(_di*LR_DECAY_EVERY, color="gray", ls=":", lw=0.8)
axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("Learning rate")
axes[0, 2].set_title("LR schedule (step decay)"); axes[0, 2].grid(True, which="both")
_rel_q = [abs(q - Q_ref)/(Q_ref + 1e-14) for q in qpred_history]
axes[1, 2].semilogy(range(1, EPOCHS+1), _rel_q, "k-o", ms=2)
axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_ylabel("Rel. error Q (monitor)")
axes[1, 2].set_title("Relative error Q (log, monitor)"); axes[1, 2].grid(True, which="both")
plt.tight_layout()
_pp = os.path.join(OUT_DIR, "optimisation_history.png")
plt.savefig(_pp, dpi=150)
plt.close("all")
log(f"Plot → {_pp}")
banner("DONE — Two-Outlet Inverse Complete (Masked Objective, Pulsed BC)")
