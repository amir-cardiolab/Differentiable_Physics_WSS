# Is velocity_field.xdmf the flow that generated C_4000.vtk and WSS.csv?
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
from fenics import *
from gpu_ns import GPUSteadyNS
from gpu_ade import GPUAdvectionDiffusion
set_log_level(LogLevel.WARNING)
from bfs_data import find_data_dir
fd = find_data_dir()
mesh = Mesh()
with XDMFFile(os.path.join(fd, "mesh.xdmf")) as f: f.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()])); S = FunctionSpace(mesh, "CG", 1)
u_gt = Function(V)
with XDMFFile(os.path.join(fd, "velocity_field.xdmf")) as f: f.read_checkpoint(u_gt, "velocity", 0)
xy = mesh.coordinates()
print("mesh x range", xy[:,0].min(), xy[:,0].max(), "y range", xy[:,1].min(), xy[:,1].max())
ys = np.sort(xy[np.isclose(xy[:,0], -0.5), 1]); print("inlet y range", ys.min(), ys.max())
yy = np.linspace(ys.min(), ys.max(), 11)
print("u_gt inlet u_x samples:", np.round([u_gt((-0.5, y))[0] for y in yy], 4))
print("u_gt max |u| (dofs):", np.abs(u_gt.vector().get_local()).max())
for xo in (4.0, 2.0):
    ys2 = np.linspace(0, 1, 11)
    print(f"u_gt u_x at x={xo}:", np.round([u_gt((xo, y))[0] for y in ys2], 4))

# ---- WSS from u_gt on the bottom wall y=0, x in [0.5, 4] vs WSS.csv (x shifted by +0.5)
d = np.loadtxt(os.path.join(HERE, "WSS.csv"), delimiter=",", skiprows=1)
mu = 0.04
Tg = TensorFunctionSpace(mesh, "DG", 1); G = project(grad(u_gt), Tg)
xs = np.linspace(0.55, 3.95, 18)
wss_gt = np.array([mu * G((x, 1e-9))[1] for x in xs])      # mu du/dy
wss_csv = np.interp(xs + 0.5, d[:,1][np.argsort(d[:,1])], d[:,0][np.argsort(d[:,1])])
print("\nWSS check (bottom wall), mu=%.3f" % mu)
print("   x   | mu*du/dy from u_gt | WSS.csv (x+0.5) | ratio")
for x, a, b in zip(xs, wss_gt, wss_csv):
    print(f" {x:5.2f} | {a:18.5f} | {b:15.5f} | {a/b if abs(b)>1e-3 else float('nan'):6.2f}")
m = np.abs(wss_csv) > 0.05
print("median ratio u_gt-WSS / csv-WSS: %.3f" % np.median(wss_gt[m] / wss_csv[m]))

# ---- concentration: which inlet scale reproduces C_4000?
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
r = vtk.vtkGenericDataObjectReader(); r.SetFileName(os.path.join(fd, "C_4000.vtk")); r.Update()
o = r.GetOutput(); vxy = vtk_to_numpy(o.GetPoints().GetData())[:, :2]; vC = vtk_to_numpy(o.GetPointData().GetArray("C")).ravel()
Ct = vC[cKDTree(vxy).query(S.tabulate_dof_coordinates().reshape(-1, 2))[1]]
inl, wal, out = "near(x[0], -0.5, DOLFIN_EPS)", "on_boundary && !(near(x[0], -0.5, DOLFIN_EPS) || near(x[0], 4.0, DOLFIN_EPS))", "near(x[0], 4.0, DOLFIN_EPS)"
ns = GPUSteadyNS(W, V, 0.04 / 1.06, inl, wal, out, check=False, verbose=False)
ade = GPUAdvectionDiffusion(S, W, 1e-3, 1e-3, inl, check=False, verbose=False)
Sx = S.tabulate_dof_coordinates().reshape(-1, 2)
band = (Sx[:, 1] >= 0.3) & (Sx[:, 1] <= 0.4)
def err(c):
    e = c - Ct
    return np.linalg.norm(e) / np.linalg.norm(Ct), np.linalg.norm(e[band]) / np.linalg.norm(Ct[band])
# (a) ADE driven directly by u_gt
w_gt = Function(W); assign(w_gt.sub(0), u_gt)
c, _ = ade.forward(w_gt.vector().get_local(), 4000)
print("\nC_4000 misfit with u_gt directly:  domain %.3e  band %.3e" % err(c))
# (b) NS with inlet = s * (u_gt inlet profile), s scan
print(" scale s | Q        | C misfit domain | C misfit band")
for s_ in (1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625):
    g = u_gt.vector().get_local() * s_
    w = ns.forward(g)
    c, _ = ade.forward(w.cpu().numpy(), 4000)
    ed, eb = err(c)
    print(f"  {s_:6.4f} | {2.6667*s_:8.4f} | {ed:15.4e} | {eb:13.4e}")
