"""
Figures + metrics for a finished Diff_Physics_GPU.py run.

    python postprocess.py <OUT_DIR> [<OUT_DIR_2> ...]

For each results folder, writes into <OUT_DIR>/report/:
  training.png       objective (total / data / regulariser), flow-rate error, s/iteration
  inlet_profile.png  inlet velocity: ground truth vs reconstructed vs initial guess
  velocity.png       |u| ground truth, reconstructed, and |u_opt - u_gt|
  concentration.png  C target, reconstructed, and their difference at the final step
  metrics.json       all numbers quoted in the summary
With several folders, also writes compare_inlet.png into the first one.
"""
from __future__ import print_function
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from fenics import (Mesh, XDMFFile, HDF5File, VectorFunctionSpace, FunctionSpace, Function,
                    MeshFunction, CompiledSubDomain, Measure, assemble, inner, dx,
                    set_log_level, LogLevel, Expression)

set_log_level(LogLevel.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfs_data import find_data_dir
DATA_DIR = find_data_dir()
x_min_geo = -0.5
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

mesh = Mesh()
with XDMFFile(os.path.join(DATA_DIR, "mesh.xdmf")) as f:
    f.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
S = FunctionSpace(mesh, "CG", 1)
xy = mesh.coordinates()
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
bnd = MeshFunction("size_t", mesh, 1, 0)
CompiledSubDomain(f"near(x[0], {x_min_geo}, DOLFIN_EPS)").mark(bnd, 1)
ds_in = Measure("ds", domain=mesh, subdomain_data=bnd)(1)


def read_h5(path, space, name):
    fn = Function(space)
    with HDF5File(mesh.mpi_comm(), path, "r") as f:
        f.read(fn, name)
    return fn


u_gt = Function(V)
with XDMFFile(os.path.join(DATA_DIR, "velocity_field.xdmf")) as f:
    f.read_checkpoint(u_gt, "velocity", 0)

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
r = vtk.vtkGenericDataObjectReader()
r.SetFileName(os.path.join(DATA_DIR, "C_4000.vtk"))
r.Update()
o = r.GetOutput()
c_tgt = Function(S)
c_tgt.vector().set_local(vtk_to_numpy(o.GetPointData().GetArray("C")).ravel()[
    cKDTree(vtk_to_numpy(o.GetPoints().GetData())[:, :2]).query(S.tabulate_dof_coordinates().reshape(-1, 2))[1]])
c_tgt.vector().apply("insert")

ys_in = np.sort(xy[np.isclose(xy[:, 0], x_min_geo), 1])
yy = np.linspace(ys_in.min(), ys_in.max(), 300)
prof = lambda fn: np.array([fn((x_min_geo, y))[0] for y in yy])
p_gt = prof(u_gt)
Q_true = assemble(u_gt[0] * ds_in)
l2 = lambda e: float(np.sqrt(assemble(e * dx(domain=mesh))))


def field(ax, vals, title, cmap, vmin, vmax, band=None):
    t = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(title, fontsize=10, color=INK)
    for yb in (band or []):
        ax.axhline(yb, color=INK, linewidth=0.8, linestyle=":")
    plt.colorbar(t, ax=ax, fraction=0.025, pad=0.01)


def report(res_dir):
    out = os.path.join(res_dir, "report")
    os.makedirs(out, exist_ok=True)
    H = json.load(open(os.path.join(res_dir, "history.json")))
    hist, opt = H["history"], H["optimizer"]
    it = np.array([h["it"] for h in hist])
    J, Jd, Jr = (np.array([h[k] for h in hist]) for k in ("J", "J_data", "J_reg"))
    qerr, tit = np.array([h["Q_rel_err"] for h in hist]), np.array([h["time_s"] for h in hist])
    y1, y2 = 0.3, 0.4

    # ---- training history (one y-axis per panel)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].semilogy(it, J, color=INK, label="J total")
    ax[0].semilogy(it, Jd, color=BLUE, label="data misfit")
    ax[0].semilogy(it, Jr, color=ORANGE, label="regulariser")
    ax[0].set(title="Objective", xlabel="iteration")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(it, 100 * qerr, color=BLUE)
    ax[1].set(title="Inlet flow-rate error |Q−Q_true|/Q_true [%]", xlabel="iteration", ylim=(0, None))
    ax[2].plot(it, tit, color=BLUE)
    ax[2].set(title="Wall time per iteration [s]  (total %.1f min)" % (tit.sum() / 60),
              xlabel="iteration", ylim=(0, None))
    fig.suptitle(f"{opt.upper()} — {len(it)} iterations, all-GPU adjoint", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training.png"), dpi=160)
    plt.close(fig)

    # ---- fields
    ctrl = read_h5(os.path.join(res_dir, "inlet_control.h5"), V, "InletControl")
    u_opt = read_h5(os.path.join(res_dir, "velocity.h5"), V, "velocity")
    c_opt = read_h5(os.path.join(res_dir, "concentration.h5"), S, "C")
    mask = Expression("x[1] >= a && x[1] <= b ? 1.0 : 0.0", a=y1, b=y2, degree=0)
    p_opt = prof(ctrl)
    m = dict(
        optimizer=opt, iterations=int(len(it)), J=float(J[-1]), J_data=float(Jd[-1]), J_reg=float(Jr[-1]),
        Q_true=float(Q_true), Q_opt=float(assemble(ctrl[0] * ds_in)),
        inlet_profile_rel_L2=float(np.sqrt(np.trapz((p_opt - p_gt) ** 2, yy) / np.trapz(p_gt ** 2, yy))),
        velocity_rel_L2=l2(inner(u_opt - u_gt, u_opt - u_gt)) / l2(inner(u_gt, u_gt)),
        conc_rel_L2_band=l2(mask * (c_opt - c_tgt) ** 2) / l2(mask * c_tgt ** 2),
        conc_rel_L2_domain=l2((c_opt - c_tgt) ** 2) / l2(c_tgt ** 2),
        time_total_s=float(tit.sum()), time_per_iter_s=float(tit.mean()))
    m["Q_rel_err"] = abs(m["Q_opt"] - Q_true) / Q_true

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(p_gt, yy, "--", color=INK, label="ground truth")
    ax.plot(np.where((yy > yy[0] + 1e-9) & (yy < yy[-1] - 1e-9), 1.0, 0.0), yy, color=ORANGE,
            linewidth=1.5, label="initial guess")
    ax.plot(p_opt, yy, color=BLUE, label=f"{opt} ({len(it)} it.)")
    ax.set(xlabel="u_x at inlet (x = %.1f)" % x_min_geo, ylabel="y",
           title="Inlet profile — Q err %.2f%%, profile err %.1f%%"
                 % (100 * m["Q_rel_err"], 100 * m["inlet_profile_rel_L2"]))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "inlet_profile.png"), dpi=160)
    plt.close(fig)

    mag = lambda fn: np.linalg.norm(fn.compute_vertex_values(mesh).reshape(2, -1), axis=0)
    m_gt, m_op = mag(u_gt), mag(u_opt)
    top = max(m_gt.max(), m_op.max())
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.2))
    field(ax[0], m_gt, "|u| ground truth", "Blues", 0, top)
    field(ax[1], m_op, f"|u| reconstructed ({opt})", "Blues", 0, top)
    field(ax[2], np.linalg.norm((u_opt.compute_vertex_values(mesh) - u_gt.compute_vertex_values(mesh))
                                .reshape(2, -1), axis=0),
          "|u_opt − u_gt|   (domain rel. L2 error %.1f%%)" % (100 * m["velocity_rel_L2"]), "Oranges", 0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "velocity.png"), dpi=160)
    plt.close(fig)

    cg, co = c_tgt.compute_vertex_values(mesh), c_opt.compute_vertex_values(mesh)
    dmax = max(np.abs(co - cg).max(), 1e-12)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.2))
    field(ax[0], cg, "C target (step 4000)   ·· observation band y∈[%.1f, %.1f]" % (y1, y2), "Blues", 0, 1, (y1, y2))
    field(ax[1], co, f"C reconstructed ({opt})", "Blues", 0, 1, (y1, y2))
    field(ax[2], co - cg, "C_opt − C_target   (rel. L2: band %.2f%%, domain %.2f%%)"
          % (100 * m["conc_rel_L2_band"], 100 * m["conc_rel_L2_domain"]), "RdBu_r", -dmax, dmax, (y1, y2))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "concentration.png"), dpi=160)
    plt.close(fig)

    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(res_dir, json.dumps(m, indent=2))
    return opt, len(it), p_opt, m


runs = [report(d) for d in sys.argv[1:]]
if len(runs) > 1:
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot(p_gt, yy, "--", color=INK, label="ground truth")
    for (opt, n, p, m), col in zip(runs, (BLUE, ORANGE, AQUA)):
        ax.plot(p, yy, color=col, label=f"{opt} ({n} it.): Q err {100 * m['Q_rel_err']:.1f}%")
    ax.set(xlabel="u_x at inlet", ylabel="y", title="Reconstructed inlet profiles")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(sys.argv[1], "report", "compare_inlet.png"), dpi=160)
