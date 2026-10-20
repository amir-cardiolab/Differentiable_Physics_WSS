"""
Steady incompressible Navier–Stokes (Taylor–Hood P2/P1, 2-D triangles) solved
entirely on the GPU, with its exact discrete adjoint, exposed to dolfin-adjoint
as one custom pyadjoint Block  (control: inlet velocity Function  ->  w = (u, p)).

    F(w) = int (grad(u) u).v + nu grad(u):grad(v) - div(v) p + q div(u)   (as in Diff_Physics.py)

* The Stokes part is linear: FEniCS assembles its matrix A0 ONCE (this also fixes
  the sparsity pattern of the Jacobian).
* The convective residual and Jacobian are computed per cell on the GPU
  (P2 basis, 7-point degree-5 quadrature = exact) and scattered into A0's pattern.
* Newton linear systems and the adjoint system are solved with cuDSS on the GPU
  (analysis once; numeric factorization per Newton step).

Adjoint (w_B = g on Dirichlet dofs B, R_I(w) = F_I(w) = 0 elsewhere):
    [J^T]_bc lambda = dJ/dw  with rhs_B = 0      (J^T with bc rows -> identity)
    dJ/dg          = dJ/dw_B - (J^T lambda)_B
and dJ/dg is mapped onto the dofs of the inlet-control Function (same P2 space,
so the Dirichlet values are exact copies of its dofs).
"""
from __future__ import print_function

import time
import numpy as np
import torch
from scipy.spatial import cKDTree

from fenics import (TestFunctions, split, inner, dot, grad, div, dx, derivative,
                    Constant)
import fenics
from fenics_adjoint import Function
from pyadjoint import Block
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating

from gpu_fem import (TriGeometry, p2_tables, p2_node_bary, cell_dofs,
                     check_dof_positions, petsc_to_scipy, CSRPattern)
from gpu_cudss import CuDSS


class GPUSteadyNS(object):
    """
    Parameters
    ----------
    W        : mixed P2-vector / P1 space
    V_ctrl   : P2 vector space of the inlet-control Function
    nu       : kinematic viscosity (float or Constant)
    inlet    : string marking the inlet (velocity = control)
    walls    : string marking no-slip walls (velocity = 0)
    outlet   : string marking the outlet (pressure = 0)
    """

    def __init__(self, W, V_ctrl, nu, inlet, walls, outlet, device="cuda",
                 rtol=1e-9, atol=1e-10, maxit=50, verbose=True, check=True):
        t0 = time.time()
        self.W, self.V_ctrl = W, V_ctrl
        self.device = torch.device(device)
        self.rtol, self.atol, self.maxit = rtol, atol, maxit
        self.verbose = verbose
        self.last_stats = {}
        self.n = n = W.dim()
        mesh = W.mesh()
        dev, f64 = self.device, torch.float64

        with stop_annotating():
            # ---- Stokes matrix A0 (= Jacobian at w = 0) and the pattern ----------
            w0 = fenics.Function(W)
            u, p = split(w0)
            v, q = TestFunctions(W)
            F = (inner(dot(grad(u), u), v) * dx + Constant(float(nu)) * inner(grad(u), grad(v)) * dx
                 - div(v) * p * dx + q * div(u) * dx)
            self._F_form, self._J_form, self._w_fe = F, derivative(F, w0), w0
            A0 = petsc_to_scipy(fenics.assemble(self._J_form))
            self.pat = pat = CSRPattern(A0)

            # ---- Dirichlet dofs ---------------------------------------------------
            def bdofs(sub, val, where):
                return np.array(sorted(fenics.DirichletBC(sub, val, where)
                                       .get_boundary_values().keys()), dtype=np.int64)
            self.in_dofs = bdofs(W.sub(0), Constant((0.0, 0.0)), inlet)
            wall_dofs = bdofs(W.sub(0), Constant((0.0, 0.0)), walls)
            out_dofs = bdofs(W.sub(1), Constant(0.0), outlet)
            self.bc_dofs = np.unique(np.concatenate([self.in_dofs, wall_dofs, out_dofs]))
            # dolfin applies [inlet, walls, outlet] in order; our sets must not overlap
            if np.intersect1d(self.in_dofs, wall_dofs).size:
                raise RuntimeError("inlet and wall Dirichlet dofs overlap")

            # ---- inlet W dofs -> control V dofs (same component, same point) ----
            Wx = W.tabulate_dof_coordinates().reshape(-1, 2)
            Vx = V_ctrl.tabulate_dof_coordinates().reshape(-1, 2)
            self.in_vdofs = np.empty_like(self.in_dofs)
            for d in range(2):
                wsub = np.array(W.sub(0).sub(d).dofmap().dofs(), dtype=np.int64)
                vsub = np.array(V_ctrl.sub(d).dofmap().dofs(), dtype=np.int64)
                sel = np.isin(self.in_dofs, wsub)
                dist, k = cKDTree(Vx[vsub]).query(Wx[self.in_dofs[sel]])
                if dist.max() > 1e-10:
                    raise RuntimeError("inlet dof matching failed")
                self.in_vdofs[sel] = vsub[k]

            # ---- per-cell velocity dofs & geometry --------------------------------
            geom = TriGeometry(mesh)
            E = geom.E
            vd = np.stack([cell_dofs(W.sub(0).sub(d).dofmap(), E) for d in range(2)], axis=2)  # (E,6,2)
            for d in range(2):
                check_dof_positions(Wx, vd[:, :, d], geom, p2_node_bary(), "P2 velocity")
            N, dNref = p2_tables()
            dN = geom.phys_grad(dNref)                                   # (E,Q,6,2)

            # ---- CSR scatter positions of the 12x12 cell blocks -------------------
            R = np.broadcast_to(vd[:, :, :, None, None], (E, 6, 2, 6, 2))
            C = np.broadcast_to(vd[:, None, None, :, :], (E, 6, 2, 6, 2))
            pos = pat.positions(R.reshape(-1), C.reshape(-1))
            perm = pat.transpose_perm()
            diag = pat.diag_positions()
            bc_entries = pat.row_entry_positions(self.bc_dofs)

        # ---- device tensors ---------------------------------------------------------
        self._vd = torch.as_tensor(vd, device=dev)
        self._vd_flat = self._vd.reshape(-1)
        self._N = torch.as_tensor(N, dtype=f64, device=dev)              # (Q,6)
        self._dN = torch.as_tensor(dN, dtype=f64, device=dev)            # (E,Q,6,2)
        self._wq = torch.as_tensor(geom.wq, dtype=f64, device=dev)       # (E,Q)
        self._pos = torch.as_tensor(pos, device=dev)
        self._perm = torch.as_tensor(perm, device=dev)
        self._bc = torch.as_tensor(self.bc_dofs, device=dev)
        self._in = torch.as_tensor(self.in_dofs, device=dev)
        self._bc_entries = torch.as_tensor(bc_entries, device=dev)
        self._bc_diag = torch.as_tensor(diag[self.bc_dofs], device=dev)
        self._A0 = torch.as_tensor(A0.data, dtype=f64, device=dev)
        self._A0_csr = pat.torch_csr(self._A0, dev)
        self._eye = torch.eye(2, dtype=f64, device=dev)

        ip32 = torch.as_tensor(pat.indptr, dtype=torch.int32, device=dev)
        ix32 = torch.as_tensor(pat.indices, dtype=torch.int32, device=dev)
        self._Jvals = torch.zeros(pat.nnz, dtype=f64, device=dev)       # cuDSS-owned buffers
        self._JTvals = torch.zeros(pat.nnz, dtype=f64, device=dev)
        self._Jvals.copy_(self._A0); self._JTvals.copy_(self._A0)
        self._apply_bc_rows(self._Jvals); self._apply_bc_rows(self._JTvals)
        self._fwd = CuDSS(ip32, ix32, self._Jvals)
        self._adj = CuDSS(ip32, ix32, self._JTvals)
        self.w_last = None

        if check:
            self._check_against_fenics()
        if verbose:
            print("[gpu_ns] device=%s | W=%d nnz(J)=%d cells=%d bc=%d (inlet %d) | setup %.1fs"
                  % (self.device, n, pat.nnz, E, self.bc_dofs.size, self.in_dofs.size,
                     time.time() - t0))

    # -------------------------------------------------------------------------
    # GPU assembly of the convective term
    # -------------------------------------------------------------------------
    def _conv(self, w, jacobian):
        """Cell residual (E,6,2) of (grad u) u . v and, optionally, its Jacobian (E,6,2,6,2)."""
        ue = w[self._vd]                                                 # (E,6,2)
        uq = torch.einsum("qa,ead->eqd", self._N, ue)                    # u at qp
        gq = torch.einsum("eqam,ead->eqdm", self._dN, ue)                # du_d/dx_m at qp
        conv = torch.einsum("eqdm,eqm->eqd", gq, uq)
        Nw = self._N.unsqueeze(0) * self._wq.unsqueeze(-1)               # (E,Q,6) weighted test
        res = torch.einsum("eqa,eqd->ead", Nw, conv)
        if not jacobian:
            return res, None
        adv = torch.einsum("eqm,eqbm->eqb", uq, self._dN)                # u . grad(N_b)
        A1 = torch.einsum("eqa,eqb->eab", Nw, adv)                       # (grad du) u term
        A2 = torch.einsum("eqa,eqdf,qb->eadbf", Nw, gq, self._N)         # (grad u) du term
        Jloc = A1[:, :, None, :, None] * self._eye[None, None, :, None, :] + A2
        return res, Jloc

    def residual(self, w):
        res, _ = self._conv(w, False)
        F = torch.mv(self._A0_csr, w)
        F.index_add_(0, self._vd_flat, res.reshape(-1))
        return F

    def jacobian_values(self, w):
        """Values (in the fixed pattern) of the full Jacobian dF/dw, no bc applied."""
        _, Jloc = self._conv(w, True)
        vals = self._A0.clone()
        vals.index_add_(0, self._pos, Jloc.reshape(-1))
        return vals

    def _apply_bc_rows(self, vals):
        vals[self._bc_entries] = 0.0
        vals[self._bc_diag] = 1.0

    # -------------------------------------------------------------------------
    def _check_against_fenics(self):
        """GPU residual/Jacobian vs FEniCS assemble at a random state (setup-time test)."""
        rng = np.random.RandomState(0)
        wv = rng.rand(self.n)
        with stop_annotating():
            self._w_fe.vector().set_local(wv)
            self._w_fe.vector().apply("insert")
            Fr = fenics.assemble(self._F_form).get_local()
            Jr = petsc_to_scipy(fenics.assemble(self._J_form))
        wt = torch.as_tensor(wv, device=self.device)
        Fg = self.residual(wt).cpu().numpy()
        Jg = self.jacobian_values(wt).cpu().numpy()
        eF = np.linalg.norm(Fg - Fr) / np.linalg.norm(Fr)
        eJ = np.linalg.norm(Jg - Jr.data) / np.linalg.norm(Jr.data)
        if self.verbose:
            print("[gpu_ns] GPU vs FEniCS assembly: residual rel err %.1e, Jacobian rel err %.1e"
                  % (eF, eJ))
        if eF > 1e-10 or eJ > 1e-10:
            raise RuntimeError("GPU NS assembly does not match FEniCS")

    # -------------------------------------------------------------------------
    def forward(self, g_ctrl, w_init=None):
        """Newton solve. g_ctrl: control dof values (numpy, V_ctrl). Returns w (torch)."""
        torch.cuda.synchronize()
        t0 = time.time()
        dev = self.device
        if w_init is None:
            w_init = self.w_last
        w = (w_init.clone() if w_init is not None
             else torch.zeros(self.n, dtype=torch.float64, device=dev))
        w[self._bc] = 0.0
        w[self._in] = torch.as_tensor(np.asarray(g_ctrl)[self.in_vdofs], dtype=torch.float64, device=dev)
        r0 = None
        t_fact = 0.0
        for it in range(self.maxit + 1):
            F = self.residual(w)
            F[self._bc] = 0.0
            rn = float(torch.linalg.norm(F))
            if r0 is None:
                r0 = rn
            if rn < self.atol or (r0 > 0 and rn / r0 < self.rtol):
                break
            if it == self.maxit:
                raise RuntimeError("GPU Newton did not converge (|F| = %.3e)" % rn)
            self._Jvals.copy_(self.jacobian_values(w))
            self._apply_bc_rows(self._Jvals)
            tf = time.time()
            self._fwd.factorize()
            dw = self._fwd.solve(-F)
            t_fact += time.time() - tf
            w += dw
        torch.cuda.synchronize()
        self.w_last = w.clone()
        self.last_stats.update(forward_s=time.time() - t0, newton_its=it,
                               factor_solve_s=t_fact, residual=rn)
        return w

    def adjoint(self, w, dJdw):
        """dJ/dw (numpy, W)  ->  dJ/d(control dofs) (numpy, V_ctrl)."""
        torch.cuda.synchronize()
        t0 = time.time()
        JT = self.jacobian_values(w)[self._perm]                       # full J^T, no bc
        self._JTvals.copy_(JT)
        self._apply_bc_rows(self._JTvals)
        self._adj.factorize()
        rhs = torch.as_tensor(dJdw, dtype=torch.float64, device=self.device)
        rhs_bc = rhs.clone()
        rhs_bc[self._bc] = 0.0
        lam = self._adj.solve(rhs_bc)
        y = torch.mv(self.pat.torch_csr(JT, self.device), lam)          # J^T lambda
        gB = (rhs[self._in] - y[self._in]).cpu().numpy()
        g = np.zeros(self.V_ctrl.dim())
        np.add.at(g, self.in_vdofs, gB)
        self.last_stats.update(adjoint_s=time.time() - t0)
        return g


# =============================================================================
# pyadjoint glue
# =============================================================================
class NSGPUBlock(Block):
    def __init__(self, op, u_ctrl, w_out):
        super(NSGPUBlock, self).__init__()
        self.op = op
        self.add_dependency(u_ctrl)          # pyadjoint 2019.1: pass the object
        self.add_output(w_out.create_block_variable())

    def __str__(self):
        return "NSGPUBlock"

    def recompute_component(self, inputs, block_variable, idx, prepared):
        w = self.op.forward(inputs[0].vector().get_local())
        out = Function(self.op.W)
        out.vector().set_local(w.cpu().numpy())
        out.vector().apply("insert")
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        w_saved = self.get_outputs()[0].saved_output.vector().get_local()
        w = torch.as_tensor(w_saved, device=self.op.device)
        g = self.op.adjoint(w, adj_inputs[0].get_local())
        out = Function(self.op.V_ctrl).vector()
        out.set_local(g)
        out.apply("insert")
        return out


def ns_solve_gpu(op, u_ctrl, name="w"):
    """Annotated steady NS solve on the GPU: returns w = (u, p) as a Function(W)."""
    w_t = op.forward(u_ctrl.vector().get_local())
    with stop_annotating():
        w = Function(op.W, name=name)
        w.vector().set_local(w_t.cpu().numpy())
        w.vector().apply("insert")
    if annotate_tape():
        get_working_tape().add_block(NSGPUBlock(op, u_ctrl, w))
    return w
