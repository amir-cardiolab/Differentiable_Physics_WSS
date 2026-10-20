"""
GPU advection–diffusion (ADE) time integrator with an exact discrete adjoint,
exposed to dolfin-adjoint as a single custom pyadjoint Block.

Why
---
In the original Diff_Physics.py the ADE is integrated with Crank–Nicolson for
N = final_idx steps using a LinearVariationalSolver. Within one epoch the
velocity and dt are fixed, so the system matrix is CONSTANT — yet it is
re-assembled at every step, and pyadjoint records N solve blocks and replays
them (re-assembling again) in the backward pass.

Here the matrices are built ONCE per epoch on the GPU (the u-dependent
advection matrix K(u) is assembled per cell on the GPU; M and L are constant
and come from FEniCS at setup), and the N forward steps plus the N adjoint
steps run as sparse linear algebra on the GPU (float64). The step matrix is
factorized ONCE per epoch with cuDSS (GPU sparse LU) and each step is a pair of
triangular solves; Jacobi-BiCGStab (PyTorch) is available as a fallback.

Discretisation (identical to the original script)
-------------------------------------------------
    (c^{n+1}-c^n)/dt + u.grad(c_mid) - D lap(c_mid) = 0,   c_mid = (c^{n+1}+c^n)/2
    c = 1 on the inlet (Dirichlet, applied dolfin-style: row -> identity)

    A_bc c^{n+1} = B_bc c^n + r,     A = M/dt + K(u)/2 + L/2,  B = M/dt - K(u)/2 - L/2
    K(u)_ij = int phi_i u.grad(phi_j),  L_ij = D int grad(phi_i).grad(phi_j)

Adjoint (derivation in README.md)
---------------------------------
    A~ mu^N = P dJ/dc^N,   A~ mu^n = P B^T mu^{n+1}       (A~ = A^T, bc rows -> identity)
    dJ/du [v] = - sum_n int v . ( mu^{n+1} grad(c_mid^n) ) dx

mu^{n+1} grad(c_mid^n) is (P1 x piecewise-constant) = exactly a DG1 vector field,
so the whole time-sum is accumulated on the GPU as one DG1 vector field eta and
the gradient is a single sparse mat-vec with the (V x DG1) mixed mass matrix.
This is the same discrete gradient pyadjoint computes (up to the linear-solver
tolerance), see verify_gradient.py.

Serial only (one MPI rank); 2-D triangles, CG1 concentration, P2 velocity.
"""
from __future__ import print_function

import time
import numpy as np
import torch

import fenics
from fenics import (VectorFunctionSpace, TestFunction, TrialFunction, Constant,
                    dot, grad, inner, dx)
# overloaded type: c^N must be a pyadjoint-aware Function
from fenics_adjoint import Function
from pyadjoint import Block
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating

from gpu_fem import (TriGeometry, p1_tables, p2_tables, p2_node_bary, cell_dofs,
                     check_dof_positions, petsc_to_scipy, CSRPattern)
from gpu_cudss import CuDSS


# =============================================================================
# helpers
# =============================================================================
class _TorchCSR(object):
    """Sparse CSR matrix on a torch device (+ inverse diagonal for Jacobi)."""

    def __init__(self, pattern, values, diag_pos=None):
        self.n = pattern.n
        self.M = pattern.torch_csr(values, values.device)
        self.dinv = None if diag_pos is None else 1.0 / values[diag_pos]

    def mv(self, x):
        return torch.mv(self.M, x)




_SHADOW = {}


def _shadow(b):
    key = (b.numel(), b.dtype, b.device)
    if key not in _SHADOW:
        g = torch.Generator().manual_seed(1234)
        _SHADOW[key] = torch.rand(b.numel(), generator=g, dtype=b.dtype).to(b.device) + 0.5
    return _SHADOW[key]


def bicgstab(A, b, x0, tol=1e-12, maxiter=500, check_every=1):
    """Right-Jacobi-preconditioned BiCGStab; scalars stay on-device (no host
    sync) except for the convergence check every `check_every` iterations.
    Returns (x, iterations)."""
    bnorm = torch.linalg.norm(b)
    bn = float(bnorm)
    if bn == 0.0:
        return torch.zeros_like(b), 0
    x = x0.clone()
    r = b - A.mv(x)
    if float(torch.linalg.norm(r)) <= tol * bn:
        return x, 0
    # fixed random shadow residual: r_hat = r0 breaks down here, because r0 of
    # the first step lives only on the (identity) Dirichlet rows, which are
    # satisfied after one iteration -> r_hat . r == 0 exactly
    r_hat = _shadow(b)
    one = torch.ones((), dtype=b.dtype, device=b.device)
    rho, alpha, omega = one, one, one
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)
    thresh = tol * bnorm
    for it in range(1, maxiter + 1):
        rho_new = torch.dot(r_hat, r)
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        phat = A.dinv * p
        v = A.mv(phat)
        alpha = rho_new / torch.dot(r_hat, v)
        s = r - alpha * v
        check = (it % check_every == 0)
        if check and bool(torch.linalg.norm(s) <= thresh):   # converged at half step
            return x + alpha * phat, it
        shat = A.dinv * s
        t = A.mv(shat)
        tt = torch.dot(t, t)
        omega = torch.where(tt > 0, torch.dot(t, s) / tt, torch.zeros_like(tt))
        x = x + alpha * phat + omega * shat
        r = s - omega * t
        rho = rho_new
        if check and bool(torch.linalg.norm(r) <= thresh):
            return x, it
    raise RuntimeError("BiCGStab did not converge in %d iterations (rel res %.3e)"
                       % (maxiter, float(torch.linalg.norm(r)) / bn))



# =============================================================================
# The operator (all mesh-only precomputation lives here, built once)
# =============================================================================
class GPUAdvectionDiffusion(object):
    """
    Parameters
    ----------
    S        : scalar CG1 FunctionSpace for c
    W        : mixed (P2 velocity, pressure) space of the NS solve; the block
               takes the NS solution w in W and uses its velocity component
    D, dt    : diffusivity / time step (floats or Constants)
    inlet    : string/SubDomain marking the c = 1 Dirichlet boundary
    solver   : "cudss" (GPU sparse LU, factor once per epoch) or "bicgstab"
    """

    def __init__(self, S, W, D, dt, inlet, device=None, dtype=torch.float64,
                 solver="cudss", tol=1e-12, maxiter=500, check_every=1, verbose=True,
                 check=True):
        if S.ufl_element().degree() != 1 or S.ufl_element().family() != "Lagrange":
            raise ValueError("gpu_ade supports a CG1 concentration space only")
        self.S, self.W = S, W
        self.mesh = mesh = S.mesh()
        self.D, self.dt = float(D), float(dt)
        self.device = dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.tol, self.maxiter, self.check_every = tol, maxiter, check_every
        if solver == "cudss" and (self.device.type != "cuda" or dtype != torch.float64):
            solver = "bicgstab"
        self.solver = solver
        self.verbose = verbose
        self.last_stats = {}

        with stop_annotating():
            geom = TriGeometry(mesh)
            E = geom.E
            # ---- velocity dofs of W per cell (P2) -----------------------------
            Wx = W.tabulate_dof_coordinates().reshape(-1, 2)
            vd = np.stack([cell_dofs(W.sub(0).sub(d).dofmap(), E) for d in range(2)], axis=2)
            for d in range(2):
                check_dof_positions(Wx, vd[:, :, d], geom, p2_node_bary(), "P2 velocity")
            self.V0, cmap = W.sub(0).collapse(collapsed_dofs=True)
            self.w_idx = np.array([cmap[k] for k in range(self.V0.dim())], dtype=np.int64)

            # ---- CG1 dofs per cell --------------------------------------------
            Sx = S.tabulate_dof_coordinates().reshape(-1, 2)
            self.cell_dofs = cd = cell_dofs(S.dofmap(), E)
            check_dof_positions(Sx, cd, geom, np.eye(3), "P1 concentration")
            P1, dP1ref = p1_tables()
            gradphi = geom.phys_grad(dP1ref)[:, 0]                      # (E,3,2), constant
            N2, _ = p2_tables()

            # ---- Dirichlet (c = 1) dofs ---------------------------------------
            bc = fenics.DirichletBC(S, Constant(1.0), inlet)
            self.bc_dofs = np.array(sorted(bc.get_boundary_values().keys()), dtype=np.int64)
            if self.bc_dofs.size == 0:
                raise RuntimeError("No Dirichlet dofs found for the ADE inlet")

            # ---- constant matrices + pattern ----------------------------------
            phi, ct = TestFunction(S), TrialFunction(S)
            Msp = petsc_to_scipy(fenics.assemble(phi * ct * dx))
            Lsp = petsc_to_scipy(fenics.assemble(Constant(self.D) * dot(grad(phi), grad(ct)) * dx))
            self.pat = pat = CSRPattern(Msp)
            if not (np.array_equal(Lsp.indptr, Msp.indptr) and np.array_equal(Lsp.indices, Msp.indices)):
                raise RuntimeError("M and L patterns differ")
            R = np.broadcast_to(cd[:, :, None], (E, 3, 3))
            C = np.broadcast_to(cd[:, None, :], (E, 3, 3))
            posK = pat.positions(R.reshape(-1), C.reshape(-1))
            permS = pat.transpose_perm()
            diagS = pat.diag_positions()
            bc_entries = pat.row_entry_positions(self.bc_dofs)

            # ---- DG1 vector space: eta = sum_n mu grad(c_mid) lives here ------
            Vdg = VectorFunctionSpace(mesh, "DG", 1)
            dg_idx = np.stack([cell_dofs(Vdg.sub(d).dofmap(), E) for d in range(2)], axis=2)
            xdg = Vdg.tabulate_dof_coordinates().reshape(-1, 2)
            for d in range(2):
                check_dof_positions(xdg, dg_idx[:, :, d], geom, np.eye(3), "DG1")
            self.n_dg = Vdg.dim()
            Mmix = petsc_to_scipy(fenics.assemble(inner(TestFunction(self.V0), TrialFunction(Vdg)) * dx))

            if check:
                self._K_form_check = (phi, ct)

        f = dict(dtype=dtype, device=dev)
        self._vd = torch.as_tensor(vd, device=dev)
        self._N2 = torch.as_tensor(N2, **f)                                # (Q,6)
        self._P1w = torch.as_tensor(P1, **f).unsqueeze(0) * torch.as_tensor(geom.wq, **f).unsqueeze(-1)  # (E,Q,3)
        self._cd = torch.as_tensor(cd, device=dev)
        self._gradphi = torch.as_tensor(gradphi, **f)
        self._posK = torch.as_tensor(posK, device=dev)
        self._permS = torch.as_tensor(permS, device=dev)
        self._diagS = torch.as_tensor(diagS, device=dev)
        self._bc_entries = torch.as_tensor(bc_entries, device=dev)
        self._bc_diag = torch.as_tensor(diagS[self.bc_dofs], device=dev)
        self._bc = torch.as_tensor(self.bc_dofs, device=dev)
        self._M = torch.as_tensor(Msp.data, **f)
        self._L = torch.as_tensor(Lsp.data, **f)
        self._r = torch.zeros(S.dim(), **f)
        self._r[self._bc] = 1.0
        self._dg_idx = torch.as_tensor(dg_idx.reshape(-1), device=dev)
        self._Mmix = torch.sparse_csr_tensor(
            torch.as_tensor(Mmix.indptr, dtype=torch.int64), torch.as_tensor(Mmix.indices, dtype=torch.int64),
            torch.as_tensor(Mmix.data, dtype=dtype), size=Mmix.shape).to(dev)
        self._w_idx = torch.as_tensor(self.w_idx, device=dev)

        if self.solver == "cudss":
            # persistent value buffers: cuDSS reads them in place (pattern fixed)
            self._ip32 = torch.as_tensor(pat.indptr, dtype=torch.int32, device=dev)
            self._ix32 = torch.as_tensor(pat.indices, dtype=torch.int32, device=dev)
            self._Avals = torch.zeros(pat.nnz, dtype=dtype, device=dev)
            self._Atvals = torch.zeros(pat.nnz, dtype=dtype, device=dev)
            self._lu_fwd = self._lu_adj = None

        if check:
            self._check_against_fenics()
        if verbose:
            print("[gpu_ade] device=%s dtype=%s solver=%s | nS=%d cells=%d V0=%d DG1=%d bc=%d"
                  % (self.device, str(dtype).replace("torch.", ""), self.solver, S.dim(), E,
                     self.V0.dim(), self.n_dg, self.bc_dofs.size))

    # -------------------------------------------------------------------------
    def _K_values(self, w):
        """Values of K(u)_ij = int phi_i u.grad(phi_j), assembled on the GPU."""
        uq = torch.einsum("qa,ead->eqd", self._N2, w[self._vd])          # u at qp (E,Q,2)
        adv = torch.einsum("eqd,ejd->eqj", uq, self._gradphi)            # u.grad(phi_j)
        Kloc = torch.einsum("eqi,eqj->eij", self._P1w, adv)
        K = torch.zeros_like(self._M)
        K.index_add_(0, self._posK, Kloc.reshape(-1))
        return K

    def _check_against_fenics(self):
        phi, ct = self._K_form_check
        rng = np.random.RandomState(1)
        wv = rng.rand(self.W.dim())
        with stop_annotating():
            wf = fenics.Function(self.W)
            wf.vector().set_local(wv)
            wf.vector().apply("insert")
            Kr = petsc_to_scipy(fenics.assemble(phi * dot(fenics.split(wf)[0], grad(ct)) * dx))
        Kg = self._K_values(torch.as_tensor(wv, dtype=self.dtype, device=self.device)).cpu().numpy()
        err = np.linalg.norm(Kg - Kr.data) / np.linalg.norm(Kr.data)
        if self.verbose:
            print("[gpu_ade] GPU vs FEniCS K(u) assembly rel err %.1e" % err)
        if err > 1e-10:
            raise RuntimeError("GPU ADE assembly does not match FEniCS")

    def _assemble_operators(self, w):
        """Build A_bc, B_bc, A~ (= A^T with bc rows -> I), B^T on the GPU."""
        self._sync()
        t0 = time.time()
        K = self._K_values(w)
        Mdt = self._M / self.dt
        A = Mdt + 0.5 * K + 0.5 * self._L
        B = Mdt - 0.5 * K - 0.5 * self._L
        A_bc, B_bc = A.clone(), B.clone()
        A_bc[self._bc_entries] = 0.0
        A_bc[self._bc_diag] = 1.0
        B_bc[self._bc_entries] = 0.0
        At = A[self._permS]
        At[self._bc_entries] = 0.0
        At[self._bc_diag] = 1.0
        ops = dict(A_bc=_TorchCSR(self.pat, A_bc, self._diagS),
                   B_bc=_TorchCSR(self.pat, B_bc),
                   At=_TorchCSR(self.pat, At, self._diagS),
                   Bt=_TorchCSR(self.pat, B[self._permS]))
        if self.solver == "cudss":
            self._Avals.copy_(A_bc)
            self._Atvals.copy_(At)
            if self._lu_fwd is None:   # symbolic analysis once (pattern never changes)
                self._lu_fwd = CuDSS(self._ip32, self._ix32, self._Avals, matching=False)
                self._lu_adj = CuDSS(self._ip32, self._ix32, self._Atvals, matching=False)
            self._lu_fwd.factorize()
            self._lu_adj.factorize()
        self._sync()
        self.last_stats["assemble_s"] = time.time() - t0
        return ops

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _solve(self, ops, which, rhs, x0):
        """Solve with A_bc (which='fwd') or A~ (which='adj'); returns (x, iterations)."""
        if self.solver == "cudss":
            return (self._lu_fwd if which == "fwd" else self._lu_adj).solve(rhs), 1
        A = ops["A_bc"] if which == "fwd" else ops["At"]
        return bicgstab(A, rhs, x0, self.tol, self.maxiter, self.check_every)

    # -------------------------------------------------------------------------
    def forward(self, w_vals, n_steps, progress=None):
        """Run N Crank–Nicolson steps from c^0 = 0. Returns (c^N numpy, state)."""
        w = torch.as_tensor(np.asarray(w_vals), dtype=self.dtype, device=self.device)
        ops = self._assemble_operators(w)
        t0 = time.time()
        nS = self.S.dim()
        traj = torch.empty((n_steps + 1, nS), dtype=self.dtype, device=self.device)
        traj[0].zero_()
        c = traj[0]
        iters = 0
        for n in range(n_steps):
            rhs = ops["B_bc"].mv(c) + self._r
            c, it = self._solve(ops, "fwd", rhs, c)
            iters += it
            traj[n + 1] = c
            if progress is not None:
                progress(n, n_steps)
        self._sync()
        self.last_stats.update(forward_s=time.time() - t0,
                               forward_avg_iters=iters / max(1, n_steps))
        return traj[n_steps].cpu().numpy(), dict(ops=ops, traj=traj, n_steps=n_steps)

    # -------------------------------------------------------------------------
    def adjoint(self, state, dJdc):
        """Given dJ/dc^N (assembled covector on S), return dJ/dw (covector on W)."""
        ops, traj, N = state["ops"], state["traj"], state["n_steps"]
        self._sync()
        t0 = time.time()
        cd, gradphi = self._cd, self._gradphi
        ncell, nloc, gdim = gradphi.shape

        rhs = torch.as_tensor(dJdc, dtype=self.dtype, device=self.device)
        rhs[self._bc] = 0.0
        mu, iters = self._solve(ops, "adj", rhs, torch.zeros_like(rhs))
        eta = torch.zeros((ncell, nloc, gdim), dtype=self.dtype, device=self.device)
        for n in range(N - 1, -1, -1):
            cm = 0.5 * (traj[n + 1] + traj[n])
            cmc = cm[cd]                                                # (ncell, nloc)
            gradc = gradphi[:, 0, :] * cmc[:, :1]                       # (ncell, gdim)
            for j in range(1, nloc):
                gradc = gradc + gradphi[:, j, :] * cmc[:, j:j + 1]
            eta += mu[cd].unsqueeze(-1) * gradc.unsqueeze(1)
            if n > 0:
                rhs = ops["Bt"].mv(mu)
                rhs[self._bc] = 0.0
                mu, it = self._solve(ops, "adj", rhs, mu)
                iters += it
        eta_dg = torch.zeros(self.n_dg, dtype=self.dtype, device=self.device)
        eta_dg[self._dg_idx] = eta.reshape(-1)
        g_W = torch.zeros(self.W.dim(), dtype=self.dtype, device=self.device)
        g_W[self._w_idx] = -torch.mv(self._Mmix, eta_dg)
        g_W = g_W.cpu().numpy()
        self.last_stats.update(adjoint_s=time.time() - t0,
                               adjoint_avg_iters=iters / max(1, N))
        return g_W


# =============================================================================
# pyadjoint glue
# =============================================================================
class ADEGPUBlock(Block):
    def __init__(self, op, w, c_out, n_steps, state):
        super(ADEGPUBlock, self).__init__()
        self.op, self.n_steps, self.state = op, n_steps, state
        self.add_dependency(w)   # pyadjoint 2019.1: pass the object
        self.add_output(c_out.create_block_variable())

    def __str__(self):
        return "ADEGPUBlock(N=%d)" % self.n_steps

    def recompute_component(self, inputs, block_variable, idx, prepared):
        w = inputs[0]
        cN, self.state = self.op.forward(w.vector().get_local(), self.n_steps)
        out = Function(self.op.S)
        out.vector().set_local(cN)
        out.vector().apply("insert")
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        dJdc = adj_inputs[0].get_local()
        g_W = self.op.adjoint(self.state, dJdc)
        out = Function(self.op.W).vector()
        out.set_local(g_W)
        out.apply("insert")
        return out


def ade_solve_gpu(op, w, n_steps, progress=None, name="C"):
    """Annotated drop-in for the original ADE time loop: returns c^N as a Function(S)."""
    cN, state = op.forward(w.vector().get_local(), n_steps, progress=progress)
    with stop_annotating():
        c = Function(op.S, name=name)
        c.vector().set_local(cN)
        c.vector().apply("insert")
    if annotate_tape():
        block = ADEGPUBlock(op, w, c, n_steps, state)
        get_working_tape().add_block(block)
    return c
