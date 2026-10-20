"""
Small GPU finite-element toolkit for affine triangles (P1 / P2 Lagrange),
used to assemble the u-dependent terms of NS and ADE on the GPU.

Everything that depends only on the mesh (reference basis tables, quadrature,
cell -> dof maps, CSR scatter positions) is computed once with FEniCS/numpy;
per-epoch / per-Newton-step work is pure torch.

Local dof ordering follows FIAT/FFC (checked against FEniCS dof coordinates):
  P1: vertices 0,1,2 (dolfin local vertex order)
  P2: vertices 0,1,2, then edge i (opposite vertex i) for i = 0,1,2
"""
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import torch

from fenics import as_backend_type

# 7-point, degree-5 rule on the reference triangle (barycentric coords, weights sum to 1)
_A1, _B1, _W1 = 0.059715871789770, 0.470142064105115, 0.132394152788506
_A2, _B2, _W2 = 0.797426985353087, 0.101286507323456, 0.125939180544827
QUAD_BARY = np.array([[1 / 3, 1 / 3, 1 / 3],
                      [_A1, _B1, _B1], [_B1, _A1, _B1], [_B1, _B1, _A1],
                      [_A2, _B2, _B2], [_B2, _A2, _B2], [_B2, _B2, _A2]])
QUAD_W = np.array([0.225, _W1, _W1, _W1, _W2, _W2, _W2])

# gradients of the barycentric coordinates w.r.t. reference coords (xi, eta)
_DL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


def p1_tables(bary=QUAD_BARY):
    """(Q,3) values and (Q,3,2) reference gradients of the P1 basis."""
    Q = bary.shape[0]
    return bary.copy(), np.broadcast_to(_DL, (Q, 3, 2)).copy()


def p2_tables(bary=QUAD_BARY):
    """(Q,6) values and (Q,6,2) reference gradients of the P2 basis."""
    L = bary
    Q = L.shape[0]
    N = np.empty((Q, 6))
    dN = np.empty((Q, 6, 2))
    for i in range(3):
        N[:, i] = L[:, i] * (2 * L[:, i] - 1)
        dN[:, i, :] = (4 * L[:, i] - 1)[:, None] * _DL[i]
    for k, (a, b) in enumerate([(1, 2), (0, 2), (0, 1)]):   # edge k opposite vertex k
        N[:, 3 + k] = 4 * L[:, a] * L[:, b]
        dN[:, 3 + k, :] = 4 * (L[:, b][:, None] * _DL[a] + L[:, a][:, None] * _DL[b])
    return N, dN


def p2_node_bary():
    """Barycentric coordinates of the 6 P2 nodes (for ordering checks)."""
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=float)


class TriGeometry(object):
    """Affine geometry of every cell: vertex coords, J^{-1}, quadrature weights."""

    def __init__(self, mesh):
        if mesh.geometry().dim() != 2 or mesh.ufl_cell().cellname() != "triangle":
            raise ValueError("gpu_fem supports 2-D triangle meshes only")
        self.cells = mesh.cells().astype(np.int64)                 # (E,3) local vertex order
        self.X = mesh.coordinates()[self.cells]                    # (E,3,2)
        Jm = np.stack([self.X[:, 1] - self.X[:, 0], self.X[:, 2] - self.X[:, 0]], axis=2)  # cols
        self.detJ = np.linalg.det(Jm)
        self.Jinv = np.linalg.inv(Jm)                              # (E,2,2): d xi_k / d x_m
        self.wq = 0.5 * np.abs(self.detJ)[:, None] * QUAD_W[None, :]  # (E,Q)
        self.E = self.cells.shape[0]

    def phys_grad(self, dNref):
        """(Q,n,2) reference gradients -> (E,Q,n,2) physical gradients."""
        return np.einsum("qak,ekm->eqam", dNref, self.Jinv)

    def points(self, bary):
        """(E,P,2) physical coordinates of barycentric points."""
        return np.einsum("pi,eid->epd", bary, self.X)


def cell_dofs(dofmap, E):
    return np.array([dofmap.cell_dofs(e) for e in range(E)], dtype=np.int64)


def check_dof_positions(space_coords, dofs, geom, node_bary, what):
    """Assert FEniCS dof coordinates match our local node positions."""
    ref = geom.points(node_bary)                                   # (E,n,2)
    if not np.allclose(space_coords[dofs], ref, atol=1e-10):
        raise RuntimeError("local dof ordering mismatch for " + what)


# =============================================================================
# CSR helpers
# =============================================================================
def petsc_to_scipy(A):
    indptr, indices, data = as_backend_type(A).mat().getValuesCSR()
    M = sp.csr_matrix((data, indices, indptr), shape=(A.size(0), A.size(1)))
    M.sort_indices()
    return M


class CSRPattern(object):
    """A fixed sparsity pattern with O(log nnz) (row, col) -> position lookup."""

    def __init__(self, M):
        M = M.tocsr()
        M.sort_indices()
        self.n = M.shape[0]
        self.indptr = M.indptr.astype(np.int64)
        self.indices = M.indices.astype(np.int64)
        self.rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))
        self.keys = self.rows * self.n + self.indices               # sorted
        self.nnz = self.indices.size

    def positions(self, rows, cols):
        q = np.asarray(rows, dtype=np.int64) * self.n + np.asarray(cols, dtype=np.int64)
        pos = np.searchsorted(self.keys, q)
        pos = np.minimum(pos, self.nnz - 1)
        if not np.array_equal(self.keys[pos], q):
            raise RuntimeError("entry not in sparsity pattern")
        return pos

    def transpose_perm(self):
        """perm such that data_T = data[perm] (requires a structurally symmetric pattern)."""
        return self.positions(self.indices, self.rows)

    def diag_positions(self):
        r = np.arange(self.n, dtype=np.int64)
        return self.positions(r, r)

    def row_entry_positions(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return np.concatenate([np.arange(self.indptr[r], self.indptr[r + 1]) for r in rows]) \
            if rows.size else np.zeros(0, dtype=np.int64)

    def torch_csr(self, values, device):
        """torch sparse CSR sharing this pattern (values: torch tensor on device)."""
        if not hasattr(self, "_t_indptr") or self._t_indptr.device != torch.device(device):
            self._t_indptr = torch.as_tensor(self.indptr, device=device)
            self._t_indices = torch.as_tensor(self.indices, device=device)
        return torch.sparse_csr_tensor(self._t_indptr, self._t_indices, values,
                                       size=(self.n, self.n))
