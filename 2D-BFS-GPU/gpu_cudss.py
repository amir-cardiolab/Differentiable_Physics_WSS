"""
Minimal ctypes binding to NVIDIA cuDSS (GPU sparse direct solver, v0.8) for
float64 CSR systems whose sparsity pattern never changes:

    s = CuDSS(indptr, indices, values)     # torch cuda tensors; analysis once
    values.copy_(new_values); s.factorize() # numeric factorization on the GPU
    x = s.solve(b)

The library comes from the pip wheel `nvidia-cudss-cu12` (installed by
cluster/setup_env.sbatch). nvmath-python would wrap this, but its cuDSS support
needs Python >= 3.10 and FEniCS 2019 pins us to 3.9.
"""
from __future__ import print_function

import ctypes
import glob
import os
import site

import torch

# enums from cudss_data_types.h (v0.8) / library_types.h
CUDA_R_64F, CUDA_R_32I = 1, 10
CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO = 0, 0, 0
CUDSS_LAYOUT_COL_MAJOR = 0
PHASE_ANALYSIS, PHASE_FACTORIZATION, PHASE_REFACTORIZATION = 3, 4, 8
PHASE_SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
CONFIG_MATCHING_ALG, CONFIG_IR_N_STEPS = 3, 5
DATA_INFO = 0
_STATUS = ["SUCCESS", "NOT_INITIALIZED", "ALLOC_FAILED", "INVALID_VALUE",
           "NOT_SUPPORTED", "EXECUTION_FAILED", "INTERNAL_ERROR", "IR_FAILED"]

_lib = None


def _find(pattern):
    for d in site.getsitepackages() + [site.getusersitepackages()]:
        hits = sorted(glob.glob(os.path.join(d, "nvidia", "*", "lib", pattern)))
        if hits:
            return hits[0]
    raise OSError("cannot find %s (pip install nvidia-cudss-cu12)" % pattern)


def _load():
    global _lib
    if _lib is None:
        # cuDSS needs cuBLAS 12 (torch itself may use a CUDA-11 cuBLAS; both coexist)
        for name in ("libcublasLt.so.12", "libcublas.so.12"):
            ctypes.CDLL(_find(name), mode=ctypes.RTLD_GLOBAL)
        _lib = ctypes.CDLL(_find("libcudss.so.0"))
    return _lib


def _check(status, what):
    if status != 0:
        raise RuntimeError("cuDSS %s failed: %s" % (what, _STATUS[status] if status < len(_STATUS) else status))


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


class CuDSS(object):
    def __init__(self, indptr, indices, values, matching=True, ir_steps=0):
        """indptr/indices: int32 cuda tensors; values: float64 cuda tensor that the
        caller updates IN PLACE before each factorize()."""
        lib = _load()
        assert indptr.dtype == torch.int32 and indices.dtype == torch.int32
        assert values.dtype == torch.float64 and values.is_cuda
        self.n = indptr.numel() - 1
        self.indptr, self.indices, self.values = indptr, indices, values  # keep alive
        dev = values.device
        self.b = torch.zeros(self.n, dtype=torch.float64, device=dev)
        self.x = torch.zeros(self.n, dtype=torch.float64, device=dev)

        self.handle, self.config, self.data = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        _check(lib.cudssCreate(ctypes.byref(self.handle)), "create")
        _check(lib.cudssConfigCreate(ctypes.byref(self.config)), "config")
        _check(lib.cudssDataCreate(self.handle, ctypes.byref(self.data)), "data")
        if matching:   # max-product matching + scaling: robust for zero pressure diagonal
            v = ctypes.c_int(6)   # CUDSS_MATCHING_ALG_AUTO
            _check(lib.cudssConfigSet(self.config, CONFIG_MATCHING_ALG, ctypes.byref(v),
                                      ctypes.sizeof(v)), "config matching")
        if ir_steps:
            v = ctypes.c_int(ir_steps)
            _check(lib.cudssConfigSet(self.config, CONFIG_IR_N_STEPS, ctypes.byref(v),
                                      ctypes.sizeof(v)), "config ir")

        self.A, self.X, self.B = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        nnz = indices.numel()
        _check(lib.cudssMatrixCreateCsr(
            ctypes.byref(self.A), ctypes.c_int64(self.n), ctypes.c_int64(self.n), ctypes.c_int64(nnz),
            _ptr(indptr), None, _ptr(indices), _ptr(values),
            CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO), "matrix csr")
        for M, buf in ((self.X, self.x), (self.B, self.b)):
            _check(lib.cudssMatrixCreateDn(ctypes.byref(M), ctypes.c_int64(self.n), ctypes.c_int64(1),
                                           ctypes.c_int64(self.n), _ptr(buf), CUDA_R_64F,
                                           CUDSS_LAYOUT_COL_MAJOR), "matrix dense")
        torch.cuda.synchronize()
        self._exec(PHASE_ANALYSIS, "analysis")
        self._factored = False

    def _exec(self, phase, what):
        _check(_load().cudssExecute(self.handle, phase, self.config, self.data,
                                    self.A, self.X, self.B), what)

    def info(self):
        v, w = ctypes.c_int(0), ctypes.c_size_t(0)
        _check(_load().cudssDataGet(self.handle, self.data, DATA_INFO, ctypes.byref(v),
                                    ctypes.sizeof(v), ctypes.byref(w)), "info")
        return v.value

    def factorize(self):
        torch.cuda.synchronize()          # values were written by torch
        self._exec(PHASE_FACTORIZATION, "factorization")
        torch.cuda.synchronize()
        inf = self.info()
        if inf != 0:
            raise RuntimeError("cuDSS factorization info = %d (singular / zero pivot)" % inf)
        self._factored = True

    def solve(self, b):
        assert self._factored
        self.b.copy_(b)
        torch.cuda.synchronize()
        self._exec(PHASE_SOLVE, "solve")
        torch.cuda.synchronize()
        return self.x.clone()

    def __del__(self):
        try:
            lib = _load()
            for M in (self.A, self.X, self.B):
                lib.cudssMatrixDestroy(M)
            lib.cudssDataDestroy(self.handle, self.data)
            lib.cudssConfigDestroy(self.config)
            lib.cudssDestroy(self.handle)
        except Exception:
            pass
