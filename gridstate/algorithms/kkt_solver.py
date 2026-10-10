"""Solver for a sequence of Newton KKT systems (normal equations).

Within a single solve (WLS LM loop, IPM inner Newton) each iteration solves
``G·d = rhs`` with ``G = HᵀR⁻¹H (+ diagonal term)`` — a symmetric
positive definite matrix whose sparsity structure is practically unchanged between
iterations: only the values change (V/δ in the
trigonometry of H, huber weights in R⁻¹, the barrier diagonal). This makes it possible to
do the symbolic analysis once and afterwards only a numeric
refactorization — ``scipy.sparse.linalg.spsolve`` cannot do that (every
call = full analysis + LU).

The ``"cholmod"`` backend uses CHOLMOD from SuiteSparse via cvxopt
(manylinux wheels, a light optional dependency): ``cholmod.symbolic`` on the
pattern + ``cholmod.numeric`` per iteration, simplicial LDLᵀ mode
(``options['supernodal']=0``): late IPM iterations are quasi-definite
(barrier ~1e14 on the diagonal) and supernodal LLᵀ breaks down on them with a
reused fill order. On a 16k-node production model (339 systems, n=16k,
nnz=237k): 17.4 s spsolve → 1.2 s (×14). Fallback on a singular matrix (loss of
observability) — a single ``spsolve`` with a warning.

Pattern stabilisation pitfall: scipy prunes numerical zeros both in sparse
matmul and in addition, so the pattern of G jitters by a few entries between
iterations (236564..236569 nnz on that model), and projecting onto a superset
by adding a zero pattern matrix falls apart (zero sums are pruned too). The fix is a pattern matrix with data
``±PATTERN_EPS=1e-300``: the sum in the "holes" is non-zero (not pruned), and
for real values of G (≫1e-280) the addition lies beyond the precision of
double — the data is bit-for-bit the same.

Fixed-pattern Gram path (:meth:`KKTSolver.solve_normal`): within one solve the
Jacobian keeps its sparsity pattern (``JacobianPattern``), so the pattern of
``HᵀH`` is fixed as well. With the ``cholmod`` backend and numba installed
(the ``numba`` extra), ``G = Hᵀ·diag(w)·H`` is assembled by a numba kernel
straight into the sorted lower triangle of that pattern (CHOLMOD reads only
the lower triangle), and the growing superset pattern above is emulated with
a boolean mask, skipping scipy's sparse product, sums, sort and pattern
comparison. The values handed to CHOLMOD, the pattern growth and therefore
every symbolic/numeric factorization are bit-identical to the scipy path,
which remains the fallback (no numba, ``GRIDSTATE_NUMBA=0``, a Jacobian whose
pattern changes between calls, the ``scipy`` backend).
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, tril
from scipy.sparse.linalg import spsolve

from gridstate.utils import scale_csr_rows


logger = logging.getLogger(__name__)

# Environment switch of the numba Gram path: "0" disables it (scipy path).
NUMBA_ENV = "GRIDSTATE_NUMBA"

# Value of the superset pattern matrix: not 0.0 (pruned on addition), but
# well beyond the double precision of any real value of G.
PATTERN_EPS = 1e-300


def _cvxopt_available() -> bool:
    try:
        import cvxopt.cholmod  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str) -> str:
    """Resolve ``"auto"`` into a concrete backend depending on whether cvxopt is available.

    Args:
        backend: ``"auto"`` | ``"cholmod"`` | ``"scipy"``.

    Returns:
        ``"cholmod"`` or ``"scipy"``.

    Raises:
        ImportError: an explicit ``"cholmod"`` request without cvxopt installed.
        ValueError: unknown backend name.
    """
    if backend == "scipy":
        return "scipy"
    if backend == "cholmod":
        if not _cvxopt_available():
            raise ImportError(
                "kkt_solver='cholmod' requires cvxopt (pip install cvxopt "
                "or gridstate[fast]); use 'auto' for an automatic fallback"
            )
        return "cholmod"
    if backend == "auto":
        return "cholmod" if _cvxopt_available() else "scipy"
    raise ValueError(f"unknown kkt_solver: {backend!r}; available: 'auto', 'cholmod', 'scipy'")


def gram_scipy(
    H: csr_matrix,
    w: np.ndarray,
    diag: np.ndarray | None = None,
    HtW: Any = None,
) -> csc_matrix:
    """``Hᵀ·diag(w)·H (+ diag(diag))`` the way the solvers build it with scipy.

    ``HtW`` — precomputed ``scale_csr_rows(H, w).T`` (built here when omitted).
    """
    if HtW is None:
        HtW = scale_csr_rows(H, w).T
    G = (HtW @ H).tocsc()
    if diag is not None:
        G = G + diags(diag, format="csc")
    return G


GramKernel = Callable[..., None]


@functools.lru_cache(maxsize=1)
def _load_gram_kernel() -> GramKernel | None:
    """Import and JIT the numba Gram kernel; ``None`` when unusable.

    The kernel is compiled once per process (or loaded from numba's on-disk
    cache) and self-checked against the scipy product on a case where a fused
    multiply-add would change the last bit; any import, compilation or
    agreement failure disables the fast path.
    """
    try:
        from gridstate.algorithms._gram_numba import gram_lower
    except Exception as exc:  # ImportError (no numba) or a numba-side failure
        logger.debug("kkt: numba Gram kernel unavailable: %s", exc)
        return None
    try:
        ok = _self_check(gram_lower)
    except Exception as exc:
        logger.warning("kkt: numba Gram kernel failed to compile/run, scipy path used: %s", exc)
        return None
    if not ok:
        logger.warning("kkt: numba Gram kernel disagrees with scipy, scipy path used")
        return None
    kernel: GramKernel = gram_lower
    return kernel


def numba_gram_available() -> bool:
    """The numba Gram kernel is installed, compiled and agrees with scipy.

    The first call pays the JIT (or numba cache load) cost, so it doubles as a
    warm-up for long-running processes. ``GRIDSTATE_NUMBA=0`` still disables
    the path at solve time.
    """
    return _load_gram_kernel() is not None


def _self_check(kernel: GramKernel) -> bool:
    """Bitwise agreement of ``kernel`` with :func:`gram_scipy` on a tiny case.

    ``G[0, 0]`` accumulates ``-(1+2⁻²⁹) + (1+2⁻³⁰)²``: rounding the product
    first gives exactly ``0`` (scipy), a fused multiply-add gives ``2⁻⁶⁰``.
    """
    a = 1.0 + 2.0**-30
    H = csr_matrix(np.array([[1.0, 0.0], [a, 0.5], [0.0, 3.0]]))
    w = np.array([-(1.0 + 2.0**-29), 1.0, 0.75])
    pattern = _GramPattern(H)
    got = np.zeros((2, 2))
    got[pattern.g_row, pattern.g_col] = pattern.assemble(kernel, H, w, None)
    ref = tril(gram_scipy(H, w)).toarray()
    return bool(np.array_equal(got, ref) and ref[0, 0] == 0.0)


def _canonical_csr(H: Any) -> bool:
    """``H`` is a float64 CSR matrix with strictly increasing column indices per row."""
    if not isinstance(H, csr_matrix) or H.dtype != np.float64:
        return False
    if H.nnz == 0:
        return True
    increasing = np.diff(H.indices.astype(np.int64)) > 0
    # A row start may step down; mark the first entry of every non-empty row.
    first = H.indptr[:-1][np.diff(H.indptr) > 0]
    row_start = np.zeros(H.nnz, dtype=bool)
    row_start[first] = True
    return bool(np.all(increasing | row_start[1:]))


class _GramPattern:
    """Fixed sparsity structure of ``Hᵀ·diag(w)·H`` for one Jacobian pattern.

    Holds ``H``'s CSR and CSC index arrays and the sorted CSC pattern of the
    lower triangle of ``HᵀH`` plus the full diagonal (the barrier Hessian may
    land on any diagonal position).
    """

    def __init__(self, H: csr_matrix) -> None:
        n = int(H.shape[1])
        self.shape = (n, n)
        self.h_indptr = H.indptr.copy()
        self.h_indices = H.indices.copy()
        self.h_ptr = H.indptr.astype(np.int64)
        self.h_col = H.indices.astype(np.int64)
        # CSC view of H as a permutation of its CSR data slots.
        slots = csr_matrix(
            (np.arange(H.nnz, dtype=np.int64), H.indices, H.indptr), shape=H.shape
        ).tocsc()
        slots.sort_indices()
        self.hc_ptr = slots.indptr.astype(np.int64)
        self.hc_row = slots.indices.astype(np.int64)
        self.hc_slot = slots.data.astype(np.int64)
        ones = csr_matrix((np.ones(H.nnz), H.indices, H.indptr), shape=H.shape)
        eye = csc_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
        S = tril((ones.T @ ones).tocsc() + eye, format="csc")
        S.sort_indices()
        self.g_ptr = S.indptr.astype(np.int64)
        self.g_row = S.indices.astype(np.int64)
        self.g_col = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.g_ptr))
        on_diag = self.g_row == self.g_col
        self.diag_pos = np.empty(n, dtype=np.int64)
        self.diag_pos[self.g_col[on_diag]] = np.flatnonzero(on_diag)

    def matches(self, H: csr_matrix) -> bool:
        """``H`` has exactly the pattern this structure was built for."""
        if H.shape[1] != self.shape[0]:
            return False
        same_ptr = H.indptr is self.h_indptr or np.array_equal(H.indptr, self.h_indptr)
        return same_ptr and (
            H.indices is self.h_indices or np.array_equal(H.indices, self.h_indices)
        )

    def assemble(
        self, kernel: GramKernel, H: csr_matrix, w: np.ndarray, diag: np.ndarray | None
    ) -> np.ndarray:
        """Lower-triangle values of ``Hᵀ·diag(w)·H (+ diag(diag))`` on the pattern."""
        out = np.empty(self.g_row.size)
        kernel(
            self.shape[0],
            self.hc_ptr,
            self.hc_row,
            self.hc_slot,
            self.h_ptr,
            self.h_col,
            np.ascontiguousarray(H.data, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64),
            self.g_ptr,
            self.g_row,
            out,
        )
        if diag is not None:
            d = self.diag_pos
            out[d] = out[d] + diag
        return out


class KKTSolver:
    """Solver of ``G·d = rhs`` that reuses the symbolic factorization.

    One instance lives for one iterative loop (solve_wls / solve_ipm):
    it keeps the accumulating superset pattern and the CHOLMOD symbolic factor,
    refactorizing only the values until the pattern grows.

    The ``"scipy"`` backend is the former behaviour (spsolve), bit for bit.
    """

    def __init__(self, backend: str = "auto") -> None:
        self.backend = resolve_backend(backend)
        # superset pattern (csc with data=PATTERN_EPS), cvxopt matrix and
        # symbolic factor; they live while the pattern is stable.
        self._pattern: csc_matrix | None = None
        self._A: Any = None
        self._F: Any = None
        # Fixed-pattern Gram path (solve_normal). _mode records which
        # representation holds the superset pattern: None (nothing yet),
        # "gram" (_seen mask over _gram's lower pattern; _A holds the lower
        # triangle only) or "matrix" (_pattern, the scipy path). "matrix" is
        # final for the solver's lifetime.
        self._mode: str | None = None
        self._gram: _GramPattern | None = None
        self._seen: np.ndarray | None = None

    def solve_normal(
        self,
        H: csr_matrix,
        w: np.ndarray,
        rhs: np.ndarray,
        *,
        diag: np.ndarray | None = None,
        HtW: Any = None,
    ) -> np.ndarray:
        """Solve ``(Hᵀ·diag(w)·H + diag(diag))·d = rhs``.

        Bit-identical to ``solve(gram_scipy(H, w, diag, HtW), rhs)``. With
        the ``cholmod`` backend and numba available the matrix is assembled
        on the fixed pattern of ``H`` (see the module docstring). ``HtW`` —
        the caller's ``scale_csr_rows(H, w).T``, reused by the scipy path.
        """
        gram = self._gram_pattern_for(H)
        kernel = _load_gram_kernel() if gram is not None else None
        if gram is None or kernel is None:
            return self.solve(gram_scipy(H, w, diag, HtW), rhs)
        values = gram.assemble(kernel, H, w, diag)
        # Same stability cascade as solve(): cached pattern, fresh pattern,
        # then a one-off spsolve of the full matrix.
        try:
            return self._solve_gram_cholmod(gram, values, rhs)
        except ArithmeticError:
            pass
        try:
            self._seen = None
            return self._solve_gram_cholmod(gram, values, rhs)
        except ArithmeticError:
            logger.warning(
                "kkt cholmod: matrix is not PD even on a fresh analysis — falling back to spsolve"
            )
            G = gram_scipy(H, w, diag, HtW)
            return np.asarray(spsolve(G, rhs), dtype=np.float64).ravel()

    def _gram_pattern_for(self, H: Any) -> _GramPattern | None:
        """Fixed Gram structure for ``H`` when the numba path applies."""
        if self.backend != "cholmod" or self._mode == "matrix":
            return None
        if self._mode == "gram":
            assert self._gram is not None
            if self._gram.matches(H):
                return self._gram
            self._gram_to_matrix_mode()
            return None
        if os.environ.get(NUMBA_ENV, "1") == "0" or not _canonical_csr(H):
            return None
        if _load_gram_kernel() is None:
            return None
        self._gram = _GramPattern(H)
        self._mode = "gram"
        return self._gram

    def _solve_gram_cholmod(
        self, gram: _GramPattern, values: np.ndarray, rhs: np.ndarray
    ) -> np.ndarray:
        """``_solve_cholmod`` on lower-triangle values of the fixed pattern.

        ``_seen`` is the lower triangle of the superset pattern the scipy path
        accumulates: positions nonzero at least once since the last reset.
        Seen positions get ``+PATTERN_EPS`` exactly as ``G + _pattern`` does.
        """
        from cvxopt import cholmod, matrix, spmatrix

        nonzero = values != 0
        seen = self._seen
        if seen is None:
            mask = nonzero
            data = values[mask]
            changed = True
        else:
            changed = bool(np.any(nonzero & ~seen))
            mask = seen | nonzero if changed else seen
            data = np.where(seen, values + PATTERN_EPS, values)[mask]
        if changed:
            self._seen = mask
            self._A = spmatrix(data, gram.g_row[mask], gram.g_col[mask], gram.shape)
            self._F = cholmod.symbolic(self._A)
        else:
            self._A.V = matrix(data)
        cholmod.numeric(self._A, self._F)
        b = matrix(np.asarray(rhs, dtype=np.float64))
        cholmod.solve(self._F, b)
        return np.asarray(b, dtype=np.float64).ravel()

    def _gram_to_matrix_mode(self) -> None:
        """Hand the accumulated superset pattern over to the scipy path.

        The lower triangle carries over exactly; the upper one is mirrored
        from it. CHOLMOD reads only the lower triangle, so the upper part of
        ``_pattern`` can at most trigger an extra symbolic analysis of an
        unchanged lower pattern, which yields the same factorization.
        """
        gram, seen = self._gram, self._seen
        self._mode = "matrix"
        self._gram = None
        self._seen = None
        self._pattern = None
        self._A = None
        self._F = None
        if gram is None or seen is None:
            return
        from cvxopt import cholmod, spmatrix

        rows, cols = gram.g_row[seen], gram.g_col[seen]
        off = rows != cols
        r = np.concatenate([rows, cols[off]])
        c = np.concatenate([cols, rows[off]])
        pat = csc_matrix((np.full(r.size, PATTERN_EPS), (r, c)), shape=gram.shape)
        pat.sort_indices()
        coo = pat.tocoo()
        self._pattern = pat
        self._A = spmatrix(coo.data, coo.row.astype(np.int64), coo.col.astype(np.int64), pat.shape)
        self._F = cholmod.symbolic(self._A)

    def solve(self, G: csc_matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve ``G·d = rhs``; G is an SPD csc matrix.

        Stability cascade of the cholmod backend: late IPM iterations give
        quasi-definite G (barrier ~1e14 on the diagonal, rcond down to 1e-24),
        on which LLᵀ with a reused fill-order (AMD from an early matrix of the
        pattern) may break down into not-PD even though a fresh analysis of the current
        matrix succeeds. Hence: (1) retry with a pattern reset — the fresh
        ``symbolic`` lives on for the following systems of the tail of the
        μ schedule; (2) only then a single ``spsolve`` fallback
        (a truly singular G — loss of observability).

        Simplicial LDLᵀ (``options['supernodal']=0``) was REJECTED as a cure:
        the diagonal D without pivoting is noisy on the same quasi-definite
        systems, and IPM on a large model stalled (it=4 instead of 9 with a
        failure-free factorisation).
        """
        if self.backend == "scipy":
            return np.asarray(spsolve(G, rhs), dtype=np.float64).ravel()
        if self._mode == "gram":
            self._gram_to_matrix_mode()
        self._mode = "matrix"
        try:
            return self._solve_cholmod(G, rhs)
        except ArithmeticError:
            pass
        try:
            self._pattern = None  # retry: fresh symbolic from the current G
            return self._solve_cholmod(G, rhs)
        except ArithmeticError:
            logger.warning(
                "kkt cholmod: matrix is not PD even on a fresh analysis — falling back to spsolve"
            )
            return np.asarray(spsolve(G, rhs), dtype=np.float64).ravel()

    def _solve_cholmod(self, G: csc_matrix, rhs: np.ndarray) -> np.ndarray:
        from cvxopt import cholmod, matrix, spmatrix

        if self._pattern is None:
            P = G.copy()
            P.sort_indices()
            changed = True
        else:
            P = G + self._pattern
            # IMPORTANT: scipy does not guarantee that the indices of the result of an
            # addition are sorted — and the V-update below puts P.data into the cvxopt matrix,
            # which stores its data in canonical CSC order. Without sort_indices()
            # the values are scattered past their positions (a garbage matrix,
            # a cascade of not-PD). We sort BEFORE comparing with the pattern.
            P.sort_indices()
            changed = (
                P.nnz != self._pattern.nnz
                or not np.array_equal(P.indices, self._pattern.indices)
                or not np.array_equal(P.indptr, self._pattern.indptr)
            )
        if changed:
            pat = P.copy()
            pat.data = np.full_like(pat.data, PATTERN_EPS)
            self._pattern = pat
            coo = P.tocoo()
            self._A = spmatrix(
                coo.data,
                coo.row.astype(np.int64),
                coo.col.astype(np.int64),
                P.shape,
            )
            self._F = cholmod.symbolic(self._A)
        else:
            # The order of data in cvxopt CSC matches scipy canonical csc.
            self._A.V = matrix(P.data)
        cholmod.numeric(self._A, self._F)
        b = matrix(np.asarray(rhs, dtype=np.float64))
        cholmod.solve(self._F, b)
        return np.asarray(b, dtype=np.float64).ravel()


__all__ = [
    "NUMBA_ENV",
    "PATTERN_EPS",
    "KKTSolver",
    "gram_scipy",
    "numba_gram_available",
    "resolve_backend",
]
