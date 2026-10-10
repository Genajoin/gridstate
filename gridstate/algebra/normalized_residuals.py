"""Normalized residuals ``r_N = |r| / √diag(Ω)``, ``Ω = R − H G⁻¹ Hᵀ``.

The single implementation shared by the quality summary and the bad-data
(``rn_max``) test. Neither path forms an ``m × m`` matrix. ``diag(H G⁻¹ Hᵀ)``
is computed by one of two methods:

* ``"solve"`` (default): triangular solves with every row of ``H`` — sparse
  CHOLMOD with sparse right-hand sides; without cvxopt, a dense Cholesky factor
  of ``G`` and a blocked triangular solve.
* ``"selected_inversion"``: a hybrid. First the entries of ``G⁻¹`` on the
  pattern of the sparse Cholesky factor ``L`` (Takahashi recurrence), then one
  quadratic form per row of ``H``. Every pair of states coupled by a row of
  ``H`` is an entry of ``G`` and so lies in the filled pattern of ``L``; no
  per-row solve is needed. ``L`` is the CHOLMOD factor (the same symbolic and
  numeric factorization as ``"solve"``); without cvxopt it is assembled from a
  scipy SuperLU factorization with a symmetric ordering and diagonal pivots
  (``G = L D Lᵀ``). Then the rows whose ratio ``Ω_ii / σ²_i`` from the selected
  inverse is below ``refine_below`` (:data:`SELINV_REFINE_BELOW` by default),
  and the rows whose ``Ω_ii`` is at or below the critical cut-off
  (:data:`OMEGA_CUTOFF`), are recomputed by the ``"solve"`` path: per-row
  solves on a CHOLMOD factorization identical to the first one (extracting the
  factor consumes it, so ``G`` is factorized once more when any row needs a
  solve), or the dense fallback of ``"solve"`` without cvxopt. The kernels need numba (the optional ``numba`` extra); without
  numba, with ``GRIDSTATE_NUMBA=0`` or when no usable factor is available,
  ``"solve"`` is used for every row.

The selected inverse is less accurate than the per-row solves on an
ill-conditioned ``G``: the recurrence propagates the rounding through every
entry of ``G⁻¹`` on the factor pattern, and on a gain with a condition number
of ~1e14 it loses 1.5–2 more digits of ``(H G⁻¹ Hᵀ)_ii``. Where ``Ω_ii`` is a
small difference of two nearly equal numbers (``σ²_i ≈ (H G⁻¹ Hᵀ)_ii``, a
nearly critical measurement), the cancellation amplifies that rounding by
``σ²_i / Ω_ii``: ``r_N`` may be off by tens of per cent, and a critical
measurement may look redundant or vice versa. Hence the refinement: every row
whose selected-inversion ratio is below the threshold carries the ``"solve"``
value, and so does every row reported as critical (``r_N = inf`` only on a
``"solve"`` value). With the default threshold the rows left to the selected
inverse are off from ``"solve"`` by a small multiple of the rounding of
``(H G⁻¹ Hᵀ)_ii`` (relative ~1e-4 of ``r_N`` at a ratio of 1e-2 on such a
gain, less on better-conditioned ones). With ``refine_below=0`` only the
critical rows are recomputed: no false ``inf``, but a critical measurement
whose selected-inversion ratio came out positive keeps a finite ``r_N``.
Even ``"solve"`` loses the leading digits of a very small ratio;
:func:`compute_normalized_residuals` reports the ratio ``Ω_ii / σ²_i`` so that
callers can tell such measurements apart.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import numbers
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


logger = logging.getLogger(__name__)

#: Methods of ``diag(H G⁻¹ Hᵀ)`` (see the module docstring).
RN_METHODS = ("solve", "selected_inversion")

#: ``Ω_ii`` at or below this value marks a critical (non-redundant)
#: measurement: its ``r_N`` is ``inf``.
OMEGA_CUTOFF = 1e-12

#: Default ratio ``Ω_ii / σ²_i`` below which ``"selected_inversion"``
#: recomputes a row by per-row solves (see the module docstring).
SELINV_REFINE_BELOW = 1e-2


@dataclass(frozen=True)
class NormalizedResiduals:
    """Normalized residuals with the conditioning of their denominators.

    Attributes:
        rn: (m,) ``|r| / √Ω_ii``. ``inf`` for a numerically non-positive
            ``Ω_ii`` (critical measurement) and everywhere when ``G`` is
            singular; ``nan`` for rows left out by ``rows_mask``.
        omega_ratio: (m,) ``Ω_ii / σ²_i = 1 − (H G⁻¹ Hᵀ)_ii / σ²_i``, the
            fraction of the measurement variance left after the estimate
            (1: fully redundant, 0: critical). A small ratio means the
            computed ``Ω_ii`` (and ``r_N``) carries the rounding of
            ``(H G⁻¹ Hᵀ)_ii`` amplified by ``1 / ratio``. ``nan`` for rows
            left out by ``rows_mask`` and when ``G`` is singular.
    """

    rn: np.ndarray
    omega_ratio: np.ndarray


def normalized_residuals(
    r: np.ndarray,
    H: csr_matrix,
    sigma2: np.ndarray,
    rows_mask: np.ndarray | None = None,
    *,
    method: str = "solve",
    refine_below: float = SELINV_REFINE_BELOW,
) -> np.ndarray:
    """``r_N = |r| / √diag(Ω)``, where ``Ω = R − H G⁻¹ Hᵀ``, ``R = diag(sigma2)``.

    A numerically non-positive ``Ω_ii`` marks a non-redundant (critical)
    measurement: its ``r_N`` is ``inf``. A singular ``G`` gives ``inf``
    everywhere. See :func:`compute_normalized_residuals`.

    Args:
        rows_mask: (m,) bool — compute ``r_N`` only for these rows (the rest
            get ``nan``). ``G`` is always assembled from ALL rows of ``H``
            (every measurement weighs in the estimate); only the costly
            per-row part is masked. ``None`` — all rows.
        method: ``"solve"`` | ``"selected_inversion"`` (see the module
            docstring).
        refine_below: ``"selected_inversion"`` only: rows whose ratio
            ``Ω_ii / σ²_i`` from the selected inverse is below this value are
            recomputed by per-row solves. ``0`` disables this refinement;
            rows at or below the critical cut-off are recomputed regardless,
            ``1`` recomputes every row (see the module docstring). Ignored by
            ``"solve"``.
    """
    return compute_normalized_residuals(
        r, H, sigma2, rows_mask, method=method, refine_below=refine_below
    ).rn


def compute_normalized_residuals(
    r: np.ndarray,
    H: csr_matrix,
    sigma2: np.ndarray,
    rows_mask: np.ndarray | None = None,
    *,
    method: str = "solve",
    refine_below: float = SELINV_REFINE_BELOW,
) -> NormalizedResiduals:
    """``r_N`` together with the ratio ``Ω_ii / σ²_i`` of every row.

    Same arguments as :func:`normalized_residuals`. The ``rn`` values do not
    depend on whether the ratio is used.

    Raises:
        ValueError: unknown ``method`` or a negative / non-finite
            ``refine_below``.
    """
    check_rn_options(method, refine_below)
    if r.size == 0:
        empty = np.array([], dtype=np.float64)
        return NormalizedResiduals(rn=empty, omega_ratio=empty.copy())

    H_csr = H.tocsr()
    m = H_csr.shape[0]
    row_idx = np.arange(m) if rows_mask is None else np.where(rows_mask)[0]
    HGH_diag = _hgh_diag(H_csr, sigma2, row_idx, method, refine_below)
    if HGH_diag is None:
        return NormalizedResiduals(
            rn=np.full_like(r, np.inf, dtype=np.float64),
            omega_ratio=np.full(m, np.nan, dtype=np.float64),
        )

    omega_diag = sigma2 - HGH_diag
    omega_ratio = omega_diag / sigma2
    omega_diag = np.where(omega_diag > OMEGA_CUTOFF, omega_diag, np.nan)
    rn = np.abs(r) / np.sqrt(omega_diag)
    if rows_mask is not None:
        # Unmasked rows stay nan (left out of any top list); masked rows with
        # Ω ≤ 0 become inf (non-redundant).
        bad = np.isnan(rn) & (np.asarray(rows_mask, dtype=bool))
        rn = np.where(bad, np.inf, rn)
        return NormalizedResiduals(rn=rn, omega_ratio=omega_ratio)
    return NormalizedResiduals(rn=np.where(np.isnan(rn), np.inf, rn), omega_ratio=omega_ratio)


def check_rn_options(method: str, refine_below: float = SELINV_REFINE_BELOW) -> None:
    """Validate the options of :func:`compute_normalized_residuals`.

    Raises:
        ValueError: unknown ``method`` or a negative / non-finite
            ``refine_below``.
    """
    if method not in RN_METHODS:
        raise ValueError(f"unknown normalized residuals method {method!r}; available: {RN_METHODS}")
    value: object = refine_below  # checked at runtime whatever the annotation says
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            f"selected inversion refine_below must be a finite number >= 0, got {refine_below!r}"
        )


def _hgh_diag(
    H_csr: csr_matrix,
    sigma2: np.ndarray,
    row_idx: np.ndarray,
    method: str,
    refine_below: float = SELINV_REFINE_BELOW,
) -> np.ndarray | None:
    """``diag(H G⁻¹ Hᵀ)`` for ``row_idx`` by ``method``; ``None`` — ``G`` is singular."""
    if method == "selected_inversion":
        try:
            return hgh_diag_selinv(H_csr, sigma2, row_idx, refine_below=refine_below)
        except SelectedInversionUnavailableError as exc:
            logger.debug("r_N: selected inversion unavailable, per-row solves used (%s)", exc)
    try:
        return hgh_diag_cholmod(H_csr, sigma2, row_idx)
    except ImportError:
        return hgh_diag_dense(H_csr, sigma2, row_idx)


def _gain(H_csr: csr_matrix, sigma2: np.ndarray) -> csc_matrix:
    """``G = Hᵀ R⁻¹ H`` (CSC)."""
    from scipy.sparse import diags

    return (H_csr.T @ diags(1.0 / sigma2) @ H_csr).tocsc()


def _cholmod_lower(G: csc_matrix) -> Any:
    """Lower triangle of ``G`` as a cvxopt sparse matrix (CHOLMOD input)."""
    from cvxopt import spmatrix
    from scipy.sparse import tril

    lower = tril(G).tocoo()
    return spmatrix(lower.data, lower.row.astype(np.int64), lower.col.astype(np.int64), G.shape)


@contextlib.contextmanager
def _supernodal_cholmod() -> Iterator[Any]:
    """``cvxopt.cholmod`` with a supernodal ``LLᵀ`` factorization (option restored)."""
    from cvxopt import cholmod

    saved = cholmod.options.get("supernodal")
    cholmod.options["supernodal"] = 2
    try:
        yield cholmod
    finally:
        if saved is None:
            cholmod.options.pop("supernodal", None)
        else:
            cholmod.options["supernodal"] = saved


def hgh_diag_cholmod(
    H_csr: csr_matrix, sigma2: np.ndarray, row_idx: np.ndarray, block: int = 16384
) -> np.ndarray | None:
    """``diag(H G⁻¹ Hᵀ)`` for ``row_idx`` via sparse CHOLMOD (cvxopt).

    ``G = P Lᵀ L Pᵀ`` (supernodal LLᵀ); ``hᵢᵀ G⁻¹ hᵢ = ‖L⁻¹ Pᵀ hᵢ‖²``. The
    right-hand sides ``hᵢ`` stay sparse (``spsolve``), so the cost follows the
    fill of ``L⁻¹ hᵢ`` rather than ``n_state × m``: an order of magnitude
    faster than the dense triangular solve on large networks, and it does not
    depend on the BLAS thread count.

    Raises:
        ImportError: cvxopt is not installed.

    Returns:
        ``None`` when ``G`` is not positive definite.
    """
    import cvxopt  # noqa: F401  (ImportError before assembling G)

    A = _cholmod_lower(_gain(H_csr, sigma2))
    hgh = np.full(H_csr.shape[0], np.nan, dtype=np.float64)
    with _supernodal_cholmod() as cholmod:
        F = _cholmod_numeric(cholmod, A)
        if F is None:
            return None
        _cholmod_hgh_rows(cholmod, F, H_csr, row_idx, hgh, block)
    return hgh


def _cholmod_numeric(cholmod: Any, A: Any) -> Any | None:
    """Symbolic and numeric CHOLMOD factorization; ``None`` — not positive definite."""
    F = cholmod.symbolic(A, uplo="L")
    try:
        cholmod.numeric(A, F)
    except ArithmeticError as exc:
        logger.warning("r_N: G is not positive definite — r_N = inf (%s)", exc)
        return None
    return F


def _cholmod_hgh_rows(
    cholmod: Any,
    F: Any,
    H_csr: csr_matrix,
    row_idx: np.ndarray,
    hgh: np.ndarray,
    block: int = 16384,
) -> None:
    """Write ``‖L⁻¹ Pᵀ hᵢ‖²`` of the rows ``row_idx`` into ``hgh`` (CHOLMOD factor ``F``)."""
    from cvxopt import spmatrix

    HT = H_csr.T.tocsc()
    for s in range(0, row_idx.size, block):
        sel = row_idx[s : s + block]
        rhs = HT[:, sel].tocoo()
        B = spmatrix(
            rhs.data,
            rhs.row.astype(np.int64),
            rhs.col.astype(np.int64),
            (HT.shape[0], sel.size),
        )
        Y = cholmod.spsolve(F, cholmod.spsolve(F, B, sys=7), sys=4)  # L⁻¹ · Pᵀ · B
        values = np.asarray(Y.V).ravel()
        colptr = np.asarray(Y.CCS[0]).ravel()
        sq = np.zeros(sel.size, dtype=np.float64)
        filled = colptr[1:] > colptr[:-1]
        if values.size:
            sq[filled] = np.add.reduceat(values * values, colptr[:-1][filled])
        hgh[sel] = sq


def hgh_diag_dense(H_csr: csr_matrix, sigma2: np.ndarray, row_idx: np.ndarray) -> np.ndarray | None:
    """``diag(H G⁻¹ Hᵀ)`` for ``row_idx`` via dense Cholesky + blocked trsm.

    Fallback without cvxopt. Returns ``None`` when ``G`` is singular.
    """
    from scipy.linalg import cho_factor, solve_triangular
    from scipy.sparse import diags

    G = np.asarray((H_csr.T @ diags(1.0 / sigma2) @ H_csr).todense())
    try:
        L, lower = cho_factor(G, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        logger.warning("r_N: G is singular — r_N = inf (%s)", exc)
        return None
    hgh = np.full(H_csr.shape[0], np.nan, dtype=np.float64)
    # Blocks of H rows bound the memory of the dense RHS (n_state × block).
    block = 4096
    for s in range(0, row_idx.size, block):
        sel = row_idx[s : s + block]
        Y = solve_triangular(L, H_csr[sel, :].toarray().T, lower=lower, check_finite=False)
        hgh[sel] = np.einsum("ij,ij->j", Y, Y)
    return hgh


# ---------------------------------------------------------------------------
# Selected inversion
# ---------------------------------------------------------------------------


class SelectedInversionUnavailableError(RuntimeError):
    """The selected inversion cannot be used here; the caller uses per-row solves."""


@dataclass(frozen=True)
class SelinvKernels:
    """The two kernels of :mod:`gridstate.algebra._selinv_kernels` (compiled or not)."""

    selinv_lower: Callable[..., int]
    quad_forms: Callable[..., int]


@dataclass(frozen=True)
class CholeskyFactor:
    """``P G Pᵀ = L Lᵀ``: ``L`` lower CSC (rows ascending, diagonal first).

    ``perm_inv[k]`` is the position of state ``k`` in the factor ordering.
    """

    colptr: np.ndarray
    rowind: np.ndarray
    lval: np.ndarray
    perm_inv: np.ndarray

    @property
    def n(self) -> int:
        return int(self.colptr.size - 1)


def _numba_switched_off() -> bool:
    from gridstate.algorithms.kkt_solver import NUMBA_ENV

    return os.environ.get(NUMBA_ENV, "1") == "0"


@functools.lru_cache(maxsize=1)
def _load_selinv_kernels() -> SelinvKernels | None:
    """numba-compiled selected inversion kernels; ``None`` when unusable.

    Compiled once per process (or loaded from numba's on-disk cache, see
    ``NUMBA_CACHE_DIR``) and checked against a dense inverse on a small
    matrix; any import, compilation or agreement failure disables the path.
    """
    try:
        from numba import njit

        from gridstate.algebra import _selinv_kernels as k
    except Exception as exc:  # ImportError (no numba) or a numba-side failure
        logger.debug("r_N: numba unavailable for the selected inversion: %s", exc)
        return None
    try:
        try:
            kernels = SelinvKernels(
                selinv_lower=njit(cache=True)(k._selinv_lower),
                quad_forms=njit(cache=True)(k._quad_forms),
            )
        except RuntimeError:  # no writable cache location: compile per process
            kernels = SelinvKernels(
                selinv_lower=njit(k._selinv_lower), quad_forms=njit(k._quad_forms)
            )
        ok = _self_check(kernels)
    except Exception as exc:
        logger.warning("r_N: selected inversion kernels failed to compile/run: %s", exc)
        return None
    if not ok:
        logger.warning("r_N: selected inversion kernels disagree with a dense inverse")
        return None
    return kernels


def python_selinv_kernels() -> SelinvKernels:
    """The kernels as plain Python (reference and tests; slow on large models)."""
    from gridstate.algebra import _selinv_kernels as k

    return SelinvKernels(selinv_lower=k._selinv_lower, quad_forms=k._quad_forms)


def _self_check(kernels: SelinvKernels) -> bool:
    """``diag(H G⁻¹ Hᵀ)`` of a small dense case against ``numpy.linalg.inv``."""
    H = csr_matrix(
        np.array(
            [
                [2.0, -1.0, 0.0, 0.0],
                [0.0, 1.5, 0.5, 0.0],
                [0.0, 0.0, 1.0, -2.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 3.0],
            ]
        )
    )
    sigma2 = np.array([1.0, 0.5, 2.0, 1.5, 1.0, 0.25])
    rows = np.arange(H.shape[0])
    got = _selinv_hgh(_dense_lower_factor(_gain(H, sigma2)), H, rows, kernels)
    Hd = H.toarray()
    G = Hd.T @ np.diag(1.0 / sigma2) @ Hd
    ref = np.einsum("ij,ji->i", Hd, np.linalg.inv(G) @ Hd.T)
    return bool(got is not None and np.allclose(got, ref, rtol=1e-12, atol=0.0))


def _dense_lower_factor(G: csc_matrix) -> CholeskyFactor:
    """Natural-order Cholesky factor of a small ``G`` with a full lower pattern."""
    n = G.shape[0]
    L = np.linalg.cholesky(G.toarray())
    rows, cols = np.tril_indices(n)
    order = np.lexsort((rows, cols))
    rows, cols = rows[order], cols[order]
    colptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(cols, minlength=n), out=colptr[1:])
    return CholeskyFactor(
        colptr=colptr,
        rowind=rows.astype(np.int64),
        lval=np.ascontiguousarray(L[rows, cols]),
        perm_inv=np.arange(n, dtype=np.int64),
    )


def _selinv_hgh(
    factor: CholeskyFactor, H_csr: csr_matrix, row_idx: np.ndarray, kernels: SelinvKernels
) -> np.ndarray | None:
    """Quadratic forms on the selected inverse; ``None`` — the pattern is not closed.

    Rows whose entries are missing from the pattern stay ``nan``.
    """
    z = np.empty_like(factor.lval)
    if kernels.selinv_lower(factor.n, factor.colptr, factor.rowind, factor.lval, z) != 0:
        return None
    Hc = H_csr.tocsr(copy=True)
    Hc.sum_duplicates()
    hgh = np.full(Hc.shape[0], np.nan, dtype=np.float64)
    kernels.quad_forms(
        np.ascontiguousarray(row_idx, dtype=np.int64),
        Hc.indptr.astype(np.int64),
        Hc.indices.astype(np.int64),
        np.ascontiguousarray(Hc.data, dtype=np.float64),
        factor.perm_inv,
        factor.colptr,
        factor.rowind,
        z,
        hgh,
    )
    return hgh


def cholmod_factor(G: csc_matrix) -> CholeskyFactor | None:
    """CHOLMOD factor of ``G`` (the factorization used by :func:`hgh_diag_cholmod`).

    Raises:
        ImportError: cvxopt is not installed.
        SelectedInversionUnavailableError: the factor is not in the expected
            form.

    Returns:
        ``None`` when ``G`` is not positive definite.
    """
    A = _cholmod_lower(G)
    with _supernodal_cholmod() as cholmod:
        F = _cholmod_numeric(cholmod, A)
        if F is None:
            return None
        return _factor_of_cholmod(cholmod, F, G.shape[0])


def _factor_of_cholmod(cholmod: Any, F: Any, n: int) -> CholeskyFactor:
    """:class:`CholeskyFactor` of a numeric CHOLMOD factor ``F``.

    ``cholmod.getfactor`` moves the numeric values out of ``F``: ``F`` is
    left symbolic and cannot be used for solves afterwards.

    Raises:
        SelectedInversionUnavailableError: the factor is not in the expected
            form.
    """
    from cvxopt import matrix

    # x = P b: perm[k] is the state at position k of the factor ordering.
    b = matrix(np.arange(n, dtype=np.float64))
    cholmod.solve(F, b, sys=7)
    perm = np.asarray(b, dtype=np.float64).ravel().astype(np.int64)
    L = cholmod.getfactor(F)
    colptr = np.asarray(L.CCS[0], dtype=np.int64).ravel()
    rowind = np.asarray(L.CCS[1], dtype=np.int64).ravel()
    lval = np.asarray(L.CCS[2], dtype=np.float64).ravel()
    perm_inv = np.empty(n, dtype=np.int64)
    perm_inv[perm] = np.arange(n, dtype=np.int64)
    factor = _checked_factor(colptr, rowind, lval, perm_inv)
    if factor is None:
        raise SelectedInversionUnavailableError("CHOLMOD factor is not in the expected form")
    return factor


def superlu_factor(G: csc_matrix) -> CholeskyFactor:
    """``L √D`` from a SuperLU ``G = L D Lᵀ`` (symmetric ordering, diagonal pivots).

    The no-cvxopt source of the factor. ``SymmetricMode`` with a zero pivot
    threshold keeps the pivots on the diagonal, so for a symmetric positive
    definite ``G`` the row and column permutations coincide and
    ``U = D Lᵀ``.

    Raises:
        SelectedInversionUnavailableError: the factorization failed or is not of
            that form (e.g. ``G`` is not positive definite); the caller then
            uses the dense path, which also decides about singularity.
    """
    from scipy.sparse.linalg import splu

    try:
        lu = splu(
            G,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
    except RuntimeError as exc:  # singular
        raise SelectedInversionUnavailableError(f"SuperLU failed: {exc}") from exc
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise SelectedInversionUnavailableError("SuperLU chose off-diagonal pivots")
    d = lu.U.diagonal()
    if not np.all(d > 0.0) or not np.all(np.isfinite(d)):
        raise SelectedInversionUnavailableError("G is not positive definite")
    L = csc_matrix(lu.L)
    L.sort_indices()
    lval = L.data * np.repeat(np.sqrt(d), np.diff(L.indptr))
    # Pr·G·Pc = L·U with Pc[i, perm_c[i]] = 1: state i sits at position perm_c[i].
    factor = _checked_factor(
        L.indptr.astype(np.int64),
        L.indices.astype(np.int64),
        np.ascontiguousarray(lval, dtype=np.float64),
        np.asarray(lu.perm_c, dtype=np.int64),
    )
    if factor is None:
        raise SelectedInversionUnavailableError("SuperLU factor is not in the expected form")
    return factor


def _checked_factor(
    colptr: np.ndarray, rowind: np.ndarray, lval: np.ndarray, perm_inv: np.ndarray
) -> CholeskyFactor | None:
    """The factor when every column starts with its diagonal and rows ascend."""
    n = colptr.size - 1
    starts = colptr[:-1]
    if colptr[-1] != rowind.size or np.any(np.diff(colptr) < 1):
        return None
    if not np.array_equal(rowind[starts], np.arange(n)):
        return None
    step = np.diff(rowind)
    step[colptr[1:-1] - 1] = 1  # column boundaries
    if np.any(step <= 0):
        return None
    return CholeskyFactor(colptr=colptr, rowind=rowind, lval=lval, perm_inv=perm_inv)


def hgh_diag_selinv(
    H_csr: csr_matrix,
    sigma2: np.ndarray,
    row_idx: np.ndarray,
    *,
    kernels: SelinvKernels | None = None,
    refine_below: float = SELINV_REFINE_BELOW,
) -> np.ndarray | None:
    """``diag(H G⁻¹ Hᵀ)`` for ``row_idx``: selected inversion refined by per-row solves.

    The selected inverse of the factor of ``G`` gives every row; the rows
    returned by :func:`_rows_to_refine` (ratio ``Ω_ii / σ²_i`` below
    ``refine_below``, ``Ω_ii`` at or below :data:`OMEGA_CUTOFF`, or a pair of
    states missing from the factor pattern because their entry of ``G``
    cancelled to an exact zero) are then recomputed as :func:`hgh_diag_cholmod`
    does, on an identical CHOLMOD factorization. Without cvxopt the factor is SuperLU's
    (:func:`superlu_factor`) and the refined rows come from
    :func:`hgh_diag_dense`, the ``"solve"`` path without cvxopt.

    Args:
        kernels: kernels to run; ``None`` — the numba-compiled ones.
        refine_below: ratio ``Ω_ii / σ²_i`` below which a row is recomputed;
            ``0`` recomputes only the rows at the cut-off and outside the
            pattern.

    Raises:
        SelectedInversionUnavailableError: no compiled kernels (numba missing or
            switched off with ``GRIDSTATE_NUMBA=0``) or no usable factor; the
            caller uses per-row solves.

    Returns:
        ``None`` when ``G`` is not positive definite.
    """
    if kernels is None:
        if _numba_switched_off():
            raise SelectedInversionUnavailableError("numba switched off")
        kernels = _load_selinv_kernels()
        if kernels is None:
            raise SelectedInversionUnavailableError("numba kernels unavailable")
    G = _gain(H_csr, sigma2)
    try:
        A = _cholmod_lower(G)
    except ImportError:
        return _hgh_selinv_superlu(G, H_csr, sigma2, row_idx, kernels, refine_below)
    with _supernodal_cholmod() as cholmod:
        F = _cholmod_numeric(cholmod, A)
        if F is None:
            return None
        factor = _factor_of_cholmod(cholmod, F, G.shape[0])
    hgh = _selinv_hgh(factor, H_csr, row_idx, kernels)
    if hgh is None:
        raise SelectedInversionUnavailableError("factor pattern is not closed under elimination")
    refine = _rows_to_refine(hgh, sigma2, row_idx, refine_below)
    if refine.size:
        # Extracting the factor leaves F symbolic: factorize again, exactly as
        # hgh_diag_cholmod does (a deterministic factorization, so the refined
        # rows carry the "solve" values).
        with _supernodal_cholmod() as cholmod:
            F = _cholmod_numeric(cholmod, A)
            if F is None:
                return None
            _cholmod_hgh_rows(cholmod, F, H_csr, refine, hgh)
    return hgh


def _hgh_selinv_superlu(
    G: csc_matrix,
    H_csr: csr_matrix,
    sigma2: np.ndarray,
    row_idx: np.ndarray,
    kernels: SelinvKernels,
    refine_below: float,
) -> np.ndarray | None:
    """:func:`hgh_diag_selinv` without cvxopt: SuperLU factor, dense refinement."""
    hgh = _selinv_hgh(superlu_factor(G), H_csr, row_idx, kernels)
    if hgh is None:
        raise SelectedInversionUnavailableError("factor pattern is not closed under elimination")
    refine = _rows_to_refine(hgh, sigma2, row_idx, refine_below)
    if refine.size:
        rest = hgh_diag_dense(H_csr, sigma2, refine)
        if rest is None:
            return None
        hgh[refine] = rest[refine]
    return hgh


def _rows_to_refine(
    hgh: np.ndarray, sigma2: np.ndarray, row_idx: np.ndarray, refine_below: float
) -> np.ndarray:
    """Rows of ``row_idx`` whose selected-inversion value is recomputed by solves.

    A row is recomputed when its ``(H G⁻¹ Hᵀ)_ii`` is missing (outside the
    factor pattern), when ``Ω_ii / σ²_i < refine_below`` (the cancellation in
    ``Ω_ii`` amplifies the rounding of the selected inverse by
    ``σ²_i / Ω_ii``), and when ``Ω_ii ≤ OMEGA_CUTOFF``, so that the critical
    decision of :func:`compute_normalized_residuals` is taken on a solve.
    """
    values = hgh[row_idx]
    s2 = sigma2[row_idx]
    omega = s2 - values
    with np.errstate(invalid="ignore"):
        refine = np.isnan(values) | (omega <= OMEGA_CUTOFF) | (omega / s2 < refine_below)
    rows: np.ndarray = row_idx[refine]
    if rows.size:
        logger.debug("r_N: %d of %d rows recomputed by per-row solves", rows.size, row_idx.size)
    return rows
