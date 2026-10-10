"""Normalized residuals ``r_N = |r| / √diag(Ω)``, ``Ω = R − H G⁻¹ Hᵀ``.

The single implementation shared by the quality summary and the bad-data
(``rn_max``) test. ``diag(H G⁻¹ Hᵀ)`` is computed by sparse CHOLMOD with sparse
right-hand sides; without cvxopt — by a dense Cholesky factor of ``G`` and a
blocked triangular solve. Neither path forms an ``m × m`` matrix.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)


def normalized_residuals(
    r: np.ndarray,
    H: csr_matrix,
    sigma2: np.ndarray,
    rows_mask: np.ndarray | None = None,
) -> np.ndarray:
    """``r_N = |r| / √diag(Ω)``, where ``Ω = R − H G⁻¹ Hᵀ``, ``R = diag(sigma2)``.

    A numerically non-positive ``Ω_ii`` marks a non-redundant (critical)
    measurement: its ``r_N`` is ``inf``. ``diag(H G⁻¹ Hᵀ)`` comes from sparse
    CHOLMOD with sparse right-hand sides (:func:`hgh_diag_cholmod`); without
    cvxopt — dense Cholesky + blocked triangular solve
    (:func:`hgh_diag_dense`). A singular ``G`` gives ``inf`` everywhere.

    Args:
        rows_mask: (m,) bool — compute ``r_N`` only for these rows (the rest
            get ``nan``). ``G`` is always assembled from ALL rows of ``H``
            (every measurement weighs in the estimate); only the costly
            blocked solve is masked. ``None`` — all rows.
    """
    if r.size == 0:
        return np.array([], dtype=np.float64)

    H_csr = H.tocsr()
    m = H_csr.shape[0]
    row_idx = np.arange(m) if rows_mask is None else np.where(rows_mask)[0]
    try:
        HGH_diag = hgh_diag_cholmod(H_csr, sigma2, row_idx)
    except ImportError:
        HGH_diag = hgh_diag_dense(H_csr, sigma2, row_idx)
    if HGH_diag is None:
        return np.full_like(r, np.inf, dtype=np.float64)

    omega_diag = sigma2 - HGH_diag
    omega_diag = np.where(omega_diag > 1e-12, omega_diag, np.nan)
    rn = np.abs(r) / np.sqrt(omega_diag)
    if rows_mask is not None:
        # Unmasked rows stay nan (left out of any top list); masked rows with
        # Ω ≤ 0 become inf (non-redundant).
        bad = np.isnan(rn) & (np.asarray(rows_mask, dtype=bool))
        rn = np.where(bad, np.inf, rn)
        return rn
    return np.where(np.isnan(rn), np.inf, rn)


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
    from cvxopt import cholmod, spmatrix
    from scipy.sparse import diags, tril

    G = (H_csr.T @ diags(1.0 / sigma2) @ H_csr).tocsc()
    lower = tril(G).tocoo()
    A = spmatrix(lower.data, lower.row.astype(np.int64), lower.col.astype(np.int64), G.shape)
    hgh = np.full(H_csr.shape[0], np.nan, dtype=np.float64)
    HT = H_csr.T.tocsc()
    saved = cholmod.options.get("supernodal")
    cholmod.options["supernodal"] = 2
    try:
        F = cholmod.symbolic(A, uplo="L")
        try:
            cholmod.numeric(A, F)
        except ArithmeticError as exc:
            logger.warning("r_N: G is not positive definite — r_N = inf (%s)", exc)
            return None
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
    finally:
        if saved is None:
            cholmod.options.pop("supernodal", None)
        else:
            cholmod.options["supernodal"] = saved
    return hgh


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
