"""Kernels of the selected inversion of a sparse Cholesky factor (Takahashi).

Plain Python on purpose: :func:`gridstate.algebra.normalized_residuals` compiles
them with numba (the optional ``numba`` extra) and the tests run them as they
are on small matrices. Kept apart from the rest of the package so that the
optional Cython build leaves the module as plain Python (numba cannot compile a
compiled function). The kernels call no helper functions, so numba compiles
each of them on its own.

Both kernels take a lower-triangular factor ``L`` of ``P G Pᵀ = L Lᵀ`` in CSC
form (``colptr``/``rowind``/``lval``): row indices ascending within a column,
the diagonal entry first.
"""

from __future__ import annotations

import numpy as np


def _selinv_lower(
    n: int,
    colptr: np.ndarray,
    rowind: np.ndarray,
    lval: np.ndarray,
    z: np.ndarray,
) -> int:
    """Entries of ``Z = (L Lᵀ)⁻¹`` on the pattern of ``L`` (Takahashi recurrence).

    Fills ``z`` (same layout as ``lval``) column by column from the last one::

        Z_ij = −(1/L_jj) · Σ_{k ∈ S_j} Z_ik L_kj           (i ∈ S_j)
        Z_jj = 1/L_jj² − (1/L_jj) · Σ_{k ∈ S_j} L_kj Z_kj

    where ``S_j`` are the off-diagonal rows of column ``j``. Every ``Z_ik``
    with ``i, k ∈ S_j`` lies in the pattern of ``L`` when that pattern is
    closed under elimination (a filled Cholesky pattern, possibly with
    explicit zeros, is); it is read from column ``min(i, k)``, which was
    computed earlier.

    Returns:
        ``0`` on success; ``j + 1`` when column ``j`` needs an entry that is
        not in the pattern (the pattern is not closed; ``z`` is then
        incomplete and must not be used).
    """
    acc = np.zeros(n)
    for j in range(n - 1, -1, -1):
        p0 = colptr[j]
        p1 = colptr[j + 1]
        ljj = lval[p0]
        s = p1 - p0 - 1
        for t in range(s):
            acc[t] = 0.0
        # acc[t] = Σ_u Z[r_t, r_u] · L[r_u, j], r_t, r_u ∈ S_j. Each unordered
        # pair (u ≤ t) is read once, from column c = r_u, at row r_t ≥ c.
        for u in range(s):
            c = rowind[p0 + 1 + u]
            lu = lval[p0 + 1 + u]
            q = colptr[c]
            qe = colptr[c + 1]
            for t in range(u, s):
                r = rowind[p0 + 1 + t]
                while q < qe and rowind[q] < r:
                    q += 1
                if q == qe or rowind[q] != r:
                    return j + 1
                zrc = z[q]
                acc[t] += zrc * lu
                if t > u:
                    acc[u] += zrc * lval[p0 + 1 + t]
        dsum = 0.0
        for t in range(s):
            zij = -acc[t] / ljj
            z[p0 + 1 + t] = zij
            dsum += lval[p0 + 1 + t] * zij
        z[p0] = 1.0 / (ljj * ljj) - dsum / ljj
    return 0


def _quad_forms(
    rows: np.ndarray,
    h_ptr: np.ndarray,
    h_col: np.ndarray,
    h_data: np.ndarray,
    perm_inv: np.ndarray,
    colptr: np.ndarray,
    rowind: np.ndarray,
    z: np.ndarray,
    out: np.ndarray,
) -> int:
    """``out[i] = hᵢ G⁻¹ hᵢᵀ`` for every ``i`` in ``rows`` from the selected inverse.

    ``h_ptr``/``h_col``/``h_data`` is ``H`` in CSR form without duplicate
    column indices; ``perm_inv[k]`` is the position of state ``k`` in the
    factor ordering; ``z`` holds ``(L Lᵀ)⁻¹`` on the pattern of ``L``
    (:func:`_selinv_lower`). Only pairs of non-zero entries of a row are read;
    they belong to the pattern of ``G`` unless their contributions to ``G``
    cancel exactly. A row that needs an entry outside the pattern gets
    ``nan``.

    Returns:
        The number of rows set to ``nan``.
    """
    missing = 0
    for k in range(rows.size):
        i = rows[k]
        a0 = h_ptr[i]
        a1 = h_ptr[i + 1]
        acc = 0.0
        ok = True
        for a in range(a0, a1):
            ha = h_data[a]
            if ha == 0.0:
                continue
            pa = perm_inv[h_col[a]]
            acc += ha * ha * z[colptr[pa]]
            for b in range(a + 1, a1):
                hb = h_data[b]
                if hb == 0.0:
                    continue
                pb = perm_inv[h_col[b]]
                c = pa
                r = pb
                if pb < pa:
                    c = pb
                    r = pa
                lo = colptr[c] + 1
                hi = colptr[c + 1] - 1
                q = -1
                while lo <= hi:
                    mid = (lo + hi) >> 1
                    x = rowind[mid]
                    if x == r:
                        q = mid
                        break
                    if x < r:
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if q < 0:
                    ok = False
                    break
                acc += 2.0 * ha * hb * z[q]
            if not ok:
                break
        if ok:
            out[i] = acc
        else:
            out[i] = np.nan
            missing += 1
    return missing
