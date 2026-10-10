"""numba kernel of the Gram matrix ``G = Hᵀ·diag(w)·H`` on a fixed pattern.

Imported lazily by :mod:`gridstate.algorithms.kkt_solver`; importing this
module requires numba (the optional ``numba`` extra). Kept apart from the rest
of the package so that the optional Cython build leaves it as plain Python
(a compiled module cannot be decorated by numba).

The kernel reproduces scipy's sparse product ``(scale_csr_rows(H, w).T @ H)``
value for value: scipy computes column ``k`` of the product as
``Σ_j H[j, k] · (H[j, i] · w[j])`` accumulated over the rows ``j`` of column
``k`` in ascending order, and so does the kernel. numba compiles without
``fastmath``, so LLVM does not contract the multiply-add into an FMA; the
bit-for-bit agreement is checked at load time (see
``kkt_solver._load_gram_kernel``).

The JIT result is cached on disk (``cache=True``) next to this file or, when
that directory is not writable, in numba's user cache directory; set
``NUMBA_CACHE_DIR`` to choose the location (e.g. a writable volume in a
read-only container image).
"""

from __future__ import annotations

import numpy as np
from numba import njit


def _gram_lower(
    n: int,
    hc_ptr: np.ndarray,
    hc_row: np.ndarray,
    hc_slot: np.ndarray,
    h_ptr: np.ndarray,
    h_col: np.ndarray,
    h_data: np.ndarray,
    w: np.ndarray,
    g_ptr: np.ndarray,
    g_row: np.ndarray,
    out: np.ndarray,
) -> None:
    """Lower triangle of ``Hᵀ·diag(w)·H`` into the sorted CSC pattern ``g``.

    ``H`` is a canonical CSR matrix (``h_ptr``/``h_col``/``h_data``: sorted
    column indices, no duplicates); ``hc_ptr``/``hc_row`` is its CSC pattern
    with rows ascending and ``hc_slot`` the CSR data slot of each CSC entry.
    ``g_ptr``/``g_row`` must cover every lower-triangle position of the
    pattern of ``HᵀH``; positions without contributions get ``0.0``.

    Row ``j`` of ``H`` is sorted, so the entries ``i >= k`` of the row start
    at the slot of ``(j, k)`` itself.
    """
    sums = np.zeros(n)
    for k in range(n):
        for jj in range(hc_ptr[k], hc_ptr[k + 1]):
            j = hc_row[jj]
            q0 = hc_slot[jj]
            v = h_data[q0]
            wj = w[j]
            for q in range(q0, h_ptr[j + 1]):
                sums[h_col[q]] += v * (h_data[q] * wj)
        for p in range(g_ptr[k], g_ptr[k + 1]):
            i = g_row[p]
            out[p] = sums[i]
            sums[i] = 0.0


try:
    gram_lower = njit(cache=True)(_gram_lower)
except RuntimeError:  # no writable cache location: compile per process
    gram_lower = njit(_gram_lower)
