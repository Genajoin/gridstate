"""Small shared helpers used across gridstate modules.

Collects patterns that were previously copy-pasted per module: id -> position
maps over contract arrays, branch endpoint maps, and the sigma^2 floor guard
for building R^-1 diagonals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.constants import SIGMA2_FLOOR


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ["branch_endpoints_map", "floored_sigma2", "id_to_pos_map", "sparse_diag"]


def id_to_pos_map(ids: Iterable[int] | np.ndarray) -> dict[int, int]:
    """Map object id -> position (row index) in the given order.

    Duplicate ids resolve to the last occurrence (dict build order), which is
    the convention every former inline copy of this map relied on.
    """
    return {int(v): pos for pos, v in enumerate(np.asarray(ids).tolist())}


def branch_endpoints_map(branches_arr: np.ndarray) -> dict[int, tuple[int, int]]:
    """Map branch id -> (from_node, to_node) from a contract branches array."""
    return {
        int(i): (int(f), int(t))
        for i, f, t in zip(
            branches_arr["id"],
            branches_arr["from_node"],
            branches_arr["to_node"],
            strict=True,
        )
    }


def floored_sigma2(diag: np.ndarray) -> np.ndarray:
    """Copy of a variance diagonal with the ``SIGMA2_FLOOR`` guard applied.

    Guards ``1/sigma^2`` against overflow on near-zero variances; see
    :data:`gridstate.constants.SIGMA2_FLOOR` for the rationale behind the value.
    """
    sigma2 = np.asarray(diag, dtype=np.float64).copy()
    sigma2[sigma2 < SIGMA2_FLOOR] = SIGMA2_FLOOR
    return sigma2


def sparse_diag(diag: np.ndarray) -> csr_matrix:
    """Square CSR matrix with ``diag`` on the main diagonal."""
    n = diag.shape[0]
    idx = np.arange(n)
    return csr_matrix((diag, (idx, idx)), shape=(n, n))
