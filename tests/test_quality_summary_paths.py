"""Sparse (CHOLMOD) and dense paths of ``diag(H G⁻¹ Hᵀ)`` agree."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from gridstate.quality_summary import _hgh_diag_cholmod, _hgh_diag_dense


pytest.importorskip("cvxopt")


@pytest.mark.parametrize("seed", [0, 1])
def test_cholmod_path_matches_dense_path(seed: int) -> None:
    rng = np.random.default_rng(seed)
    m, n = 400, 120
    H = sparse_random(m, n, density=0.03, random_state=seed, format="csr")
    H = H.tolil()
    for j in range(n):  # every state is observed, G is positive definite
        H[j, j] = 1.0 + rng.random()
    H = H.tocsr()
    sigma2 = rng.uniform(0.5, 2.0, m)
    rows = np.sort(rng.choice(m, 150, replace=False))

    sparse = _hgh_diag_cholmod(H, sigma2, rows, block=64)
    dense = _hgh_diag_dense(H, sigma2, rows)
    assert sparse is not None and dense is not None
    np.testing.assert_allclose(sparse[rows], dense[rows], rtol=1e-10, atol=1e-12)
    assert np.isnan(np.delete(sparse, rows)).all()


def test_singular_gain_returns_none() -> None:
    H = sparse_random(10, 4, density=0.0, format="csr")  # no information at all
    assert _hgh_diag_cholmod(H, np.ones(10), np.arange(10)) is None
