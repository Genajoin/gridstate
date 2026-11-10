"""Sparse (CHOLMOD) and dense paths of ``diag(H G⁻¹ Hᵀ)`` agree."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from gridstate.algebra.normalized_residuals import hgh_diag_cholmod, hgh_diag_dense


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

    sparse = hgh_diag_cholmod(H, sigma2, rows, block=64)
    dense = hgh_diag_dense(H, sigma2, rows)
    assert sparse is not None and dense is not None
    np.testing.assert_allclose(sparse[rows], dense[rows], rtol=1e-10, atol=1e-12)
    assert np.isnan(np.delete(sparse, rows)).all()


def test_singular_gain_returns_none() -> None:
    H = sparse_random(10, 4, density=0.0, format="csr")  # no information at all
    assert hgh_diag_cholmod(H, np.ones(10), np.arange(10)) is None


@pytest.mark.parametrize("seed", [0, 1])
def test_splu_path_matches_dense_path(seed: int) -> None:
    """The large-network no-cvxopt path (sparse SuperLU) gives the dense values."""
    from gridstate.algebra.normalized_residuals import hgh_diag_splu

    rng = np.random.default_rng(seed)
    m, n = 400, 120
    H = sparse_random(m, n, density=0.03, random_state=seed, format="lil")
    for j in range(n):
        H[j, j] = 1.0 + rng.random()
    H = H.tocsr()
    sigma2 = rng.uniform(0.5, 2.0, m)
    rows = np.sort(rng.choice(m, 150, replace=False))

    splu = hgh_diag_splu(H, sigma2, rows, block=64)
    dense = hgh_diag_dense(H, sigma2, rows)
    assert splu is not None and dense is not None
    np.testing.assert_allclose(splu[rows], dense[rows], rtol=1e-10, atol=1e-12)


def test_large_network_never_forms_a_dense_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    import gridstate.algebra.normalized_residuals as nr

    monkeypatch.setattr(nr, "DENSE_MAX_STATES", 10)
    called = []
    monkeypatch.setattr(nr, "hgh_diag_splu", lambda *a, **k: called.append(1) or np.zeros(1))
    H = sparse_random(40, 20, density=0.2, random_state=0, format="csr")
    nr.hgh_diag_dense(H, np.ones(40), np.arange(40))
    assert called


def test_gain_not_pd_by_rounding_still_gives_residuals() -> None:
    """Weights 1 and 1e19: LLᵀ of G breaks down, a diagonal shift does not, r_N stay finite."""
    from scipy.sparse import eye

    from gridstate.algebra.normalized_residuals import compute_normalized_residuals

    H = (sparse_random(80, 40, density=0.1, random_state=1) + eye(80, 40)).tocsr()
    sigma2 = np.ones(80)
    sigma2[:5] = 1e-19
    r = np.random.default_rng(0).standard_normal(80) * np.sqrt(sigma2)
    for method in ("solve", "selected_inversion"):
        out = compute_normalized_residuals(r, H, sigma2, method=method)
        assert np.isfinite(out.rn[5:]).sum() > 30, method
