"""Primal-dual IPM, equality rows and the single-factorization KKT path on synthetic problems."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gridstate.algorithms.ipm import solve_ipm
from gridstate.algorithms.ipm_pd import solve_ipm_pd
from gridstate.algorithms.kkt_solver import KKTSolver, resolve_backend
from gridstate.algorithms.status import SolveStatus


def _linear(A: np.ndarray, z: np.ndarray):
    """``residual_fn`` / ``jacobian_fn`` of the linear model ``h(x) = A x``."""
    H = csr_matrix(A)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return z - A @ x

    def jacobian_fn(x: np.ndarray) -> csr_matrix:
        return H

    return residual_fn, jacobian_fn


BACKENDS = ["scipy"] + (["cholmod"] if resolve_backend("auto") == "cholmod" else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_factor_normal_matches_solve_normal(backend: str) -> None:
    rng = np.random.default_rng(0)
    H = csr_matrix(rng.normal(size=(12, 5)))
    w = rng.uniform(0.5, 2.0, size=12)
    diag = rng.uniform(0.0, 1.0, size=5)
    rhs1, rhs2 = rng.normal(size=5), rng.normal(size=5)
    solve = KKTSolver(backend).factor_normal(H, w, diag=diag)
    ref = KKTSolver(backend)
    np.testing.assert_allclose(solve(rhs1), ref.solve_normal(H, w, rhs1, diag=diag), rtol=1e-10)
    np.testing.assert_allclose(solve(rhs2), ref.solve_normal(H, w, rhs2, diag=diag), rtol=1e-10)


def test_pd_unconstrained_is_least_squares() -> None:
    rng = np.random.default_rng(1)
    A = rng.normal(size=(10, 3))
    z = rng.normal(size=10)
    w = rng.uniform(1.0, 4.0, size=10)
    res, jac = _linear(A, z)
    out = solve_ipm_pd(np.zeros(3), res, jac, w, step_tol=1e-10)
    sw = np.sqrt(w)
    x_ls = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)[0]
    assert out.status == "kkt" and out.solve_status is SolveStatus.KKT
    np.testing.assert_allclose(out.x, x_ls, atol=1e-8)


def test_pd_variable_box_matches_primal_ipm() -> None:
    # Two measurements of x0 pull it to 3; the box caps it at 2.
    A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z = np.array([3.0, 3.2, 1.0])
    w = np.ones(3)
    res, jac = _linear(A, z)
    box: dict[str, np.ndarray] = {
        "box_idx": np.array([0]),
        "box_lo": np.array([-5.0]),
        "box_hi": np.array([2.0]),
    }
    pd = solve_ipm_pd(np.zeros(2), res, jac, w, **box)
    primal = solve_ipm(np.zeros(2), res, jac, w, **box)
    assert pd.success
    assert pd.x[0] == pytest.approx(2.0, abs=1e-5)
    np.testing.assert_allclose(pd.x, primal.x, atol=1e-4)


def test_pd_measurement_box_bounds_the_estimate() -> None:
    # Row 2 measures x0 + x1 = 5 with a large sigma but its value may not exceed 3.
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z = np.array([2.0, 2.0, 5.0])
    w = np.array([1.0, 1.0, 0.01])
    res, jac = _linear(A, z)
    free = solve_ipm_pd(np.zeros(2), res, jac, w)
    assert A[2] @ free.x > 3.5
    # The bound applies to h − z = −r of the row: h ≤ 3 ⇔ h − z ≤ −2.
    out = solve_ipm_pd(
        np.zeros(2),
        res,
        jac,
        w,
        meas_box_rows=np.array([2]),
        meas_box_lo=np.array([-np.inf]),
        meas_box_hi=np.array([3.0 - 5.0]),
    )
    assert out.success
    assert A[2] @ out.x == pytest.approx(3.0, abs=1e-5)


def _constrained_ls(A, z, w, C, d):
    """Exact solution of min Σ w r² s.t. C x = d (KKT system)."""
    n = A.shape[1]
    K = np.block([[A.T @ (w[:, None] * A), C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
    rhs = np.concatenate([A.T @ (w * z), d])
    return np.linalg.solve(K, rhs)[:n]


@pytest.mark.parametrize("solver", ["pd", "primal"])
def test_equality_rows_are_enforced(solver: str) -> None:
    # Rows 0-3: data; row 4: equality x0 + x1 + x2 = 1 (a "balance").
    rng = np.random.default_rng(2)
    A_data = rng.normal(size=(4, 3))
    z_data = rng.normal(size=4)
    w_data = rng.uniform(1.0, 2.0, size=4)
    C = np.array([[1.0, 1.0, 1.0]])
    A = np.vstack([A_data, C])
    z = np.concatenate([z_data, [1.0]])
    rho = 1e3 * np.median(w_data)
    w = np.concatenate([w_data, [rho]])
    res, jac = _linear(A, z)
    eq: dict[str, np.ndarray] = {"eq_idx": np.array([4]), "eq_tol": np.array([1e-8])}
    if solver == "pd":
        out = solve_ipm_pd(np.zeros(3), res, jac, w, step_tol=1e-10, **eq)
    else:
        out = solve_ipm(np.zeros(3), res, jac, w, inner_tol=1e-10, **eq)
    assert out.eq_inf_final <= 1e-8
    np.testing.assert_allclose(out.x, _constrained_ls(A_data, z_data, w_data, C, [1.0]), atol=1e-6)


def test_equality_rows_with_a_box_free_common_level() -> None:
    # The equality fixes only g − l (net injection); data say nothing about the
    # common level of g and l, which the barrier alone sets. The step stays
    # well-defined (proximal term) and the equality holds.
    A = np.array([[1.0, 0.0, 0.0], [1.0, -1.0, 1.0]])  # x0 measured; row 1: x0 − (g − l) = 0
    z = np.array([0.7, 0.0])
    w = np.array([1.0, 1e5])
    res, jac = _linear(A, z)
    kw: dict[str, np.ndarray] = {
        "box_idx": np.array([1, 2]),
        "box_lo": np.array([-50.0, -50.0]),
        "box_hi": np.array([50.0, 50.0]),
        "eq_idx": np.array([1]),
        "eq_tol": np.array([1e-6]),
    }
    pd = solve_ipm_pd(np.zeros(3), res, jac, w, **kw)
    primal = solve_ipm(np.zeros(3), res, jac, w, inner_tol=1e-8, **kw)
    for out in (pd, primal):
        assert out.eq_inf_final <= 1e-6
        assert out.x[1] - out.x[2] == pytest.approx(0.7, abs=1e-5)


def test_pd_rejects_inconsistent_inputs() -> None:
    res, jac = _linear(np.eye(2), np.zeros(2))
    with pytest.raises(ValueError, match="eq_tol"):
        solve_ipm_pd(np.zeros(2), res, jac, np.ones(2), eq_idx=np.array([0]))
    with pytest.raises(ValueError, match="measurement box"):
        solve_ipm_pd(
            np.zeros(2),
            res,
            jac,
            np.ones(2),
            meas_box_rows=np.array([0]),
            meas_box_lo=np.array([-np.inf]),
            meas_box_hi=np.array([np.inf]),
        )
