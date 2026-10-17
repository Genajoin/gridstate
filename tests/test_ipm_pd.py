"""Primal-dual IPM, equality rows and the single-factorization KKT path on synthetic problems."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gridstate.algorithms.ipm import solve_ipm
from gridstate.algorithms.ipm_pd import _Merit, solve_ipm_pd
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


def test_pd_merit_search_stops_gauss_newton_overshoot() -> None:
    # h = atan(x), z = 0 from x0 = 1.5: the Gauss-Newton step −(1 + x²)·atan(x)
    # overshoots and the full-step iteration diverges; the merit search does not.
    def residual_fn(x: np.ndarray) -> np.ndarray:
        return -np.arctan(x)

    def jacobian_fn(x: np.ndarray) -> csr_matrix:
        return csr_matrix(np.array([[1.0 / (1.0 + x[0] ** 2)]]))

    kw = {"tr_radius": 1e6, "step_tol": 1e-10, "max_iter": 50}
    no_search = solve_ipm_pd(
        np.array([1.5]), residual_fn, jacobian_fn, np.ones(1), armijo_max=0, **kw
    )
    out = solve_ipm_pd(np.array([1.5]), residual_fn, jacobian_fn, np.ones(1), **kw)
    assert no_search.solve_status is not SolveStatus.KKT
    assert out.solve_status is SolveStatus.KKT
    assert out.x[0] == pytest.approx(0.0, abs=1e-8)


def test_pd_merit_slope_matches_finite_difference() -> None:
    # Nonlinear rows, a variable box, a one-sided measurement box and an
    # equality row; the slack step is Newton-consistent (Δs = G·Δx + r_s).
    rng = np.random.default_rng(3)
    z = np.array([0.3, -0.2, 0.5, 0.1])

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return z - np.array([np.sin(x[0]) + x[1], x[1] * x[2], np.cos(x[2]), x[0] + x[2]])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [np.cos(x[0]), 1.0, 0.0],
                [0.0, x[2], x[1]],
                [0.0, 0.0, -np.sin(x[2])],
                [1.0, 0.0, 1.0],
            ]
        )

    box_idx, mb_rows, eq_rows = np.array([1]), np.array([2]), np.array([3])
    lo = np.array([-1.0, -np.inf])
    hi = np.array([2.0, 0.4])
    il, iu = np.array([0]), np.array([0, 1])
    w_data = np.array([2.0, 3.0, 1.5, 0.0])

    def g_of(x: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.concatenate([x[box_idx], -r[mb_rows]])

    x = rng.normal(size=3)
    r = residual_fn(x)
    g = g_of(x, r)
    s_l = np.array([0.7])
    s_u = np.array([0.9, 0.3])
    r_l = g[il] - lo[il] - s_l
    r_u = hi[iu] - g[iu] - s_u
    merit = _Merit(
        w_data=w_data,
        lam=np.array([0.4]),
        rho=np.array([50.0]),
        eq_rows=eq_rows,
        mu=0.05,
        nu=3.0,
        lo_l=lo[il],
        hi_u=hi[iu],
        il=il,
        iu=iu,
    )
    lin = SimpleNamespace(r=r, c=-r[eq_rows], s_l=s_l, s_u=s_u, r_l=r_l, r_u=r_u)
    dx = rng.normal(size=3)
    hdx = jacobian(x) @ dx
    gdx = np.concatenate([dx[box_idx], hdx[mb_rows]])
    ds_l = gdx[il] + r_l
    ds_u = -gdx[iu] + r_u

    def phi(a: float) -> float:
        xa = x + a * dx
        ra = residual_fn(xa)
        return merit.value(ra, g_of(xa, ra), s_l + a * ds_l, s_u + a * ds_u)

    eps = 1e-7
    fd = (phi(eps) - phi(0.0)) / eps
    assert merit.slope(lin, hdx, ds_l, ds_u) == pytest.approx(fd, rel=1e-4, abs=1e-6)  # type: ignore[arg-type]


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
