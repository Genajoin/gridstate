"""Tests of the IPM solver on synthetic problems.

They do not depend on the network model: they exercise the bare mechanics of log-barrier WLS.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gridstate.algorithms.ipm import (
    IPMResult,
    _barrier_grad_hess,
    _barrier_value,
    _max_step_to_boundary,
    _project_to_interior,
    solve_ipm,
)


# ---------------------------------------------------------------- barrier-helpers
class TestBarrierHelpers:
    def test_project_to_interior_clips(self) -> None:
        x = np.array([5.0, -1.0, 0.5, 100.0])
        idx = np.array([0, 1, 3], dtype=np.int64)
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 10.0])
        out = _project_to_interior(x, idx, lo, hi, eps=0.01)
        # idx=2 (x=0.5) is not in box_idx -> unchanged
        assert out[2] == 0.5
        # all the others lie strictly inside [lo + 0.01, hi - 0.01]
        assert 0.01 <= out[0] <= 0.99
        assert 0.01 <= out[1] <= 0.99
        assert 0.1 <= out[3] <= 9.9

    def test_barrier_value_at_center_is_finite(self) -> None:
        x = np.array([0.5])
        idx = np.array([0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        v = _barrier_value(x, idx, lo, hi, mu=1.0)
        assert np.isfinite(v)
        # B(0.5; μ=1) = -1·[log(0.5) + log(0.5)] = -1·(2·log 0.5) = 2·log 2
        assert v == pytest.approx(2.0 * np.log(2.0), abs=1e-12)

    def test_barrier_value_outside_is_inf(self) -> None:
        x = np.array([1.5])  # outside [0, 1]
        idx = np.array([0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        v = _barrier_value(x, idx, lo, hi, mu=1.0)
        assert v == float("inf")

    def test_barrier_grad_correct_sign(self) -> None:
        # x is closer to the lower bound -> grad must push upward (positive)
        x = np.array([0.1])
        idx = np.array([0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        g, h = _barrier_grad_hess(x, idx, lo, hi, mu=1.0)
        # dB/dx = -μ/(x-lo) + μ/(hi-x) = -1/0.1 + 1/0.9 ≈ -10 + 1.11 = -8.89
        # This is the gradient of B; to minimise B one moves along -grad -> +8.89 > 0 ✓
        assert g[0] < 0  # B decreases when moving to the right (away from lo)
        assert h[0] > 0  # convexity of the barrier

    def test_max_step_to_boundary(self) -> None:
        x = np.array([0.5])
        dx = np.array([0.4])  # towards hi=1.0; α_max = (1-0.5)/0.4 = 1.25
        idx = np.array([0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        a = _max_step_to_boundary(x, dx, idx, lo, hi, fraction=0.995)
        assert a == pytest.approx(0.995 * 1.25, abs=1e-9)

    def test_max_step_no_box_is_inf(self) -> None:
        a = _max_step_to_boundary(
            np.array([0.5]),
            np.array([1.0]),
            np.array([], dtype=np.int64),
            np.array([]),
            np.array([]),
        )
        assert a == float("inf")


# ---------------------------------------------------------------- 1D tests
class Test1DScalarFit:
    """Minimal problem: estimate a scalar x from a single "measurement" z = x + ε."""

    def _make_1d_problem(
        self,
        z_value: float,
        sigma2: float = 1.0,
    ) -> tuple:
        z = np.array([z_value])
        r_inv = np.array([1.0 / sigma2])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac(x: np.ndarray) -> csr_matrix:
            # H = ∂h/∂x where h(x) = x -> H = 1 (as in WLS, not ∂r/∂x).
            return csr_matrix(np.array([[1.0]]))

        return residual, jac, r_inv

    def test_unconstrained_recovers_measurement(self) -> None:
        """Without boxes IPM = WLS, the answer is z."""
        residual, jac, r_inv = self._make_1d_problem(z_value=3.0)
        result = solve_ipm(
            x_init=np.array([0.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
        )
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-6)

    def test_constraint_inactive_recovers_measurement(self) -> None:
        """Box [0, 10] is inactive (z=3 lies inside): the answer is z."""
        residual, jac, r_inv = self._make_1d_problem(z_value=3.0)
        result = solve_ipm(
            x_init=np.array([0.5]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),
        )
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-3)

    def test_constraint_active_upper(self) -> None:
        """Box [0, 2] is active from above (z=10 is outside): x -> 2- (pressed against hi)."""
        residual, jac, r_inv = self._make_1d_problem(z_value=10.0)
        result = solve_ipm(
            x_init=np.array([1.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([2.0]),
            mu_min=1e-8,
        )
        assert result.x[0] < 2.0
        # As μ->0 the solution must come close to 2.
        assert result.x[0] > 1.95, f"got {result.x[0]}"

    def test_constraint_active_lower(self) -> None:
        """Box [5, 10] is active from below (z=-1 is outside): x -> 5+."""
        residual, jac, r_inv = self._make_1d_problem(z_value=-1.0)
        result = solve_ipm(
            x_init=np.array([7.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([5.0]),
            box_hi=np.array([10.0]),
            mu_min=1e-8,
        )
        assert result.x[0] > 5.0
        assert result.x[0] < 5.05


# ---------------------------------------------------------------- 2D tests
class Test2DOverdetermined:
    """y = x + noise (2 measurements of one variable + 1 variable without)."""

    def test_two_measurements_average(self) -> None:
        """Without boxes: x* = ½(z1+z2)."""
        z = np.array([2.0, 4.0])
        r_inv = np.array([1.0, 1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0], x[0]])

        def jac(x: np.ndarray) -> csr_matrix:
            return csr_matrix(np.array([[1.0], [1.0]]))

        result = solve_ipm(
            x_init=np.array([0.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
        )
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-6)

    def test_two_vars_decoupled(self) -> None:
        """2 variables, 1 measurement each. Only x[1] is boxed."""
        z = np.array([5.0, 15.0])
        r_inv = np.array([1.0, 1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0], x[1]])

        def jac(x: np.ndarray) -> csr_matrix:
            return csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

        result = solve_ipm(
            x_init=np.array([0.0, 5.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([1]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),  # x[1] is pressed against 10
            mu_min=1e-8,
        )
        # x[0] is free -> must land exactly on z[0]=5
        assert result.x[0] == pytest.approx(5.0, abs=1e-3)
        # x[1] is pressed against 10
        assert result.x[1] < 10.0
        assert result.x[1] > 9.9


# ---------------------------------------------------------------- WLS-equivalence
class TestWLSEquivalence:
    def test_no_box_one_outer_pass(self) -> None:
        """Without box vars the outer loop makes a single pass (WLS mode)."""
        z = np.array([3.0])
        r_inv = np.array([1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac(x: np.ndarray) -> csr_matrix:
            # H = ∂h/∂x where h(x) = x -> H = 1 (as in WLS, not ∂r/∂x).
            return csr_matrix(np.array([[1.0]]))

        result = solve_ipm(
            x_init=np.array([0.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
        )
        assert result.iterations_outer == 1
        assert result.mu_final == 0.0  # WLS mode zeroes μ


class TestConvergenceStatus:
    """Two-level status: kkt / completed / stalled / error."""

    @staticmethod
    def _problem(z_value: float):
        z = np.array([z_value])
        r_inv = np.array([1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac(x: np.ndarray) -> csr_matrix:
            return csr_matrix(np.array([[1.0]]))

        return residual, jac, r_inv

    def test_clean_convergence_is_kkt(self) -> None:
        residual, jac, r_inv = self._problem(3.0)
        result = solve_ipm(
            x_init=np.array([0.5]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),
        )
        assert result.status == "kkt"
        assert result.solve_status == "kkt"
        assert result.success

    def test_schedule_done_loose_kkt_is_completed_and_success(self) -> None:
        """The μ schedule is exhausted but the strict KKT is not reached -> completed,
        success=True (previously a false "did not converge")."""
        residual, jac, r_inv = self._problem(10.0)
        # Active bound + tiny inner_max: the barrier gradient does not
        # get below tol in time, but μ reaches mu_min.
        result = solve_ipm(
            x_init=np.array([1.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([2.0]),
            inner_tol=1e-12,  # deliberately unreachable strict threshold
            outer_max=200,
        )
        assert result.mu_final <= 1e-6
        if result.status != "kkt":  # with an extreme tol we expect completed
            assert result.status == "completed"
            assert result.success
        assert "μ_final" in result.message

    def test_start_at_solution_is_converged_not_stalled(self) -> None:
        """Start at the exact solution (after promoting the result into the input) ->
        converged, not a false "stalled".

        Regression: the very first Newton step < inner_tol → inner break
        without accepted steps → lazy break down to μ_min → status='stalled'
        with ANY iteration limit (reproduced on a large production model,
        2026-06-10).
        Inner convergence by tolerance is NOT stagnation: the μ schedule
        must reach mu_min -> kkt/completed.
        """
        residual, jac, r_inv = self._problem(3.0)
        # Step 1: solve from a neutral start.
        first = solve_ipm(
            x_init=np.array([0.5]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),
        )
        assert first.success
        # Step 2: restart from the solution found ("start at the solution").
        second = solve_ipm(
            x_init=first.x.copy(),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),
        )
        assert second.status in ("kkt", "completed"), second.message
        assert second.success
        assert second.mu_final <= 1e-6  # the μ schedule ran to the end
        assert second.x[0] == pytest.approx(first.x[0], abs=1e-3)

    def test_real_stall_armijo_fail_stays_stalled(self) -> None:
        """Genuine stagnation (Armijo finds no step far from the optimum)
        still yields stalled: the fix does not weaken stall detection."""
        z = np.array([10.0])
        r_inv = np.array([1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac_wrong_sign(x: np.ndarray) -> csr_matrix:
            # Wrong Jacobian sign -> the Newton "descent" goes uphill:
            # Armijo accepts no step, grad is large, μ > μ_min.
            return csr_matrix(np.array([[-1.0]]))

        result = solve_ipm(
            x_init=np.array([1.0]),
            residual_fn=residual,
            jacobian_fn=jac_wrong_sign,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([100.0]),
        )
        assert result.status == "stalled"
        assert result.solve_status == "stalled"
        assert not result.success

    def test_error_status_on_nonfinite(self) -> None:
        z = np.array([3.0])
        r_inv = np.array([1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac_bad(x: np.ndarray) -> csr_matrix:
            return csr_matrix(np.array([[np.nan]]))  # nan -> non-finite step

        result = solve_ipm(
            x_init=np.array([0.5]),
            residual_fn=residual,
            jacobian_fn=jac_bad,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),
        )
        assert result.status == "error"
        assert result.solve_status == "error"
        assert not result.success


# ---------------------------------------------------------------- residual reuse
class TestResidualReuse:
    """The accepted Armijo trial's residual is reused, never recomputed at the same x."""

    def test_no_repeated_evaluation_at_the_same_point(self) -> None:
        a = np.array([[1.0, 0.5], [0.0, 2.0], [1.0, -1.0]])
        z = np.array([1.0, 4.0, -0.5])
        calls: list[np.ndarray] = []

        def residual(x: np.ndarray) -> np.ndarray:
            calls.append(x.copy())
            # Mildly nonlinear so Newton takes several inner steps.
            return z - (a @ x + 0.1 * x[0] ** 3)

        def jac(x: np.ndarray) -> csr_matrix:
            d = a.copy()
            d[:, 0] += 0.3 * x[0] ** 2
            return csr_matrix(d)

        result = solve_ipm(
            x_init=np.array([0.5, 0.5]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=np.ones(3),
            box_idx=np.array([1]),
            box_lo=np.array([0.0]),
            box_hi=np.array([10.0]),
        )
        assert result.success
        assert result.iterations_inner_total > 1
        repeated = sum(np.array_equal(p, q) for p, q in itertools.pairwise(calls))
        assert repeated == 0
        # The reported objective is evaluated at the returned point.
        r = residual(result.x)
        assert result.objective_data == 0.5 * float(r @ r)


# ---------------------------------------------------------------- trust region scope
class TestTrustRegionScope:
    """``tr_exclude_box``: the trust-region norm over the free coordinates only."""

    # x = [v, g]: v free, g boxed in [0, 100]; the data pull g from 1 to ~60,
    # a move far beyond tr_radius=0.5, while v only needs a small correction.
    _A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.01]])
    _Z = np.array([1.0, 60.0, 1.6])
    _R_INV = np.array([1e4, 1.0, 1e2])

    def _solve(self, calls: list[np.ndarray] | None = None, **kwargs: Any) -> IPMResult:
        a, z = self._A, self._Z

        def residual(x: np.ndarray) -> np.ndarray:
            if calls is not None:
                calls.append(x.copy())
            return z - a @ x

        def jac(x: np.ndarray) -> csr_matrix:
            return csr_matrix(a)

        return solve_ipm(
            x_init=np.array([0.0, 1.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=self._R_INV,
            box_idx=np.array([1]),
            box_lo=np.array([0.0]),
            box_hi=np.array([100.0]),
            **kwargs,
        )

    def test_box_move_no_longer_throttles_the_step(self) -> None:
        whole = self._solve(inner_max=200, tr_exclude_box=False)
        free_only = self._solve(inner_max=200, tr_exclude_box=True)
        assert whole.success and free_only.success
        assert free_only.iterations_inner_total < whole.iterations_inner_total / 5
        np.testing.assert_allclose(free_only.x, whole.x, atol=1e-3)

    def test_factor_from_free_part_scales_the_whole_step(self) -> None:
        """The first trial step keeps the Newton direction; its free part is
        clipped to tr_radius, the box part may exceed it."""
        calls: list[np.ndarray] = []
        self._solve(calls, tr_exclude_box=True, tr_radius=0.01, inner_max=1, outer_max=1)
        step = calls[1] - calls[0]
        assert abs(step[0]) == pytest.approx(0.01, rel=1e-9)
        assert abs(step[1]) > 0.01

        calls_all: list[np.ndarray] = []
        self._solve(calls_all, tr_exclude_box=False, tr_radius=0.01, inner_max=1, outer_max=1)
        step_all = calls_all[1] - calls_all[0]
        assert np.max(np.abs(step_all)) == pytest.approx(0.01, rel=1e-9)
        # Same direction, different length.
        cos = step @ step_all / (np.linalg.norm(step) * np.linalg.norm(step_all))
        assert cos == pytest.approx(1.0, abs=1e-12)

    def test_default_is_free_coordinates_only(self) -> None:
        default = self._solve(inner_max=200)
        free_only = self._solve(inner_max=200, tr_exclude_box=True)
        assert np.array_equal(default.x, free_only.x)
        assert default.iterations_inner_total == free_only.iterations_inner_total

    def test_without_boxes_the_flag_is_a_no_op(self) -> None:
        a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        z = np.array([3.0, -4.0, 0.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - a @ x

        def jac(x: np.ndarray) -> csr_matrix:
            return csr_matrix(a)

        res = [
            solve_ipm(
                x_init=np.zeros(2),
                residual_fn=residual,
                jacobian_fn=jac,
                r_inv_diag=np.ones(3),
                tr_exclude_box=flag,
            )
            for flag in (False, True)
        ]
        assert np.array_equal(res[0].x, res[1].x)
        assert res[0].iterations_inner_total == res[1].iterations_inner_total
