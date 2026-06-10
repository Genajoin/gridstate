"""Тесты IPM-solver'а на синтетических задачах.

Не зависят от модели сети: проверяют чистую механику log-barrier WLS.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gridstate.algorithms.ipm import (
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
        # idx=2 (x=0.5) не в box_idx → не меняется
        assert out[2] == 0.5
        # все остальные строго внутри [lo + 0.01, hi - 0.01]
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
        x = np.array([1.5])  # вне [0, 1]
        idx = np.array([0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        v = _barrier_value(x, idx, lo, hi, mu=1.0)
        assert v == float("inf")

    def test_barrier_grad_correct_sign(self) -> None:
        # x ближе к нижней границе → grad должен толкать вверх (положительный)
        x = np.array([0.1])
        idx = np.array([0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        g, h = _barrier_grad_hess(x, idx, lo, hi, mu=1.0)
        # dB/dx = -μ/(x-lo) + μ/(hi-x) = -1/0.1 + 1/0.9 ≈ -10 + 1.11 = -8.89
        # Это градиент B; чтобы минимизировать B, нужно идти в -grad → +8.89 > 0 ✓
        assert g[0] < 0  # B уменьшается при движении вправо (от lo)
        assert h[0] > 0  # выпуклость barrier'а

    def test_max_step_to_boundary(self) -> None:
        x = np.array([0.5])
        dx = np.array([0.4])  # к hi=1.0; α_max = (1-0.5)/0.4 = 1.25
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
    """Минимальная задача: оценить скаляр x по одному «измерению» z = x + ε."""

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
            # H = ∂h/∂x где h(x) = x → H = 1 (как в WLS, не ∂r/∂x).
            return csr_matrix(np.array([[1.0]]))

        return residual, jac, r_inv

    def test_unconstrained_recovers_measurement(self) -> None:
        """Без боксов IPM = WLS, ответ = z."""
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
        """Box [0, 10] — не активен (z=3 внутри): ответ = z."""
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
        """Box [0, 2] активен сверху (z=10 вне): x → 2− (упирается в hi)."""
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
        # При μ→0 решение должно подойти к 2 близко.
        assert result.x[0] > 1.95, f"got {result.x[0]}"

    def test_constraint_active_lower(self) -> None:
        """Box [5, 10] активен снизу (z=−1 вне): x → 5+."""
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
    """y = x + noise (2 измерения одной переменной + 1 переменной без)."""

    def test_two_measurements_average(self) -> None:
        """Без боксов: x* = ½(z1+z2)."""
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
        """2 переменные, по 1 измерению каждая. Только x[1] под box."""
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
            box_hi=np.array([10.0]),  # x[1] упирается в 10
            mu_min=1e-8,
        )
        # x[0] свободна → должна попасть в z[0]=5 точно
        assert result.x[0] == pytest.approx(5.0, abs=1e-3)
        # x[1] упирается в 10
        assert result.x[1] < 10.0
        assert result.x[1] > 9.9


# ---------------------------------------------------------------- WLS-equivalence
class TestWLSEquivalence:
    def test_no_box_one_outer_pass(self) -> None:
        """Без box-vars outer-loop делает 1 проход (WLS-режим)."""
        z = np.array([3.0])
        r_inv = np.array([1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac(x: np.ndarray) -> csr_matrix:
            # H = ∂h/∂x где h(x) = x → H = 1 (как в WLS, не ∂r/∂x).
            return csr_matrix(np.array([[1.0]]))

        result = solve_ipm(
            x_init=np.array([0.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
        )
        assert result.iterations_outer == 1
        assert result.mu_final == 0.0  # WLS-режим обнуляет μ


class TestConvergenceStatus:
    """Двухуровневый статус: kkt / completed / stalled / error."""

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
        assert result.success

    def test_schedule_done_loose_kkt_is_completed_and_success(self) -> None:
        """μ-расписание пройдено, но строгий KKT не достигнут → completed,
        success=True (раньше — ложный «не сошёлся»)."""
        residual, jac, r_inv = self._problem(10.0)
        # Активная граница + крошечный inner_max: grad у барьера не
        # успевает стать < tol, но μ доходит до mu_min.
        result = solve_ipm(
            x_init=np.array([1.0]),
            residual_fn=residual,
            jacobian_fn=jac,
            r_inv_diag=r_inv,
            box_idx=np.array([0]),
            box_lo=np.array([0.0]),
            box_hi=np.array([2.0]),
            inner_tol=1e-12,  # заведомо недостижимый строгий порог
            outer_max=200,
        )
        assert result.mu_final <= 1e-6
        if result.status != "kkt":  # при экстремальном tol ожидаем completed
            assert result.status == "completed"
            assert result.success
        assert "μ_final" in result.message

    def test_error_status_on_nonfinite(self) -> None:
        z = np.array([3.0])
        r_inv = np.array([1.0])

        def residual(x: np.ndarray) -> np.ndarray:
            return z - np.array([x[0]])

        def jac_bad(x: np.ndarray) -> csr_matrix:
            return csr_matrix(np.array([[np.nan]]))  # nan → non-finite шаг

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
        assert not result.success
