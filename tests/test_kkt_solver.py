"""KKTSolver: реюз символьной факторизации CHOLMOD + scipy-фолбэк.

Тесты cholmod-бэкенда скипаются без cvxopt (optional-зависимость
``gridstate[fast]``); scipy-путь и resolve-логика проверяются всегда.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import spsolve

from gridstate.algorithms import kkt_solver as ks
from gridstate.algorithms.kkt_solver import KKTSolver, resolve_backend


HAS_CVXOPT = ks._cvxopt_available()

needs_cvxopt = pytest.mark.skipif(not HAS_CVXOPT, reason="cvxopt не установлен")


def _spd(n: int, seed: int, density: float = 0.1) -> csc_matrix:
    """Случайная SPD-матрица: AᵀA + n·I."""
    rng = np.random.default_rng(seed)
    A = sparse_random(n, n, density=density, random_state=rng, format="csc")
    eye = csc_matrix((np.full(n, float(n)), (np.arange(n), np.arange(n))), shape=(n, n))
    return (A.T @ A + eye).tocsc()


class TestResolveBackend:
    def test_scipy_always(self) -> None:
        assert resolve_backend("scipy") == "scipy"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="неизвестный kkt_solver"):
            resolve_backend("pardiso")

    def test_auto_resolves(self) -> None:
        assert resolve_backend("auto") == ("cholmod" if HAS_CVXOPT else "scipy")

    def test_explicit_cholmod_without_cvxopt_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ks, "_cvxopt_available", lambda: False)
        with pytest.raises(ImportError, match="cvxopt"):
            resolve_backend("cholmod")
        assert resolve_backend("auto") == "scipy"


class TestScipyBackend:
    def test_matches_spsolve(self) -> None:
        G = _spd(30, seed=1)
        rhs = np.arange(30, dtype=np.float64)
        solver = KKTSolver("scipy")
        np.testing.assert_array_equal(solver.solve(G, rhs), spsolve(G, rhs))


@needs_cvxopt
class TestCholmodBackend:
    def test_sequence_same_pattern(self) -> None:
        """Значения меняются, паттерн фиксирован — symbolic один раз."""
        G = _spd(40, seed=2)
        solver = KKTSolver("cholmod")
        rng = np.random.default_rng(3)
        for _ in range(5):
            Gi = G.copy()
            Gi.data = Gi.data * rng.uniform(0.5, 2.0)
            Gi = (Gi + Gi.T).tocsc()  # симметрия после случайного скейла
            Gi = (
                Gi + csc_matrix((np.full(40, 80.0), (np.arange(40), np.arange(40))), (40, 40))
            ).tocsc()
            rhs = rng.normal(size=40)
            np.testing.assert_allclose(
                solver.solve(Gi, rhs), spsolve(Gi, rhs), rtol=1e-8, atol=1e-12
            )

    def test_pattern_growth_and_shrink(self) -> None:
        """Дрожь паттерна (±элементы) — superset-проекция, решения верны."""
        n = 25
        base = _spd(n, seed=4)
        extra = csc_matrix(([0.3, 0.3], ([0, n - 1], [n - 1, 0])), shape=(n, n))
        seq = [base, (base + extra).tocsc(), base, (base + 2 * extra).tocsc()]
        solver = KKTSolver("cholmod")
        for Gi in seq:
            rhs = np.ones(n)
            np.testing.assert_allclose(
                solver.solve(Gi, rhs), spsolve(Gi, rhs), rtol=1e-8, atol=1e-12
            )

    def test_non_pd_falls_back_to_spsolve(self) -> None:
        """Симметричная не-PD матрица: единичный spsolve-фолбэк."""
        G = csc_matrix(np.diag([1.0, -1.0, 2.0]))
        rhs = np.array([1.0, 2.0, 4.0])
        solver = KKTSolver("cholmod")
        np.testing.assert_allclose(solver.solve(G, rhs), [1.0, -2.0, 2.0])

    def test_pattern_eps_does_not_change_values(self) -> None:
        """Проекция на superset не искажает данные (ε вне разрядности)."""
        n = 25
        base = _spd(n, seed=5)
        extra = csc_matrix(([0.3, 0.3], ([0, n - 1], [n - 1, 0])), shape=(n, n))
        wide = (base + extra).tocsc()
        solver = KKTSolver("cholmod")
        rhs = np.ones(n)
        solver.solve(wide, rhs)  # union-паттерн = wide
        # base ⊂ wide: решение через дырки с ε == точному решению base
        np.testing.assert_allclose(
            solver.solve(base, rhs), spsolve(base, rhs), rtol=1e-8, atol=1e-12
        )


@needs_cvxopt
class TestSolverIntegration:
    def test_run_same_result_both_backends(self) -> None:
        """run() c kkt_solver='cholmod' ≈ 'scipy' (wls и ipm)."""
        import sys
        from pathlib import Path

        from gridstate.pipeline import PipelineConfig, run

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_pipeline_idempotent import _make_model_with_reactor

        for algorithm in ("wls", "ipm"):
            results = {}
            for backend in ("scipy", "cholmod"):
                # ШР заглушен: модель с реактором 605 МВАр конфликтна с
                # нулевыми Qinj-мерами и не даёт success на IPM (как в
                # test_bad_data_repass).
                model = _make_model_with_reactor(susceptance_uS=0.0)
                res = run(model, config=PipelineConfig(algorithm=algorithm, kkt_solver=backend))
                assert res.success, (algorithm, backend)
                results[backend] = res
            np.testing.assert_allclose(
                results["cholmod"].v_pu, results["scipy"].v_pu, rtol=1e-6, atol=1e-9
            )
            np.testing.assert_allclose(
                results["cholmod"].delta_rad, results["scipy"].delta_rad, rtol=1e-6, atol=1e-8
            )
