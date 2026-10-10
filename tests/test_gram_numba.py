"""Fixed-pattern Gram path of KKTSolver (numba) is bit-identical to scipy.

``KKTSolver.solve_normal`` must hand CHOLMOD exactly what
``KKTSolver.solve(gram_scipy(...))`` does: the same lower-triangle values, the
same superset-pattern growth (exact zeros pruned by scipy, ``PATTERN_EPS`` on
seen positions), the same retry/spsolve cascade. Every comparison below is
bitwise. The numba-dependent tests skip without numba or cvxopt; the fallback
(scipy) path is exercised everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, tril
from scipy.sparse import random as sparse_random

from gridstate.algorithms import kkt_solver as ks
from gridstate.algorithms.ipm import solve_ipm
from gridstate.algorithms.kkt_solver import NUMBA_ENV, KKTSolver, gram_scipy


HAS_CVXOPT = ks._cvxopt_available()
HAS_NUMBA_GRAM = ks.numba_gram_available()

needs_fast_path = pytest.mark.skipif(
    not (HAS_CVXOPT and HAS_NUMBA_GRAM),
    reason="numba not installed" if HAS_CVXOPT else "cvxopt not installed",
)


def _jacobian(m: int, n: int, seed: int, density: float = 0.08) -> csr_matrix:
    """Canonical CSR with some explicit zeros (as on the fixed Jacobian pattern)."""
    rng = np.random.default_rng(seed)
    H = sparse_random(m, n, density=density, random_state=rng, format="csr")
    H = (H + csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(m, n))).tocsr()
    H.sum_duplicates()
    H.sort_indices()
    H.data = rng.standard_normal(H.nnz)
    H.data[rng.random(H.nnz) < 0.05] = 0.0
    return H


def _with_values(H: csr_matrix, data: np.ndarray) -> csr_matrix:
    """Same pattern (same index arrays), new values."""
    return csr_matrix((data, H.indices, H.indptr), shape=H.shape)


def _zero_columns(H: csr_matrix, cols: list[int]) -> csr_matrix:
    data = H.data.copy()
    data[np.isin(H.indices, cols)] = 0.0
    return _with_values(H, data)


def _lower_dense(gram: ks._GramPattern, values: np.ndarray) -> np.ndarray:
    out = np.zeros(gram.shape)
    out[gram.g_row, gram.g_col] = values
    return out


@pytest.fixture
def numba_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NUMBA_ENV, raising=False)


# ---------------------------------------------------------------- assembly


@pytest.mark.skipif(not HAS_NUMBA_GRAM, reason="numba not installed")
class TestKernel:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("with_diag", [False, True])
    def test_lower_triangle_matches_scipy_bitwise(self, seed: int, with_diag: bool) -> None:
        rng = np.random.default_rng(100 + seed)
        H = _jacobian(120, 40, seed)
        w = rng.uniform(0.1, 1e4, 120)
        diag = rng.uniform(0.0, 1e6, 40) if with_diag else None
        if diag is not None:
            diag[::3] = 0.0
        gram = ks._GramPattern(H)
        kernel = ks._load_gram_kernel()
        assert kernel is not None
        got = _lower_dense(gram, gram.assemble(kernel, H, w, diag))
        ref = tril(gram_scipy(H, w, diag)).toarray()
        assert np.array_equal(got, ref)

    def test_self_check_passes(self) -> None:
        kernel = ks._load_gram_kernel()
        assert kernel is not None
        assert ks._self_check(kernel)

    def test_self_check_rejects_a_kernel_off_by_the_last_bit(self) -> None:
        kernel = ks._load_gram_kernel()
        assert kernel is not None

        def fused(*args: object) -> None:
            kernel(*args)
            out = args[-1]
            assert isinstance(out, np.ndarray)
            out[0] += 2.0**-60  # what an FMA leaves in G[0, 0]

        assert not ks._self_check(fused)


def test_unavailable_numba_disables_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    ks._load_gram_kernel.cache_clear()
    monkeypatch.setitem(sys.modules, "gridstate.algorithms._gram_numba", None)
    try:
        assert ks._load_gram_kernel() is None
        assert not ks.numba_gram_available()
    finally:
        monkeypatch.undo()
        ks._load_gram_kernel.cache_clear()


def test_canonical_csr_check() -> None:
    H = _jacobian(30, 10, seed=3)
    assert ks._canonical_csr(H)
    unsorted = csr_matrix((H.data[::-1].copy(), H.indices[::-1].copy(), H.indptr), shape=H.shape)
    # Reversing data/indices globally breaks the per-row order.
    assert not ks._canonical_csr(unsorted)
    assert not ks._canonical_csr(H.tocsc())
    dup = csr_matrix(
        (np.ones(2), np.array([1, 1]), np.array([0, 2])), shape=(1, 3)
    )  # duplicate (0, 1)
    assert not ks._canonical_csr(dup)


# ---------------------------------------------------------------- solve


def _sequence(seed: int) -> list[tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]]:
    """IPM-like sequence on one pattern: columns vanish (exact zeros in G) and return."""
    rng = np.random.default_rng(seed)
    H_init = _jacobian(150, 50, seed)
    steps = []
    for zeroed in ([3, 7], [], [5], [], [3, 11, 20], []):
        H = _with_values(H_init, H_init.data * rng.uniform(0.5, 1.5, H_init.nnz))
        H = _zero_columns(H, zeroed)
        w = rng.uniform(0.5, 50.0, 150)
        diag = rng.uniform(1e-3, 1.0, 50)  # barrier Hessian keeps G definite
        rhs = rng.standard_normal(50)
        steps.append((H, w, diag, rhs))
    return steps


def _reference(steps: list) -> list[np.ndarray]:
    ref = KKTSolver("cholmod")
    return [ref.solve(gram_scipy(H, w, d), rhs) for H, w, d, rhs in steps]


@needs_fast_path
@pytest.mark.usefixtures("numba_on")
class TestSolveNormal:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_sequence_with_pattern_growth_matches_scipy_path(self, seed: int) -> None:
        steps = _sequence(seed)
        solver = KKTSolver("cholmod")
        got = [solver.solve_normal(H, w, rhs, diag=d) for H, w, d, rhs in steps]
        assert solver._mode == "gram"
        for a, b in zip(got, _reference(steps), strict=True):
            assert np.array_equal(a, b)

    def test_without_diag(self) -> None:
        rng = np.random.default_rng(7)
        H = _jacobian(200, 40, seed=7, density=0.15)
        w = rng.uniform(1.0, 2.0, 200)
        rhs = rng.standard_normal(40)
        solver = KKTSolver("cholmod")
        ref = KKTSolver("cholmod")
        for scale in (1.0, 0.7, 1.3):
            Hs = _with_values(H, H.data * scale)
            assert np.array_equal(
                solver.solve_normal(Hs, w, rhs), ref.solve(gram_scipy(Hs, w), rhs)
            )
        assert solver._mode == "gram"

    def test_not_positive_definite_cascade_matches(self) -> None:
        rng = np.random.default_rng(5)
        H = _jacobian(80, 30, seed=5)
        w = rng.uniform(0.5, 2.0, 80)
        w[::4] *= -50.0  # indefinite G: numeric fails, retry fails, spsolve
        rhs = rng.standard_normal(30)
        solver = KKTSolver("cholmod")
        ref = KKTSolver("cholmod")
        a = solver.solve_normal(H, w, rhs)
        b = ref.solve(gram_scipy(H, w), rhs)
        assert np.array_equal(a, b, equal_nan=True)
        # The cascade leaves the same state: the next (definite) system agrees too.
        w2 = np.abs(w)
        assert np.array_equal(solver.solve_normal(H, w2, rhs), ref.solve(gram_scipy(H, w2), rhs))

    def test_mixing_with_plain_solve_matches(self) -> None:
        steps = _sequence(2)
        solver = KKTSolver("cholmod")
        got = []
        for i, (H, w, d, rhs) in enumerate(steps):
            if i == 2:
                got.append(solver.solve(gram_scipy(H, w, d), rhs))
            else:
                got.append(solver.solve_normal(H, w, rhs, diag=d))
        assert solver._mode == "matrix"
        for a, b in zip(got, _reference(steps), strict=True):
            assert np.array_equal(a, b)

    def test_changed_jacobian_pattern_falls_back(self) -> None:
        steps = _sequence(3)
        H_other = _jacobian(150, 50, seed=99)
        rng = np.random.default_rng(3)
        w, d, rhs = rng.uniform(0.5, 5.0, 150), rng.uniform(0.1, 1.0, 50), rng.standard_normal(50)
        seq = [*steps[:2], (H_other, w, d, rhs), *steps[2:]]
        solver = KKTSolver("cholmod")
        got = [solver.solve_normal(H, w_, rhs_, diag=d_) for H, w_, d_, rhs_ in seq]
        assert solver._mode == "matrix"
        for a, b in zip(got, _reference(seq), strict=True):
            assert np.array_equal(a, b)


@pytest.mark.skipif(not HAS_CVXOPT, reason="cvxopt not installed")
def test_env_switch_disables_the_numba_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NUMBA_ENV, "0")
    steps = _sequence(4)
    solver = KKTSolver("cholmod")
    got = [solver.solve_normal(H, w, rhs, diag=d) for H, w, d, rhs in steps]
    assert solver._mode == "matrix"
    for a, b in zip(got, _reference(steps), strict=True):
        assert np.array_equal(a, b)


def test_scipy_backend_uses_spsolve_path() -> None:
    H, w, d, rhs = _sequence(5)[0]
    solver = KKTSolver("scipy")
    got = solver.solve_normal(H, w, rhs, diag=d)
    assert solver._mode is None
    assert np.array_equal(got, KKTSolver("scipy").solve(gram_scipy(H, w, d), rhs))


# ---------------------------------------------------------------- IPM


def _ipm_problem() -> tuple:
    """Small nonlinear box-constrained least squares on a fixed Jacobian pattern."""
    rng = np.random.default_rng(11)
    A = _jacobian(60, 20, seed=11, density=0.2)
    z = rng.standard_normal(60)
    pattern = A.copy()

    def residual(x: np.ndarray) -> np.ndarray:
        return z - (A @ x + 0.05 * x**2 @ np.ones((20, 60)) / 20.0)

    def jac(x: np.ndarray) -> csr_matrix:
        dense_extra = 0.1 * x / 20.0  # d/dx_k of 0.05·Σx²/20, same for every row
        data = A.data + dense_extra[pattern.indices]
        return _with_values(pattern, data)

    return residual, jac, np.full(60, 4.0)


@needs_fast_path
@pytest.mark.usefixtures("numba_on")
def test_solve_ipm_numba_and_scipy_paths_agree_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    residual, jac, r_inv = _ipm_problem()
    kwargs = {
        "box_idx": np.array([0, 4, 9]),
        "box_lo": np.array([-0.5, -1.0, 0.0]),
        "box_hi": np.array([0.5, 1.0, 2.0]),
    }
    fast = KKTSolver("cholmod")
    res_fast = solve_ipm(np.zeros(20), residual, jac, r_inv, kkt_solver=fast, **kwargs)
    assert fast._mode == "gram"
    monkeypatch.setenv(NUMBA_ENV, "0")
    plain = KKTSolver("cholmod")
    res_plain = solve_ipm(np.zeros(20), residual, jac, r_inv, kkt_solver=plain, **kwargs)
    assert plain._mode == "matrix"
    assert res_fast.iterations_inner_total == res_plain.iterations_inner_total > 1
    assert np.array_equal(res_fast.x, res_plain.x)
    assert res_fast.objective_data == res_plain.objective_data
