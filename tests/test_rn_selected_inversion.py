"""Selected inversion (Takahashi) of ``diag(H G⁻¹ Hᵀ)`` and the near-critical report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import random as sparse_random

import gridstate.algebra.normalized_residuals as nr
from gridstate.api import estimate
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.pipeline import PipelineConfig, manifest
from gridstate.quality_summary import near_critical_measurements
from gridstate.z_vector import KIND_POWER_P, KIND_VOLTAGE, MeasurementIndex


_DATA_DIR = Path(__file__).parent / "test_data" / "ieee"


def _random_case(seed: int, m: int = 300, n: int = 90) -> tuple[csr_matrix, np.ndarray]:
    """Sparse ``H`` with every state observed (``G`` positive definite)."""
    rng = np.random.default_rng(seed)
    H = sparse_random(m, n, density=0.04, random_state=seed, format="lil")
    for j in range(n):
        H[j, j] = 1.0 + rng.random()
    return H.tocsr(), rng.uniform(0.5, 2.0, m)


def _dense_hgh(H: csr_matrix, sigma2: np.ndarray) -> np.ndarray:
    Hd = H.toarray()
    G = Hd.T @ np.diag(1.0 / sigma2) @ Hd
    return np.einsum("ij,ji->i", Hd, np.linalg.solve(G, Hd.T))


def _factor_sources() -> list:
    sources = [pytest.param(nr.superlu_factor, id="superlu")]
    try:
        import cvxopt  # noqa: F401
    except ImportError:
        sources.append(pytest.param(None, id="cholmod", marks=pytest.mark.skip("no cvxopt")))
    else:
        sources.append(pytest.param(nr.cholmod_factor, id="cholmod"))
    return sources


@pytest.fixture
def python_kernels(monkeypatch: pytest.MonkeyPatch) -> nr.SelinvKernels:
    """Route the selected inversion through the plain Python kernels (no numba needed)."""
    kernels = nr.python_selinv_kernels()
    monkeypatch.setattr(nr, "_load_selinv_kernels", lambda: kernels)
    monkeypatch.setattr(nr, "_numba_switched_off", lambda: False)
    return kernels


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor_of", _factor_sources())
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selected_inverse_matches_dense_inverse(factor_of, seed: int) -> None:
    H, sigma2 = _random_case(seed)
    G = nr._gain(H, sigma2)
    factor = factor_of(G)
    assert factor is not None
    z = np.empty_like(factor.lval)
    kernels = nr.python_selinv_kernels()
    assert kernels.selinv_lower(factor.n, factor.colptr, factor.rowind, factor.lval, z) == 0
    # Z = (P G Pᵀ)⁻¹ on the pattern of L.
    perm = np.argsort(factor.perm_inv)
    Ginv = np.linalg.inv(G.toarray()[np.ix_(perm, perm)])
    cols = np.repeat(np.arange(factor.n), np.diff(factor.colptr))
    ref = Ginv[factor.rowind, cols]
    well = np.abs(ref) > 1e-8 * np.abs(Ginv).max()
    assert well.sum() > 0.5 * ref.size
    np.testing.assert_allclose(z[well], ref[well], rtol=1e-10)
    np.testing.assert_allclose(z, ref, rtol=0, atol=1e-10 * np.abs(Ginv).max())


@pytest.mark.parametrize("factor_of", _factor_sources())
@pytest.mark.parametrize("seed", [0, 1])
def test_hgh_diag_selinv_matches_dense(factor_of, seed: int) -> None:
    H, sigma2 = _random_case(seed)
    rows = np.sort(np.random.default_rng(seed).choice(H.shape[0], 120, replace=False))
    factor = factor_of(nr._gain(H, sigma2))
    got = nr._selinv_hgh(factor, H, rows, nr.python_selinv_kernels())
    assert got is not None
    ref = _dense_hgh(H, sigma2)
    np.testing.assert_allclose(got[rows], ref[rows], rtol=1e-10, atol=1e-14)
    assert np.isnan(np.delete(got, rows)).all()


def test_superlu_factor_reproduces_gain() -> None:
    H, sigma2 = _random_case(3)
    G = nr._gain(H, sigma2)
    f = nr.superlu_factor(G)
    L = csc_matrix((f.lval, f.rowind, f.colptr), shape=G.shape)
    perm = np.argsort(f.perm_inv)
    np.testing.assert_allclose((L @ L.T).toarray(), G.toarray()[np.ix_(perm, perm)], atol=1e-12)


def test_unclosed_pattern_is_detected() -> None:
    # A full 3x3 G with a factor pattern missing entry (2, 1): column 0 needs Z_21.
    G = np.array([[4.0, 1.0, 1.0], [1.0, 3.0, 0.5], [1.0, 0.5, 2.0]])
    L = np.linalg.cholesky(G)
    colptr = np.array([0, 3, 4, 5], dtype=np.int64)
    rowind = np.array([0, 1, 2, 1, 2], dtype=np.int64)
    lval = np.array([L[0, 0], L[1, 0], L[2, 0], L[1, 1], L[2, 2]])
    z = np.empty_like(lval)
    k = nr.python_selinv_kernels()
    assert k.selinv_lower(3, colptr, rowind, lval, z) == 1  # column 0 failed


def test_checked_factor_rejects_malformed_columns() -> None:
    perm = np.arange(2, dtype=np.int64)
    good = nr._checked_factor(
        np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([1.0, 0.5, 1.0]), perm
    )
    assert good is not None
    # diagonal not first
    assert nr._checked_factor(np.array([0, 2, 3]), np.array([1, 0, 1]), np.ones(3), perm) is None
    # empty column
    assert nr._checked_factor(np.array([0, 2, 2]), np.array([0, 1]), np.ones(2), perm) is None


def test_rows_outside_the_factor_pattern_fall_back_to_solves(python_kernels) -> None:
    # Rows 0 and 1 couple states 0 and 1 with opposite signs: their contributions
    # to G[0, 1] cancel exactly, so the entry is missing from G and from L, and
    # rows 0/1 need Z[0, 1]. They are computed by per-row solves instead.
    H = csr_matrix(
        np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    sigma2 = np.ones(5)
    G = nr._gain(H, sigma2)
    assert G[0, 1] == 0.0 and G.nnz == 3
    rows = np.arange(5)
    raw = nr._selinv_hgh(nr.superlu_factor(G), H, rows, python_kernels)
    assert raw is not None
    assert np.isnan(raw[[0, 1]]).all() and np.isfinite(raw[2:]).all()
    got = nr.hgh_diag_selinv(H, sigma2, rows)
    assert got is not None
    np.testing.assert_allclose(got, _dense_hgh(H, sigma2), rtol=1e-12)


# ---------------------------------------------------------------------------
# Method selection and fallbacks
# ---------------------------------------------------------------------------


def test_unknown_method_is_rejected() -> None:
    H, sigma2 = _random_case(0, m=40, n=10)
    with pytest.raises(ValueError, match="normalized residuals method"):
        nr.normalized_residuals(np.ones(40), H, sigma2, method="inverse")


def test_selected_inversion_matches_solve(python_kernels) -> None:
    H, sigma2 = _random_case(4)
    r = np.random.default_rng(4).normal(size=H.shape[0])
    mask = np.zeros(H.shape[0], dtype=bool)
    mask[::3] = True
    solve = nr.compute_normalized_residuals(r, H, sigma2, mask, method="solve")
    selinv = nr.compute_normalized_residuals(r, H, sigma2, mask, method="selected_inversion")
    np.testing.assert_array_equal(np.isnan(solve.rn), ~mask)
    np.testing.assert_allclose(selinv.rn[mask], solve.rn[mask], rtol=1e-10)
    np.testing.assert_allclose(selinv.omega_ratio[mask], solve.omega_ratio[mask], rtol=1e-10)
    assert np.isnan(selinv.omega_ratio[~mask]).all()


def test_numba_switch_off_uses_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRIDSTATE_NUMBA", "0")
    H, sigma2 = _random_case(5)
    with pytest.raises(nr.SelectedInversionUnavailableError):
        nr.hgh_diag_selinv(H, sigma2, np.arange(H.shape[0]))
    r = np.random.default_rng(5).normal(size=H.shape[0])
    a = nr.normalized_residuals(r, H, sigma2, method="solve")
    b = nr.normalized_residuals(r, H, sigma2, method="selected_inversion")
    np.testing.assert_array_equal(a, b)


def test_missing_kernels_use_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nr, "_load_selinv_kernels", lambda: None)
    monkeypatch.setattr(nr, "_numba_switched_off", lambda: False)
    H, sigma2 = _random_case(6)
    r = np.random.default_rng(6).normal(size=H.shape[0])
    a = nr.normalized_residuals(r, H, sigma2, method="solve")
    b = nr.normalized_residuals(r, H, sigma2, method="selected_inversion")
    np.testing.assert_array_equal(a, b)


def test_singular_gain_gives_inf(python_kernels) -> None:
    H = csr_matrix((6, 3))  # no information at all
    out = nr.compute_normalized_residuals(np.ones(6), H, np.ones(6), method="selected_inversion")
    assert np.isinf(out.rn).all()
    assert np.isnan(out.omega_ratio).all()


def test_numba_kernels_compile_and_agree() -> None:
    pytest.importorskip("numba")
    kernels = nr._load_selinv_kernels()
    assert kernels is not None
    H, sigma2 = _random_case(7)
    rows = np.arange(H.shape[0])
    G = nr._gain(H, sigma2)
    fast = nr._selinv_hgh(nr.superlu_factor(G), H, rows, kernels)
    slow = nr._selinv_hgh(nr.superlu_factor(G), H, rows, nr.python_selinv_kernels())
    assert fast is not None and slow is not None
    np.testing.assert_allclose(fast, slow, rtol=1e-13)


# ---------------------------------------------------------------------------
# Near-critical measurements
# ---------------------------------------------------------------------------


def _index(kinds: list[int]) -> MeasurementIndex:
    m = len(kinds)
    return MeasurementIndex(
        kind=np.asarray(kinds, dtype=np.int64),
        object_kind=np.zeros(m, dtype=np.int64),
        object_pos=np.zeros(m, dtype=np.int64),
        branch_side=np.full(m, -1, dtype=np.int64),
        meas_id=np.arange(101, 101 + m, dtype=np.int64),
    )


@pytest.mark.parametrize("method", ["solve", "selected_inversion"])
def test_near_critical_flag(method: str, python_kernels) -> None:
    # State 0: a precise reading (row 0) and a very loose one (row 1): the
    # estimate follows row 0 almost exactly, Ω_00 / σ²_0 ≈ 1e-6. State 1: two
    # equal readings (ratio 0.5). State 2: a single reading — critical.
    H = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    sigma2 = np.array([1.0, 1e6, 1.0, 1.0, 1.0])
    r = np.array([0.1, 1.0, 0.2, -0.2, 0.0])
    out = nr.compute_normalized_residuals(r, H, sigma2, method=method)
    np.testing.assert_allclose(out.omega_ratio[:4], [1e-6 / (1 + 1e-6), 1 / (1 + 1e-6), 0.5, 0.5])
    assert abs(out.omega_ratio[4]) < 1e-12 and np.isinf(out.rn[4])

    index = _index([KIND_VOLTAGE, KIND_VOLTAGE, KIND_POWER_P, KIND_POWER_P, KIND_VOLTAGE])
    rows = near_critical_measurements(out, index, threshold=1e-4)
    assert [row.measurement_id for row in rows] == [105, 101]  # ascending ratio
    assert rows[0].normalized_residual == np.inf
    assert rows[1].kind == "V" and rows[1].omega_ratio == pytest.approx(1e-6, rel=1e-5)
    assert near_critical_measurements(out, index, threshold=0.0) == []
    every = near_critical_measurements(out, index, threshold=1.0)
    assert len(every) == 5


def test_near_critical_respects_the_scope() -> None:
    H = csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    sigma2 = np.array([1.0, 1e6, 1.0, 1.0])
    mask = np.array([False, True, True, True])
    out = nr.compute_normalized_residuals(np.ones(4), H, sigma2, mask)
    rows = near_critical_measurements(out, _index([0, 0, 0, 0]), threshold=1e-4)
    assert rows == []  # the near-critical row 0 is outside the scope


def test_estimate_reports_near_critical_and_keeps_default_rn(python_kernels) -> None:
    def solve(**kwargs):
        model = load_se_input_npz(_DATA_DIR / "case14.npz").model
        res = estimate(model, algorithm="wls", **kwargs)
        assert res.success
        return res

    base = solve()
    assert base.near_critical_measurements == []  # fully redundant IEEE telemetry
    selinv = solve(quality_summary_rn_method="selected_inversion")
    assert [r.measurement_id for r in selinv.worst_residuals] == [
        r.measurement_id for r in base.worst_residuals
    ]
    np.testing.assert_allclose(
        [r.normalized_residual for r in selinv.worst_residuals],
        [r.normalized_residual for r in base.worst_residuals],
        rtol=1e-10,
    )
    everything = solve(quality_summary_near_critical=1.0)
    ratios = [row.omega_ratio for row in everything.near_critical_measurements]
    assert len(ratios) > 0 and ratios == sorted(ratios)
    assert all(0.0 < x < 1.0 for x in ratios)
    with pytest.raises(ValueError, match="normalized residuals method"):
        solve(quality_summary_rn_method="dense")


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


def test_config_defaults_and_validation() -> None:
    cfg = PipelineConfig()
    assert cfg.normalized_residuals_method == "solve"
    assert cfg.near_critical_threshold == 1e-4
    PipelineConfig(normalized_residuals_method="selected_inversion")
    with pytest.raises(ValueError, match="normalized_residuals_method"):
        PipelineConfig(normalized_residuals_method="inverse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="near_critical_threshold"):
        PipelineConfig(near_critical_threshold=-1.0)
    with pytest.raises(ValueError, match="near_critical_threshold"):
        PipelineConfig(near_critical_threshold=2.0)


def test_config_fields_are_in_the_manifest() -> None:
    text = repr(manifest())
    assert "normalized_residuals_method" in text
    assert "selected_inversion" in text
    assert "near_critical_threshold" in text


def test_pipeline_uses_the_configured_method(python_kernels) -> None:
    se_in = load_se_input_npz(_DATA_DIR / "case30.npz")
    base = run_se(se_in, config=PipelineConfig()).result
    alt = run_se(
        load_se_input_npz(_DATA_DIR / "case30.npz"),
        config=PipelineConfig(
            normalized_residuals_method="selected_inversion", near_critical_threshold=1.0
        ),
    ).result
    assert base is not None and alt is not None
    assert [r.measurement_id for r in alt.worst_residuals] == [
        r.measurement_id for r in base.worst_residuals
    ]
    np.testing.assert_allclose(
        [r.normalized_residual for r in alt.worst_residuals],
        [r.normalized_residual for r in base.worst_residuals],
        rtol=1e-10,
    )
    assert base.near_critical_measurements == []
    assert len(alt.near_critical_measurements) > 0
