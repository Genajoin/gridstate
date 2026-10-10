"""Selected inversion (Takahashi) of ``diag(H G⁻¹ Hᵀ)`` and the near-critical report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import random as sparse_random

import gridstate.algebra.normalized_residuals as nr
from gridstate.api import estimate, populate_quality_summary
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
    assert cfg.selected_inversion_refine_below == nr.SELINV_REFINE_BELOW == 1e-2
    PipelineConfig(selected_inversion_refine_below=0.0)
    PipelineConfig(selected_inversion_refine_below=1.0)
    for bad in (-1e-3, 1.5, float("nan")):
        with pytest.raises(ValueError, match="selected_inversion_refine_below"):
            PipelineConfig(selected_inversion_refine_below=bad)


def test_config_fields_are_in_the_manifest() -> None:
    text = repr(manifest())
    assert "normalized_residuals_method" in text
    assert "selected_inversion" in text
    assert "near_critical_threshold" in text
    entry = next(p for p in manifest()["params"] if p["name"] == "selected_inversion_refine_below")
    assert entry["default"] == 1e-2
    assert (entry["min"], entry["max"]) == (0.0, 1.0)
    assert entry["depends"] == {"normalized_residuals_method": "selected_inversion"}


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


# ---------------------------------------------------------------------------
# Hybrid: selected inversion refined by per-row solves
# ---------------------------------------------------------------------------


def _ill_conditioned_case(seed: int = 12) -> tuple[csr_matrix, np.ndarray]:
    """DC-flow-like ``H`` on a 12 x 15 meshed grid with an ill-conditioned gain.

    Branch flows ``b_ij (x_i − x_j)`` with ``b`` over five decades, nodal sums
    on half of the nodes and absolute readings of ~30 % of the states whose
    variances span 14 decades: ``cond(G) ~ 1e14``. Precise readings next to
    loose ones make many measurements nearly critical (``Ω/σ²`` down to
    ~1e-12), a few exactly critical.
    """
    rng = np.random.default_rng(seed)
    rows, cols = 12, 15
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if c + 1 < cols:
                edges.append((k, k + 1))
            if r + 1 < rows and (c == 0 or rng.random() < 0.35):
                edges.append((k, k + cols))
    b = 10.0 ** rng.uniform(0.0, 5.0, len(edges))
    trip: list[tuple[int, int, float]] = []
    sigma2: list[float] = []
    for m, ((i, j), bij) in enumerate(zip(edges, b, strict=True)):
        trip += [(m, i, bij), (m, j, -bij)]
        sigma2.append(10.0 ** rng.uniform(-6.0, 0.0))
    adj: dict[int, list[tuple[int, float]]] = {}
    for (i, j), bij in zip(edges, b, strict=True):
        adj.setdefault(i, []).append((j, bij))
        adj.setdefault(j, []).append((i, bij))
    m = len(edges)
    for i in range(n):
        if rng.random() < 0.5:
            trip.append((m, i, sum(v for _, v in adj[i])))
            trip += [(m, j, -v) for j, v in adj[i]]
            sigma2.append(10.0 ** rng.uniform(-6.0, 0.0))
            m += 1
    for i in range(n):
        if rng.random() < 0.3 or i == 0:
            trip.append((m, i, 1.0))
            sigma2.append(10.0 ** rng.uniform(-10.0, 4.0))
            m += 1
    r_idx, c_idx, vals = zip(*trip, strict=True)
    return csr_matrix((vals, (r_idx, c_idx)), shape=(m, n)), np.asarray(sigma2)


def _reference_hgh(H: csr_matrix, sigma2: np.ndarray) -> np.ndarray:
    """``diag(H G⁻¹ Hᵀ)`` of the float64 ``G`` by iterative refinement.

    Dense Cholesky solves refined with residuals in extended precision
    (``np.longdouble``; plain float64 where the platform has no wider type).
    """
    from scipy.linalg import cho_factor, cho_solve

    G = nr._gain(H, sigma2).toarray()
    G_ext = G.astype(np.longdouble)
    B = H.toarray().T.astype(np.longdouble)
    c = cho_factor(G, lower=True)
    X = cho_solve(c, np.asarray(B, dtype=np.float64)).astype(np.longdouble)
    for _ in range(4):
        R = B - G_ext @ X
        X = X + cho_solve(c, np.asarray(R, dtype=np.float64)).astype(np.longdouble)
    return np.asarray(np.einsum("ij,ij->j", B, X), dtype=np.float64)


def _plain_selinv_hgh(H: csr_matrix, sigma2: np.ndarray, kernels: nr.SelinvKernels) -> np.ndarray:
    """Selected inversion without refinement, on the factor the hybrid uses."""
    G = nr._gain(H, sigma2)
    try:
        factor = nr.cholmod_factor(G)
    except ImportError:
        factor = nr.superlu_factor(G)
    assert factor is not None
    hgh = nr._selinv_hgh(factor, H, np.arange(H.shape[0]), kernels)
    assert hgh is not None and np.isfinite(hgh).all()
    return hgh


def test_hybrid_matches_solve_on_an_ill_conditioned_gain(python_kernels) -> None:
    H, sigma2 = _ill_conditioned_case()
    m = H.shape[0]
    assert np.linalg.cond(nr._gain(H, sigma2).toarray()) > 1e12
    r = np.random.default_rng(0).normal(size=m) * np.sqrt(sigma2)

    solve = nr.compute_normalized_residuals(r, H, sigma2, method="solve")
    hybrid = nr.compute_normalized_residuals(r, H, sigma2, method="selected_inversion")
    plain_hgh = _plain_selinv_hgh(H, sigma2, python_kernels)
    plain_ratio = (sigma2 - plain_hgh) / sigma2
    ref_ratio = (sigma2 - _reference_hgh(H, sigma2)) / sigma2

    # Plain selected inversion loses digits on near-critical rows, solve does not.
    # (The reference itself is good to ~1e-10 in the ratio here, so rows with
    # a ratio near zero are left out of this comparison.)
    near = (ref_ratio < 1e-4) & (ref_ratio > 1e-8)
    assert near.sum() >= 10
    err_plain = np.abs(plain_ratio - ref_ratio)[near] / ref_ratio[near]
    err_solve = np.abs(solve.omega_ratio - ref_ratio)[near] / ref_ratio[near]
    assert err_plain.max() > 1e-2
    assert err_solve.max() < 1e-4
    assert np.median(err_solve) < 1e-2 * np.median(err_plain)

    # Refined rows (selected-inversion ratio below the threshold, or Ω at the
    # cut-off) carry the solve values; inf decisions are those of solve.
    refined = (plain_ratio < nr.SELINV_REFINE_BELOW) | (sigma2 - plain_hgh <= nr.OMEGA_CUTOFF)
    assert 0 < refined.sum() < m
    np.testing.assert_array_equal(np.isinf(hybrid.rn), np.isinf(solve.rn))
    assert np.isinf(solve.rn).any()
    finite = refined & np.isfinite(solve.rn)
    np.testing.assert_allclose(hybrid.rn[finite], solve.rn[finite], rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(
        hybrid.omega_ratio[refined], solve.omega_ratio[refined], rtol=1e-9, atol=1e-15
    )
    # Every row whose (true) ratio is below the threshold was refined.
    assert refined[solve.omega_ratio < 0.5 * nr.SELINV_REFINE_BELOW].all()

    # The other rows keep the selected-inversion values.
    keep = ~refined
    np.testing.assert_array_equal(hybrid.omega_ratio[keep], plain_ratio[keep])
    # ... which are close to solve: there the rounding of the selected inverse
    # is amplified by at most 1 / SELINV_REFINE_BELOW. (Observed |Δ ratio|:
    # ~2e-6 on the CHOLMOD factor, ~7e-4 on the SuperLU one.)
    np.testing.assert_allclose(
        hybrid.omega_ratio[keep], solve.omega_ratio[keep], rtol=5e-2, atol=0.0
    )


def test_refine_below_zero_recomputes_only_the_cutoff_rows(python_kernels) -> None:
    H, sigma2 = _ill_conditioned_case()
    m = H.shape[0]
    r = np.ones(m)
    solve = nr.compute_normalized_residuals(r, H, sigma2, method="solve")
    plain_hgh = _plain_selinv_hgh(H, sigma2, python_kernels)
    at_cutoff = sigma2 - plain_hgh <= nr.OMEGA_CUTOFF
    assert at_cutoff.any()
    calls: list[np.ndarray] = []
    original = nr._rows_to_refine

    def spy(hgh, s2, row_idx, refine_below):  # type: ignore[no-untyped-def]
        rows = original(hgh, s2, row_idx, refine_below)
        calls.append(rows)
        return rows

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nr, "_rows_to_refine", spy)
        only_cutoff = nr.compute_normalized_residuals(
            r, H, sigma2, method="selected_inversion", refine_below=0.0
        )
        everything = nr.compute_normalized_residuals(
            r, H, sigma2, method="selected_inversion", refine_below=1.0
        )
    np.testing.assert_array_equal(calls[0], np.where(at_cutoff)[0])
    np.testing.assert_array_equal(calls[1], np.arange(m))
    # No false inf: a row is critical only if solve says so. (Without the
    # ratio refinement a critical row may still get a finite r_N.)
    assert not (np.isinf(only_cutoff.rn) & ~np.isinf(solve.rn)).any()
    np.testing.assert_allclose(
        only_cutoff.omega_ratio[at_cutoff], solve.omega_ratio[at_cutoff], rtol=1e-9, atol=1e-15
    )
    np.testing.assert_allclose(everything.omega_ratio, solve.omega_ratio, rtol=1e-9, atol=1e-15)
    np.testing.assert_array_equal(np.isinf(everything.rn), np.isinf(solve.rn))


def test_refinement_uses_the_solve_values_bit_for_bit(python_kernels) -> None:
    # The refined rows come from the same factorization and the same solves as
    # method="solve": with refine_below=1 every row is recomputed.
    H, sigma2 = _ill_conditioned_case(seed=4)
    r = np.random.default_rng(1).normal(size=H.shape[0])
    solve = nr.compute_normalized_residuals(r, H, sigma2, method="solve")
    full = nr.compute_normalized_residuals(
        r, H, sigma2, method="selected_inversion", refine_below=1.0
    )
    np.testing.assert_allclose(full.rn, solve.rn, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(np.isinf(full.rn), np.isinf(solve.rn))


@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf"), True, "0.01"])
def test_invalid_refine_below_is_rejected(bad) -> None:
    H, sigma2 = _random_case(0, m=40, n=10)
    with pytest.raises(ValueError, match="refine_below"):
        nr.compute_normalized_residuals(
            np.ones(40), H, sigma2, method="selected_inversion", refine_below=bad
        )


def test_refine_below_is_threaded_through(python_kernels, monkeypatch) -> None:
    seen: list[float] = []
    original = nr.hgh_diag_selinv

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs["refine_below"])
        return original(*args, **kwargs)

    monkeypatch.setattr(nr, "hgh_diag_selinv", spy)
    model = load_se_input_npz(_DATA_DIR / "case14.npz").model
    res = estimate(
        model,
        algorithm="wls",
        quality_summary_rn_method="selected_inversion",
        quality_summary_rn_refine_below=0.25,
    )
    assert seen == [0.25]
    populate_quality_summary(res, rn_method="selected_inversion", rn_refine_below=0.5)
    assert seen == [0.25, 0.5]
    run_se(
        load_se_input_npz(_DATA_DIR / "case14.npz"),
        config=PipelineConfig(
            normalized_residuals_method="selected_inversion",
            selected_inversion_refine_below=0.125,
        ),
    )
    assert seen[-1] == 0.125
    with pytest.raises(ValueError, match="refine_below"):
        estimate(
            load_se_input_npz(_DATA_DIR / "case14.npz").model,
            quality_summary_rn_refine_below=-1.0,
        )
    with pytest.raises(ValueError, match="refine_below"):
        populate_quality_summary(res, rn_refine_below=float("nan"))
