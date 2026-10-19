"""Hard nodal balance, fixed injection of non-estimated parts, measurement bounds, ipm_pd."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.sparse import diags

from gridstate.api import estimate
from gridstate.bounds import SENTINEL_VALUE
from gridstate.pipeline import PipelineConfig
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.state import StateLayout
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.z_vector import (
    KIND_VOLTAGE,
    MeasurementIndex,
    build_z_and_r,
    measurement_bounds_pu,
)
from tests.test_boxed_balance import N_BALANCE, _build_net_with_transit


def _setup(model, **kwargs):
    network_pu = model_to_pu(model)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)
    mi = MeasurementIndex(
        kind=np.full(2, KIND_VOLTAGE, dtype=np.int64),
        object_kind=np.zeros(2, dtype=np.int64),
        object_pos=np.array([0, 1], dtype=np.int64),
        branch_side=np.full(2, -1, dtype=np.int64),
        meas_id=np.array([1, 2], dtype=np.int64),
    )
    return build_ipm_setup(
        model,
        network_pu,
        np.ones(2),
        diags(np.array([1e-4, 4e-4])).tocsr(),
        mi,
        layout_base=layout,
        **kwargs,
    )


# ---------------------------------------------------------------- setup
def test_soft_mode_has_no_balance_tolerance():
    setup = _setup(_build_net_with_transit())
    assert setup.balance_tol is None
    assert setup.balance_start == 2 and setup.n_balance == N_BALANCE
    np.testing.assert_array_equal(setup.balance_rows, np.arange(2, 2 + 2 * N_BALANCE))


def test_hard_mode_penalty_and_tolerance():
    setup = _setup(
        _build_net_with_transit(),
        balance_mode="hard",
        hard_balance_rho_factor=1e3,
        balance_tol_p_mw=2.0,
        balance_tol_q_mvar=20.0,
        # Ignored under the hard balance: the row is a constraint, not a measurement.
        balance_weight_factor=0.1,
        boxed_balance_weight_factor=1e6,
    )
    sigma2 = setup.r_matrix.diagonal()[2 : 2 + 2 * N_BALANCE]
    assert np.allclose(sigma2, 2.5e-4 / 1e3)
    assert setup.balance_tol is not None
    np.testing.assert_allclose(setup.balance_tol[:N_BALANCE], 2.0 / BASE_MVA)
    np.testing.assert_allclose(setup.balance_tol[N_BALANCE:], 20.0 / BASE_MVA)


def test_unknown_balance_mode_is_rejected():
    with pytest.raises(ValueError, match="balance_mode"):
        _setup(_build_net_with_transit(), balance_mode="strict")


def test_transit_hard_mode_makes_only_transit_rows_equalities():
    # Node order 1..4: node 2 is the transit node.
    setup = _setup(
        _build_net_with_transit(),
        transit_balance_mode="hard",
        transit_balance_tol_p_mw=0.02,
        transit_balance_tol_q_mvar=0.2,
        hard_balance_rho_factor=1e3,
    )
    assert setup.balance_tol is None
    assert setup.eq_rows is not None and setup.eq_tol is not None
    np.testing.assert_array_equal(setup.eq_rows, [2 + 1, 2 + N_BALANCE + 1])
    np.testing.assert_allclose(setup.eq_tol, [0.02 / BASE_MVA, 0.2 / BASE_MVA])
    sigma2 = setup.r_matrix.diagonal()[2 : 2 + 2 * N_BALANCE]
    soft = _setup(_build_net_with_transit()).r_matrix.diagonal()[2 : 2 + 2 * N_BALANCE]
    is_eq = np.zeros(2 * N_BALANCE, dtype=bool)
    is_eq[[1, N_BALANCE + 1]] = True
    assert np.allclose(sigma2[is_eq], 2.5e-4 / 1e3)
    np.testing.assert_array_equal(sigma2[~is_eq], soft[~is_eq])


def test_soft_modes_have_no_equality_rows():
    setup = _setup(_build_net_with_transit(), transit_balance_weight_factor=10.0)
    assert setup.eq_rows is None and setup.eq_tol is None


def test_global_hard_mode_overrides_the_transit_mode():
    setup = _setup(
        _build_net_with_transit(),
        balance_mode="hard",
        balance_tol_p_mw=2.0,
        transit_balance_mode="hard",
        transit_balance_tol_p_mw=0.02,
    )
    assert setup.eq_rows is not None and setup.eq_tol is not None
    np.testing.assert_array_equal(setup.eq_rows, setup.balance_rows)
    np.testing.assert_array_equal(setup.eq_tol, setup.balance_tol)


def test_unknown_transit_balance_mode_is_rejected():
    with pytest.raises(ValueError, match="transit_balance_mode"):
        _setup(_build_net_with_transit(), transit_balance_mode="strict")
    with pytest.raises(ValueError, match="ipm_transit_balance_mode"):
        PipelineConfig(algorithm="ipm_pd", ipm_transit_balance_mode="strict")


def test_fixed_injection_from_input():
    m = _build_net_with_transit()
    # Transit node 2 gets a non-estimated load and generation.
    m.nodes.update(2, {"load_p": 7.0, "load_q": 3.0, "generation_p": 2.0, "generation_q": 1.0})
    off = _setup(m)
    on = _setup(m, fixed_injection_from_input=True)
    rows = slice(2, 2 + 2 * N_BALANCE)
    assert np.all(off.z[rows] == 0.0)
    # Node order 1..4: only node 2 has parts without a box variable.
    np.testing.assert_allclose(on.z[2 : 2 + N_BALANCE], [0.0, -5.0 / BASE_MVA, 0.0, 0.0])
    np.testing.assert_allclose(on.z[2 + N_BALANCE : rows.stop], [0.0, -2.0 / BASE_MVA, 0.0, 0.0])


# ---------------------------------------------------------------- measurement bounds
def test_measurement_bounds_pu_conversion_and_unset_pairs():
    m = _build_net_with_transit()
    # id 1: V of node 1 in [100, 115] kV; id 5: P injection of node 3 with an upper side only.
    m.measurements.update(1, {"min_value": 100.0, "max_value": 115.0})
    m.measurements.update(5, {"min_value": -SENTINEL_VALUE, "max_value": -20.0})
    network_pu = model_to_pu(m)
    _, _, mi = build_z_and_r(m, m.measurements, network_pu)
    lo, hi = measurement_bounds_pu(m.measurements, mi, network_pu)
    row = {int(i): k for k, i in enumerate(mi.meas_id)}
    assert lo[row[1]] == pytest.approx(100.0 / 110.0)
    assert hi[row[1]] == pytest.approx(115.0 / 110.0)
    assert lo[row[5]] == pytest.approx(-SENTINEL_VALUE / BASE_MVA)  # half-sentinel pair kept
    assert hi[row[5]] == pytest.approx(-20.0 / BASE_MVA)
    # Defaults (±9999 sentinels) stay unbounded.
    assert np.isinf(lo[row[2]]) and np.isinf(hi[row[2]])


# ---------------------------------------------------------------- end to end
def _imbalance(model) -> tuple[float, float]:
    n = model.nodes.to_numpy()
    p = n["p_inj_calc"] - n["generation_p_estimated"] + n["load_p_estimated"]
    q = n["q_inj_calc"] - n["generation_q_estimated"] + n["load_q_estimated"]
    return float(np.max(np.abs(p))), float(np.max(np.abs(q)))


@pytest.mark.parametrize("algorithm", ["ipm", "ipm_pd"])
def test_hard_balance_closes_the_estimated_imbalance(algorithm: str):
    m = _build_net_with_transit()
    res = estimate(
        m,
        algorithm=algorithm,
        reconcile_balance=False,
        include_quality_summary=False,
        balance_mode="hard",
        balance_tol_p_mw=0.1,
        balance_tol_q_mvar=0.1,
    )
    assert res.success
    imb_p, imb_q = _imbalance(m)
    assert imb_p <= 0.1 and imb_q <= 0.1


@pytest.mark.parametrize("algorithm", ["ipm", "ipm_pd"])
def test_transit_hard_balance_zeroes_the_transit_injection(algorithm: str):
    # Soft, the transit node 2 absorbs the data conflict (≈ 4 MW / 14 MVAr).
    soft = _build_net_with_transit()
    estimate(soft, algorithm=algorithm, reconcile_balance=False, include_quality_summary=False)
    n_soft = soft.nodes.to_numpy()
    assert abs(n_soft["p_inj_calc"][1]) > 1.0 and abs(n_soft["q_inj_calc"][1]) > 1.0
    m = _build_net_with_transit()
    res = estimate(
        m,
        algorithm=algorithm,
        reconcile_balance=False,
        include_quality_summary=False,
        transit_balance_mode="hard",
    )
    assert res.success
    n = m.nodes.to_numpy()
    assert abs(n["p_inj_calc"][1]) <= 0.01 and abs(n["q_inj_calc"][1]) <= 0.1


def test_ipm_pd_close_to_ipm():
    results = {}
    for algorithm in ("ipm", "ipm_pd"):
        m = _build_net_with_transit()
        res = estimate(m, algorithm=algorithm, include_quality_summary=False)
        assert res.success
        results[algorithm] = m.nodes.to_numpy()["voltage_magnitude"]
    np.testing.assert_allclose(results["ipm_pd"], results["ipm"], atol=1e-3 * 110.0)


def test_ipm_pd_measurement_bounds_hold():
    m = _build_net_with_transit()
    # Node 3 telemetry says the load is 30 MW; its value is bounded to [-25, 0] MW.
    m.measurements.update(5, {"min_value": -25.0, "max_value": 0.0})
    free = _build_net_with_transit()
    estimate(free, algorithm="ipm_pd", include_quality_summary=False)
    estimate(
        m,
        algorithm="ipm_pd",
        include_quality_summary=False,
        reconcile_balance=False,
        measurement_bounds=True,
    )
    p_free = free.nodes.to_numpy()["p_inj_calc"][2]
    p_bound = m.nodes.to_numpy()["p_inj_calc"][2]
    assert p_free < -25.0
    assert p_bound >= -25.0 - 1e-3


def test_measurement_bounds_require_ipm_pd():
    with pytest.raises(ValueError, match="ipm_pd"):
        estimate(_build_net_with_transit(), algorithm="ipm", measurement_bounds=True)
    with pytest.raises(ValueError, match="ipm_pd"):
        PipelineConfig(algorithm="ipm", ipm_measurement_bounds=True)


def test_voltage_guard_is_on_by_default_and_ignored_outside_ipm_pd():
    assert PipelineConfig(algorithm="ipm_pd").ipm_voltage_guard
    PipelineConfig(algorithm="ipm", ipm_voltage_guard=True)
    assert estimate(_build_net_with_transit(), algorithm="ipm", voltage_guard=True).success


def test_voltage_guard_keeps_v_within_the_node_limits():
    free = estimate(_build_net_with_transit(), algorithm="ipm_pd", voltage_guard=False)
    m = _build_net_with_transit()
    nd = m.nodes.to_numpy()
    v_free = {int(i): float(v) for i, v in zip(nd["id"], free.v_pu, strict=True)}
    nid = max(v_free, key=v_free.get)
    node = m.nodes.get_by_id(nid)
    cap = (v_free[nid] - 0.01) * float(node.voltage_nominal)
    node.voltage_max = cap
    guarded = estimate(m, algorithm="ipm_pd", voltage_guard=True)
    assert guarded.success
    k = int(np.flatnonzero(m.nodes.to_numpy()["id"] == nid)[0])
    assert float(m.nodes.to_numpy()["voltage_magnitude"][k]) <= cap * (1 + 1e-6)


def test_config_defaults_and_ranges():
    cfg = PipelineConfig(algorithm="ipm_pd")
    assert cfg.ipm_balance_mode == "soft"
    assert cfg.ipm_transit_balance_mode == "soft"
    assert not cfg.ipm_measurement_bounds and not cfg.ipm_fixed_injection_from_input
    # The soft balance weight may now exceed the median measurement weight tenfold.
    dataclasses.replace(cfg, ipm_balance_weight_factor=1e3)
    with pytest.raises(ValueError, match="ipm_balance_mode"):
        dataclasses.replace(cfg, ipm_balance_mode="strict")


def test_huber_threshold_modes(monkeypatch):
    import gridstate.api as api
    from gridstate.pipeline import PipelineConfig

    assert PipelineConfig().ipm_huber_threshold == "fixed"
    with pytest.raises(ValueError, match="ipm_huber_threshold"):
        PipelineConfig(algorithm="ipm_pd", ipm_huber_threshold="frozen")
    seen = []
    real = api.solve_ipm_pd

    def spy(*args, **kwargs):
        seen.append(kwargs["huber_adaptive_k"])
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "solve_ipm_pd", spy)
    for mode in ("fixed", "adaptive"):
        api.estimate(
            _build_net_with_transit(),
            algorithm="ipm_pd",
            include_quality_summary=False,
            huber_threshold=mode,
        )
    assert seen == [0.0, 2.0]


def _spy_guard(monkeypatch) -> list[dict]:
    import gridstate.api as api

    seen: list[dict] = []
    real = api.solve_ipm_pd

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "solve_ipm_pd", spy)
    return seen


def test_voltage_guard_defaults_follow_the_settings(monkeypatch):
    from gridstate.api import VoltageGuard
    from gridstate.pipeline.passes import _estimate_kwargs

    cfg = PipelineConfig(algorithm="ipm_pd")
    assert _estimate_kwargs(cfg, init="flat")["voltage_guard_settings"] == VoltageGuard()
    g = VoltageGuard()
    assert (g.unset_lo, g.unset_hi) == (g.lo_fallback, g.hi_fallback) == (0.55, 1.8)
    assert (g.lo_min, g.hi_max) == (0.5, 2.0)
    assert (g.v_step_rel, g.angle_step_rad, g.branch_angle_step_rad) == (0.2, 0.6, 0.35)

    seen = _spy_guard(monkeypatch)
    m = _build_net_with_transit()
    nodes = m.nodes.to_numpy()
    nodes["voltage_min"] = 0.0
    nodes["voltage_max"] = 0.0
    nodes["voltage_min"][0] = 0.3 * nodes["voltage_nominal"][0]  # implausible: fallback
    nodes["voltage_max"][1] = 1.05 * nodes["voltage_nominal"][1]
    m.nodes.update_from_array(nodes)
    settings = VoltageGuard(
        lo_fallback=0.6,
        hi_fallback=1.7,
        unset_lo=0.8,
        unset_hi=1.3,
        v_step_rel=0.1,
        angle_step_rad=0.5,
        branch_angle_step_rad=0.3,
    )
    estimate(m, algorithm="ipm_pd", include_quality_summary=False, voltage_guard_settings=settings)
    kw = seen[0]
    lo, hi = kw["state_box_lo"], kw["state_box_hi"]
    assert (lo[0], hi[0]) == (0.6, 1.7)
    assert (lo[1], hi[1]) == pytest.approx((0.6, 1.05))
    assert set(lo[2:]) == {0.8} and set(hi[2:]) == {1.3}
    assert (kw["step_rel_max"], kw["step_abs_max"], kw["step_diff_max"]) == (0.1, 0.5, 0.3)


def test_voltage_guard_config_is_validated():
    with pytest.raises(ValueError, match="ipm_voltage_guard_unset_lo"):
        PipelineConfig(ipm_voltage_guard_unset_lo=1.0, ipm_voltage_guard_unset_hi=1.0)
    with pytest.raises(ValueError, match="ipm_voltage_guard_lo_fallback"):
        PipelineConfig(ipm_voltage_guard_lo_fallback=1.0, ipm_voltage_guard_hi_fallback=1.0)
    old = PipelineConfig(ipm_voltage_guard_unset_lo=0.7, ipm_voltage_guard_unset_hi=1.5)
    assert (old.ipm_voltage_guard_unset_lo, old.ipm_voltage_guard_unset_hi) == (0.7, 1.5)
