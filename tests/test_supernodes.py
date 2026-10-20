"""Nodes joined by switches merged into supernodes (IPM ``merge_breakers_z_ohm_per_kv``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.api import estimate, populate_quality_summary
from gridstate.constants import BranchType, MeasurementObjectType, NodeType
from gridstate.pipeline import PipelineConfig, run
from gridstate.supernodes import DEFAULT_MERGE_Z_OHM_PER_KV, merged_branch_ids
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_VOLTAGE,
)
from tests.test_boxed_balance import _build_net_with_transit


BREAKER = 400
# The breaker reactance of the normalization rule at 110 kV: exactly on the threshold.
X_BREAKER = DEFAULT_MERGE_Z_OHM_PER_KV * 110.0


def _meas(m, mid: int, obj: int, kind: int, value: float, variance: float, **extra) -> None:
    m.measurements.add(
        {
            "id": mid,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": obj,
            "measurement_type": kind,
            "value": value,
            "variance": variance,
            "status": True,
            "quality": 0,
            **extra,
        }
    )


def _net(*, x_breaker: float = X_BREAKER, injection_at_5: bool = True):
    """Transit net 1-2-{3,4} plus node 5 behind a breaker at node 3.

    Node 5 carries a 10 MW / 3 MVAr load; its V and (optionally) its injection
    are measured, and so is the P flow through the breaker.
    """
    m = _build_net_with_transit()
    m.nodes.add(
        {
            "id": 5,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 108.4,
            "load_p": 10.0,
            "load_q": 3.0,
            "load_p_min": 0.0,
            "load_p_max": 50.0,
            "load_q_min": 0.0,
            "load_q_max": 25.0,
            "exist_load": 1,
            "exist_gen": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.branches.add(
        {
            "id": BREAKER,
            "from_node": 3,
            "to_node": 5,
            "resistance": 0.0,
            "reactance": x_breaker,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    _meas(m, 20, 5, KIND_VOLTAGE, 108.4, 0.01)
    if injection_at_5:
        _meas(m, 21, 5, KIND_POWER_INJECTION_P, -10.0, 0.5)
        _meas(m, 22, 5, KIND_POWER_INJECTION_Q, -3.0, 0.5)
    m.measurements.add(
        {
            "id": 23,
            "object_type": int(MeasurementObjectType.BRANCH),
            "object_id": BREAKER,
            "measurement_type": KIND_POWER_P,
            "branch_side": 0,
            "value": 25.0,  # inconsistent on purpose: it must not be used
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    return m


def _solve(m, **kwargs):
    kwargs.setdefault("merge_breakers_z_ohm_per_kv", DEFAULT_MERGE_Z_OHM_PER_KV)
    return estimate(m, algorithm="ipm_pd", huber_c=0.0, **kwargs)


def _node(res, nid: int):
    nodes = res.model.nodes.to_numpy()
    return nodes[nodes["id"] == nid][0]


def _branch(res, bid: int):
    branches = res.model.branches.to_numpy()
    return branches[branches["id"] == bid][0]


def _meas_row(res, mid: int):
    arr = res.model.measurements.to_numpy()
    return arr[arr["id"] == mid][0]


# ---------------------------------------------------------------- detection
def test_merged_branches_are_the_small_ones():
    m = _net()
    nodes, branches = m.nodes.to_numpy(), m.branches.to_numpy()
    assert merged_branch_ids(nodes, branches, DEFAULT_MERGE_Z_OHM_PER_KV).tolist() == [BREAKER]
    # Above the threshold: a branch with its X.
    m.branches.update(BREAKER, reactance=X_BREAKER * 1.01)
    assert merged_branch_ids(m.nodes.to_numpy(), m.branches.to_numpy(), 4.4e-4).size == 0
    # An inactive end is not merged.
    m.branches.update(BREAKER, reactance=X_BREAKER)
    m.nodes.update(5, status=False)
    assert merged_branch_ids(m.nodes.to_numpy(), m.branches.to_numpy(), 4.4e-4).size == 0


def test_config_default_threshold_matches_the_breaker_rule():
    assert PipelineConfig().ipm_merge_breakers_z_ohm_per_kv == pytest.approx(
        DEFAULT_MERGE_Z_OHM_PER_KV
    )


# ---------------------------------------------------------------- solve
def test_supernode_has_one_voltage_and_angle():
    res = _solve(_net())
    assert res.success, res.message
    n3, n5 = _node(res, 3), _node(res, 5)
    assert n3["voltage_magnitude"] == n5["voltage_magnitude"]
    assert n3["voltage_angle"] == n5["voltage_angle"]
    # One V between the two V measurements (108.6 at node 3, 108.4 at node 5).
    assert 108.4 < n3["voltage_magnitude"] < 108.6


def test_without_merging_the_breaker_ends_differ():
    res = _solve(_net(), merge_breakers_z_ohm_per_kv=None)
    assert res.success, res.message
    assert _node(res, 3)["voltage_magnitude"] != _node(res, 5)["voltage_magnitude"]


def test_breaker_flow_is_recovered_and_its_measurement_is_not_used():
    res = _solve(_net())
    load5 = _node(res, 5)["load_p_estimated"]
    assert load5 == pytest.approx(10.0, abs=0.5)
    br = _branch(res, BREAKER)
    # Node 5 is fed through the breaker only: the flow is its load.
    assert br["power_from_p"] == pytest.approx(load5, abs=1e-6)
    assert br["power_to_p"] == pytest.approx(-load5, abs=1e-6)
    assert br["loss_p"] == 0.0
    assert br["current_from"] > 0.0
    assert np.isnan(_meas_row(res, 23)["estimated_si"])


def test_member_injection_measures_the_box_variables_of_its_node():
    res = _solve(_net())
    n5 = _node(res, 5)
    row = _meas_row(res, 21)
    assert row["estimated_si"] == pytest.approx(-n5["load_p_estimated"], abs=1e-9)
    # The injection of a member includes what flows through the merged branch.
    assert n5["p_inj_calc"] == pytest.approx(-n5["load_p_estimated"], abs=1e-6)
    n3 = _node(res, 3)
    assert n3["p_inj_calc"] == pytest.approx(-n3["load_p_estimated"], abs=0.05)


def test_member_injection_without_a_box_variable_is_not_used():
    m = _net()
    # Node 5 becomes transit: its (zero) injection is not a function of the state.
    m.nodes.update(5, exist_load=0, load_p=0.0, load_q=0.0)
    m.measurements.update(21, value=0.0)
    m.measurements.update(22, value=0.0)
    res = _solve(m)
    assert res.success, res.message
    assert np.isnan(_meas_row(res, 21)["estimated_si"])
    assert _branch(res, BREAKER)["power_from_p"] == pytest.approx(0.0, abs=1e-6)


def test_quality_summary_runs_on_the_supernode_state():
    res = _solve(_net(), include_quality_summary=True)
    assert res.chi2 is not None
    nr = res.normalized_residuals
    assert nr is not None
    rn = dict(zip(nr.measurement_id.tolist(), nr.normalized_residual.tolist(), strict=True))
    # V of node 5 and its injection (measured on the box variables) are solved rows.
    assert {20, 21, 22} <= set(rn)
    assert np.isfinite(rn[20]) and np.isfinite(rn[21])
    assert 23 not in rn  # the merged branch flow
    assert not res.observability_warnings
    assert all(row.measurement_id > 0 for row in res.worst_residuals)
    before = res.chi2
    populate_quality_summary(res)
    assert res.chi2 == before


def test_hard_balance_and_primal_ipm_also_merge():
    for kwargs in ({"balance_mode": "hard"}, {}):
        res = estimate(
            _net(),
            algorithm="ipm",
            huber_c=0.0,
            merge_breakers_z_ohm_per_kv=DEFAULT_MERGE_Z_OHM_PER_KV,
            **kwargs,
        )
        assert res.success, res.message
        assert _node(res, 3)["voltage_magnitude"] == _node(res, 5)["voltage_magnitude"]


def test_wls_does_not_merge():
    with pytest.raises(ValueError, match="requires algorithm"):
        estimate(_net(), algorithm="wls", merge_breakers_z_ohm_per_kv=4.4e-4)
    with pytest.raises(ValueError, match="ipm_merge_breakers"):
        PipelineConfig(algorithm="wls", ipm_merge_breakers=True)


def test_no_breaker_to_merge_is_bit_identical():
    m = _net(x_breaker=1.0)
    off = _solve(m, merge_breakers_z_ohm_per_kv=None)
    on = _solve(m)
    assert np.array_equal(off.v_pu, on.v_pu)
    assert np.array_equal(off.delta_rad, on.delta_rad)


def test_pipeline_option():
    cfg = PipelineConfig.no_extras(algorithm="ipm_pd", ipm_merge_breakers=True)
    res = run(_net(), config=cfg)
    assert res.success, res.message
    assert _node(res, 3)["voltage_magnitude"] == _node(res, 5)["voltage_magnitude"]
    assert res.chi2 is not None


def _breaker(m, bid: int, f: int, t: int) -> None:
    m.branches.add(
        {
            "id": bid,
            "from_node": f,
            "to_node": t,
            "resistance": 0.0,
            "reactance": X_BREAKER,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )


def test_breaker_loop_keeps_the_balance_of_every_member_without_circulation():
    m = _net()
    # Node 6 (load 5 MW) closes a loop 3-5-6-3 of breakers.
    m.nodes.add(
        {
            "id": 6,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 108.5,
            "load_p": 5.0,
            "load_q": 1.0,
            "load_p_min": 0.0,
            "load_p_max": 50.0,
            "load_q_min": 0.0,
            "load_q_max": 25.0,
            "exist_load": 1,
            "exist_gen": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    _breaker(m, 401, 5, 6)
    _breaker(m, 402, 6, 3)
    _meas(m, 30, 6, KIND_POWER_INJECTION_P, -5.0, 0.5)
    res = _solve(m)
    assert res.success, res.message
    v = {nid: _node(res, nid)["voltage_magnitude"] for nid in (3, 5, 6)}
    assert v[3] == v[5] == v[6]
    f = {bid: _branch(res, bid)["power_from_p"] for bid in (BREAKER, 401, 402)}
    load5, load6 = _node(res, 5)["load_p_estimated"], _node(res, 6)["load_p_estimated"]
    # Kirchhoff at the members: 3 → 5 → 6 → 3.
    assert f[BREAKER] - f[401] == pytest.approx(load5, abs=1e-6)
    assert f[401] - f[402] == pytest.approx(load6, abs=1e-6)
    # Equal split around the loop: no circulating flow.
    assert f[BREAKER] + f[401] + f[402] == pytest.approx(0.0, abs=1e-6)
    for nid in (5, 6):
        n = _node(res, nid)
        assert n["p_inj_calc"] == pytest.approx(-n["load_p_estimated"], abs=1e-6)


def test_slack_in_a_supernode_fixes_its_angle():
    m = _net()
    m.nodes.add(
        {
            "id": 7,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.3,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    _breaker(m, 403, 7, 1)
    _meas(m, 31, 7, KIND_VOLTAGE, 110.3, 0.01)
    res = _solve(m)
    assert res.success, res.message
    n1, n7 = _node(res, 1), _node(res, 7)
    assert n7["voltage_angle"] == n1["voltage_angle"] == 0.0
    assert n7["voltage_magnitude"] == n1["voltage_magnitude"]
    assert _branch(res, 403)["power_from_p"] == pytest.approx(0.0, abs=1e-6)


def test_merged_transformer_keeps_its_ratio_and_shift():
    m = _net()
    m.branches.update(BREAKER, tap_ratio=1.05, phase_shift=0.02)
    res = _solve(m)
    assert res.success, res.message
    n3, n5 = _node(res, 3), _node(res, 5)
    # V_to = V_from / K, δ_to = δ_from − φ (equal nominal voltages: K = tap).
    assert n5["voltage_magnitude"] == pytest.approx(n3["voltage_magnitude"] / 1.05, rel=1e-12)
    assert n5["voltage_angle"] == pytest.approx(n3["voltage_angle"] - 0.02, abs=1e-12)
    br = _branch(res, BREAKER)
    assert br["power_from_p"] == pytest.approx(_node(res, 5)["load_p_estimated"], abs=1e-6)
    assert br["current_to"] == pytest.approx(br["current_from"] * 1.05, rel=1e-9)


def test_breaker_between_different_nominal_voltages_keeps_the_kv():
    m = _net()
    m.nodes.update(5, voltage_nominal=115.0, voltage_magnitude=108.4)
    res = _solve(m)
    assert res.success, res.message
    assert _node(res, 5)["voltage_magnitude"] == pytest.approx(
        _node(res, 3)["voltage_magnitude"], rel=1e-12
    )


def test_flat_start_propagates_the_ratio_of_a_merged_transformer():
    from gridstate.api import _solve_network
    from gridstate.state import tap_consistent_voltages
    from gridstate.supernodes import start_network

    m = _net()
    m.branches.update(BREAKER, tap_ratio=1.05)
    net, _, sn = _solve_network(m, m.measurements, DEFAULT_MERGE_Z_OHM_PER_KV)
    assert sn is not None
    v = tap_consistent_voltages(start_network(net, sn))
    pos = {int(b): k for k, b in enumerate(net.bus_ids.tolist())}
    assert v[pos[5]] == pytest.approx(v[pos[3]] / 1.05, rel=1e-6)
