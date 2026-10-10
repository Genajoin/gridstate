"""Tests of the WLS algorithm (``gridstate.algorithms.wls.solve_wls``) and its wrapper
``estimate()``.

Approach to the synthetic data:
    1. Take an arbitrary network and fix the "true" state ``E_true``
       (V, δ); slack has zero angle, V=1.0 p.u.
    2. Obtain "ideal" measurements ``z = h(E_true)`` without noise
       through ``BaseAlgebra.evaluate_h``.
    3. Run ``solve_wls`` from a flat start.
    4. Check ``E_se ≈ E_true``.

The convergence criterion (<=5 iterations, ``max|ΔE|<1e-6``) is checked on a 3-node
network with an overdetermined measurement system.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.algorithms.wls import solve_wls
from gridstate.api import estimate
from gridstate.state import StateLayout, flat_start, pack, unpack
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    build_z_and_r,
)


# ----------------------------------------------------------- helpers
def _three_bus_for_se():
    """3-node network. Slack=1; PQ nodes 2, 3. Enough lines for the test."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 6.05,
            "reactance": 30.25,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 1001,
            "ti_q_from": 1002,
            "ti_p_to": 1003,
            "ti_q_to": 1004,
        }
    )
    m.branches.add(
        {
            "id": 200,
            "from_node": 2,
            "to_node": 3,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 2001,
            "ti_q_from": 2002,
            "ti_p_to": 2003,
            "ti_q_to": 2004,
        }
    )
    m.branches.add(
        {
            "id": 300,
            "from_node": 1,
            "to_node": 3,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 3001,
            "ti_q_from": 3002,
            "ti_p_to": 3003,
            "ti_q_to": 3004,
        }
    )
    return m


def _generate_ideal_measurements(model, e_true: np.ndarray, layout: StateLayout):
    """Generate a measurement collection from ``z = h(E_true)``.

    Coverage: V at all nodes, P/Q injections at all nodes, P/Q at the "from" end of each
    branch. The measurements form an overdetermined system (m > 2n-1).
    """
    from gridstate.working import Working

    pu = model_to_pu(model)
    ybus, yf, _yt = build_ybus(pu)
    delta_true, v_true = unpack(e_true, layout)

    coll = Working.empty().measurements
    next_id = 1

    # V
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        v_kv = float(v_true[pos] * pu.bus_vn_kv[pos])
        coll.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_VOLTAGE,
                "value": v_kv,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1

    # P_inj, Q_inj at all nodes (slack included, so that the system is
    # overdetermined in angles). In real problems P/Q injections are usually given
    # only at load/generation nodes.
    V = v_true * np.exp(1j * delta_true)
    Sbus = V * np.conj(ybus @ V)
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        p_mw = float(Sbus[pos].real * BASE_MVA)
        q_mvar = float(Sbus[pos].imag * BASE_MVA)
        coll.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": p_mw,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        coll.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": q_mvar,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1

    # P_from, Q_from for each branch; set ti_p_from / ti_q_from in the branch.
    branches_arr = model.branches.to_numpy()
    If = yf @ V
    Sf = V[pu.from_idx] * np.conj(If)
    for i, b in enumerate(branches_arr):
        if not b["status"]:
            continue
        branch_id = int(b["id"])
        p_mw = float(Sf[i].real * BASE_MVA)
        q_mvar = float(Sf[i].imag * BASE_MVA)
        meas_p_id = next_id
        coll.add(
            {
                "id": meas_p_id,
                "object_type": OBJ_BRANCH,
                "object_id": branch_id,
                "measurement_type": KIND_POWER_P,
                "value": p_mw,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        meas_q_id = next_id
        coll.add(
            {
                "id": meas_q_id,
                "object_type": OBJ_BRANCH,
                "object_id": branch_id,
                "measurement_type": KIND_POWER_Q,
                "value": q_mvar,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        # bind the measurements to the branch via ti_p_from / ti_q_from
        model.branches.update(branch_id, {"ti_p_from": meas_p_id, "ti_q_from": meas_q_id})

    return coll


# ---------------------------------------------------------- basic recovery
def test_solve_wls_recovers_true_state_no_noise() -> None:
    """Without noise WLS must recover ``E_true`` exactly in a few iterations."""
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)

    rng = np.random.default_rng(0)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.05, 0.05, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.05, 0.05, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)

    coll = _generate_ideal_measurements(m, e_true, layout)
    pu = model_to_pu(m)  # recreate after the branch .update()
    ybus, yf, yt = build_ybus(pu)
    z, R, idx = build_z_and_r(m, coll, pu)

    e_init = flat_start(layout)
    res = solve_wls(
        e_init,
        z,
        R,
        ybus,
        yf,
        yt,
        idx,
        layout,
        pu,
        tolerance=1e-8,
        max_iterations=20,
    )

    assert res.success, f"WLS did not converge in {res.iterations} iterations"
    np.testing.assert_allclose(res.x, e_true, atol=1e-6)


def test_solve_wls_meets_iteration_budget() -> None:
    """Convergence criterion: WLS converges in <=5 iterations at σ=0.01 p.u."""
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)

    rng = np.random.default_rng(123)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.05, 0.05, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.03, 0.03, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)

    coll = _generate_ideal_measurements(m, e_true, layout)
    # Add a small amount of noise.
    sigma = 0.01
    for meas in coll:
        noise = rng.normal(0.0, sigma * abs(float(meas.value) + 1e-6))
        meas.value = float(meas.value) + noise

    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    z, R, idx = build_z_and_r(m, coll, pu)
    e_init = flat_start(layout)
    res = solve_wls(
        e_init,
        z,
        R,
        ybus,
        yf,
        yt,
        idx,
        layout,
        pu,
        tolerance=1e-6,
        max_iterations=20,
    )
    assert res.success
    assert res.iterations <= 5, f"expected <=5 iterations, got {res.iterations}"
    # Close to the true state (spread ~ σ).
    np.testing.assert_allclose(res.x, e_true, atol=0.05)


# ------------------------------------------------------------- estimate API
def test_estimate_writes_results_back_to_model() -> None:
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)

    rng = np.random.default_rng(7)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.04, 0.04, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.02, 0.02, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)

    coll = _generate_ideal_measurements(m, e_true, layout)
    result = estimate(m, coll, algorithm="wls", init="flat", tolerance=1e-8)
    assert result.success
    assert result.iterations >= 1

    # Check that V_kV, δ_rad are written to the model.
    nodes_arr = m.nodes.to_numpy()
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        row = nodes_arr[nodes_arr["id"] == node_id][0]
        v_kv_expected = v_true[pos] * pu.bus_vn_kv[pos]
        assert row["voltage_magnitude"] == pytest.approx(v_kv_expected, abs=1e-4)
        assert row["voltage_angle"] == pytest.approx(delta_true[pos], abs=1e-6)

    # Branch flows/currents must be populated.
    branches_arr = m.branches.to_numpy()
    assert np.all(np.abs(branches_arr["power_from_p"]) > 0)
    assert np.all(np.abs(branches_arr["current_from"]) > 0)


def test_estimate_init_flat_default() -> None:
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    e_true = pack(np.zeros(pu.n_bus), np.ones(pu.n_bus), layout)
    coll = _generate_ideal_measurements(m, e_true, layout)
    # Trivial case: truth = flat -> 1 iteration.
    result = estimate(m, coll, init="flat", tolerance=1e-10)
    assert result.success
    assert result.iterations <= 2


def test_estimate_init_results_uses_model_state() -> None:
    """init='results' uses voltage_magnitude/angle from the model."""
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    rng = np.random.default_rng(99)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.05, 0.05, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.04, 0.04, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)
    coll = _generate_ideal_measurements(m, e_true, layout)

    # Put into model.nodes the current state, close to the truth.
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        m.nodes.update(
            int(node_id),
            {
                "voltage_magnitude": float(v_true[pos] * pu.bus_vn_kv[pos]),
                "voltage_angle": float(delta_true[pos]),
            },
        )
    result = estimate(m, coll, init="results", tolerance=1e-10)
    assert result.success
    # Initialising from the truth must converge faster than a flat start.
    assert result.iterations <= 3


def test_estimate_unsupported_algorithm_raises() -> None:
    m = _three_bus_for_se()
    with pytest.raises(NotImplementedError, match=r"lav"):
        estimate(m, algorithm="lav")  # type: ignore[arg-type]


def test_estimate_zero_injection_unsupported() -> None:
    m = _three_bus_for_se()
    with pytest.raises(NotImplementedError, match=r"zero_injection"):
        estimate(m, algorithm="wls", zero_injection="aux_bus")


# ------------------------------------------------------------- empty input
def test_solve_wls_empty_measurements_warns() -> None:
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)
    from gridstate.working import Working

    z, R, idx = build_z_and_r(m, Working.empty().measurements, pu)
    res = solve_wls(flat_start(layout), z, R, ybus, yf, yt, idx, layout, pu)
    assert not res.success
    assert res.iterations == 0
    assert np.isnan(res.objective)


def test_solve_wls_singular_returns_failure() -> None:
    """With a singular gain matrix WLS must return ``success=False``
    rather than "iter=1, J=NaN, success=True" caused by an empty step.

    Scenario: in the 3-node network place only a single V measurement, so the system is
    underdetermined (rank H << n_state=5). spsolve(G, ...) fails on the
    singular matrix; the early exit must not be interpreted as
    success.
    """
    from gridstate.working import Working

    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)

    coll = Working.empty().measurements
    coll.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": 110.0,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    z, R, idx = build_z_and_r(m, coll, pu)
    res = solve_wls(
        flat_start(layout),
        z,
        R,
        ybus,
        yf,
        yt,
        idx,
        layout,
        pu,
        max_iterations=10,
    )
    assert not res.success, "WLS must not report success on a singular problem"
    assert np.isfinite(res.objective), f"J must be a number, not {res.objective}"
