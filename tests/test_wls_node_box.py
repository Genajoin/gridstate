"""Node boxes of WLS (``estimate(..., wls_node_box=True)``).

A node injects only what its generation and load allow, a transit node
nothing; the boxes are enforced by an active set of bound rows once the
unconstrained solve has converged (``gridstate.algorithms.wls_box``).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.algorithms.wls_box import (
    BoxActiveSet,
    NodeBox,
    node_injection_boxes,
    voltage_boxes,
)
from gridstate.api import estimate
from gridstate.constants import BranchType, NodeType
from gridstate.units import model_to_pu
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


def _chain(p12: float, p23: float) -> Working:
    """110 kV chain 1 - 2 - 3: node 1 generates (slack), nodes 2 and 3 have load.

    Real measurements: V at node 1, P and Q flows at the start of both lines.
    Node 2 consumes ``p12 − p23`` (minus losses); with ``p23 > p12`` the data
    make it generate, which its load box [0, 50] MW forbids.
    """
    m = Working.empty()
    nodes = (
        (1, NodeType.SLACK, True, False),
        (2, NodeType.PQ, False, True),
        (3, NodeType.PQ, False, True),
    )
    for nid, ntype, gen, load in nodes:
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
                "exist_gen": gen,
                "exist_load": load,
                "generation_p_min": 0.0,
                "generation_p_max": 500.0 if gen else 0.0,
                "generation_q_min": -200.0 if gen else 0.0,
                "generation_q_max": 200.0 if gen else 0.0,
                "load_p_min": 0.0,
                "load_p_max": 50.0 if nid == 2 else (100.0 if load else 0.0),
                "load_q_min": 0.0,
                "load_q_max": 30.0 if load else 0.0,
            }
        )
    for bid, f, t in ((12, 1, 2), (23, 2, 3)):
        m.branches.add(
            {
                "id": bid,
                "from_node": f,
                "to_node": t,
                "resistance": 2.0,
                "reactance": 12.0,
                "tap_ratio": 1.0,
                "phase_shift": 0.0,
                "status": True,
                "branch_type": int(BranchType.LINE),
            }
        )
    rows = [
        (1, OBJ_NODE, 1, KIND_VOLTAGE, 112.0, -1),
        (2, OBJ_BRANCH, 12, KIND_POWER_P, p12, 0),
        (3, OBJ_BRANCH, 12, KIND_POWER_Q, 10.0, 0),
        (4, OBJ_BRANCH, 23, KIND_POWER_P, p23, 0),
        (5, OBJ_BRANCH, 23, KIND_POWER_Q, 5.0, 0),
    ]
    for mid, ot, oid, mt, value, side in rows:
        m.measurements.add(
            {
                "id": mid,
                "object_type": ot,
                "object_id": oid,
                "measurement_type": mt,
                "value": value,
                "variance": 1.0,
                "status": True,
                "quality": 0,
                "branch_side": side,
            }
        )
    return m


def _p_inj(m: Working, nid: int) -> float:
    arr = m.nodes.to_numpy()
    return float(arr["p_inj_calc"][arr["id"] == nid][0])


def test_boxes_from_the_node_markup():
    m = _chain(60.0, 30.0)
    arr = m.nodes.to_numpy()
    # node 3 becomes a transit node, node 2 gets a compensator-like generation
    arr["exist_load"][2] = False
    m.nodes.update_from_array(arr)
    pu = model_to_pu(m)
    box = node_injection_boxes(m.nodes.to_numpy(), pu, base_mva=100.0)
    got = {
        (int(pu.bus_ids[p]), int(k)): (lo * 100.0, hi * 100.0)
        for p, k, lo, hi in zip(box.bus_pos, box.kind, box.lo, box.hi, strict=True)
    }
    assert got[(1, KIND_POWER_INJECTION_P)] == (0.0, 500.0)
    assert got[(1, KIND_POWER_INJECTION_Q)] == (-200.0, 200.0)
    assert got[(2, KIND_POWER_INJECTION_P)] == (-50.0, 0.0)
    assert got[(2, KIND_POWER_INJECTION_Q)] == (-30.0, 0.0)
    assert got[(3, KIND_POWER_INJECTION_P)] == (0.0, 0.0)
    assert got[(3, KIND_POWER_INJECTION_Q)] == (0.0, 0.0)


def test_unset_side_stays_open():
    m = _chain(60.0, 30.0)
    arr = m.nodes.to_numpy()
    arr["load_p_max"][1] = 9999.0  # sentinel on one side only: kept as data
    arr["load_q_min"][1] = 0.0
    arr["load_q_max"][1] = 0.0  # pair never filled: unset, no Q row
    m.nodes.update_from_array(arr)
    pu = model_to_pu(m)
    box = node_injection_boxes(m.nodes.to_numpy(), pu, base_mva=100.0)
    rows = {(int(pu.bus_ids[p]), int(k)) for p, k in zip(box.bus_pos, box.kind, strict=True)}
    assert (2, KIND_POWER_INJECTION_P) in rows
    assert (2, KIND_POWER_INJECTION_Q) not in rows


def test_box_keeps_a_load_from_generating():
    plain = _chain(30.0, 60.0)
    estimate(plain, algorithm="wls", huber_c=0.0)
    assert _p_inj(plain, 2) > 20.0  # the data make the load node generate

    boxed = _chain(30.0, 60.0)
    res = estimate(boxed, algorithm="wls", huber_c=0.0, wls_node_box=True)
    assert res.success
    assert -50.0 - 0.05 <= _p_inj(boxed, 2) <= 0.05
    assert "node boxes" in res.message


def test_solution_inside_the_boxes_is_plain_wls_bit_for_bit():
    plain = _chain(60.0, 30.0)
    boxed = _chain(60.0, 30.0)
    r1 = estimate(plain, algorithm="wls", huber_c=0.0)
    r2 = estimate(boxed, algorithm="wls", huber_c=0.0, wls_node_box=True)
    assert r1.iterations == r2.iterations
    assert np.array_equal(r1.v_pu, r2.v_pu)
    assert np.array_equal(r1.delta_rad, r2.delta_rad)
    assert "stage 0" in r2.message


def _set(lo: list[float], hi: list[float]) -> BoxActiveSet:
    n = len(lo)
    box = NodeBox(
        bus_pos=np.arange(n, dtype=np.int64),
        kind=np.full(n, KIND_POWER_INJECTION_P, dtype=np.int64),
        lo=np.asarray(lo),
        hi=np.asarray(hi),
    )
    return BoxActiveSet(box, None, sigma2_schedule=(1.0,), tol=1e-3, max_releases=1)  # type: ignore[arg-type]


def test_active_set_pins_and_releases():
    s = _set([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    assert s.start(np.array([-0.5, 0.5, 0.2]))
    assert s.state.tolist() == [-1, 0, 1]
    # pulled inside at a converged point: released once, the point box never
    assert s.release(np.array([0.3, 0.5, -0.1]))
    assert s.state.tolist() == [0, 0, 1]
    assert s.pin(np.array([-0.2, 0.5, 0.0]))
    assert s.state.tolist() == [-1, 0, 1]
    # max_releases reached: the row stays pinned
    assert not s.release(np.array([0.3, 0.5, -0.1]))
    assert s.state.tolist() == [-1, 0, 1]


def test_no_start_inside_the_boxes():
    s = _set([0.0], [1.0])
    assert not s.start(np.array([0.5]))
    assert s.stage == 0


def test_voltage_rows():
    box = voltage_boxes(np.array([0.9, 0.95]), np.array([1.1, 1.05]))
    assert box.kind.tolist() == [KIND_VOLTAGE, KIND_VOLTAGE]
    assert box.concat(box).bus_pos.tolist() == [0, 1, 0, 1]


def test_schedule_must_be_positive():
    with pytest.raises(ValueError, match="sigma2_schedule"):
        BoxActiveSet(  # type: ignore[arg-type]
            NodeBox(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0)),
            None,
            sigma2_schedule=(1.0, 0.0),
            tol=1e-3,
            max_releases=1,
        )


def test_pipeline_config_fields():
    from gridstate.pipeline.config import PipelineConfig
    from gridstate.pipeline.passes import _estimate_kwargs

    cfg = PipelineConfig(algorithm="wls", wls_node_box=True, wls_node_box_variance_mw=0.1)
    kw = _estimate_kwargs(cfg, init="flat")
    assert kw["wls_node_box"] is True
    assert kw["wls_node_box_variance_mw"] == 0.1
    assert "wls_node_box" not in _estimate_kwargs(PipelineConfig(algorithm="wls"), init="flat")
    assert not PipelineConfig().wls_node_box
