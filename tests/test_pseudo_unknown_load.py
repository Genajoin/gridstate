"""Zero injection prior of a load without data, loosened where measurements fix it.

``add_pseudo_measurements(loosen_unknown_load=True)``: a node with load, no load
box and nothing materialized gets a zero P/Q injection prior that states the
absence of data. Its variance is multiplied by ``boundary_inj_loose_factor``
when every active incident branch carries a real P flow (for P) or a real Q
flow / real V at both ends (for Q).
"""

from __future__ import annotations

import pytest

from gridstate.preprocessing import add_pseudo_measurements
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


def _model(*, v_at_3: bool = True, box_at_2: bool = False, load_at_3: float = 0.0):
    """Chain 1 - 2 - 3, all with load. Node 1 has a load box.

    Real measurements: P flow on both branches, Q flow on 1-2 only, V at 2 and
    (optionally) 3.
    """
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid in (1, 2, 3):
        box = (10.0, 50.0) if nid == 1 or (nid == 2 and box_at_2) else (0.0, 0.0)
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 500.0,
                "voltage_magnitude": 500.0,
                "status": True,
                "node_type": int(NodeType.SLACK if nid == 1 else NodeType.PQ),
                "exist_load": True,
                "load_p_min": box[0],
                "load_p_max": box[1],
                "load_q_min": box[0],
                "load_q_max": box[1],
                "load_p": load_at_3 if nid == 3 else 0.0,
                "load_q": 0.0,
            }
        )
    for bid, f, t in ((12, 1, 2), (23, 2, 3)):
        m.branches.add(
            {
                "id": bid,
                "from_node": f,
                "to_node": t,
                "status": True,
                "branch_type": 0,
                "tap_ratio": 1.0,
                "resistance": 1.0,
                "reactance": 10.0,
            }
        )
    rows = [
        (1, OBJ_BRANCH, 12, KIND_POWER_P, 100.0),
        (2, OBJ_BRANCH, 23, KIND_POWER_P, 60.0),
        (3, OBJ_BRANCH, 12, KIND_POWER_Q, 20.0),
        (4, OBJ_NODE, 2, KIND_VOLTAGE, 505.0),
    ]
    if v_at_3:
        rows.append((5, OBJ_NODE, 3, KIND_VOLTAGE, 503.0))
    for mid, ot, oid, mt, value in rows:
        m.measurements.add(
            {
                "id": mid,
                "object_type": ot,
                "object_id": oid,
                "measurement_type": mt,
                "branch_side": 0 if ot == OBJ_BRANCH else -1,
                "value": value,
                "variance": 4.0,
                "weight": 0.25,
                "status": True,
            }
        )
    return m


def _prior_variance(m) -> dict[tuple[int, int], float]:
    arr = m.measurements.to_numpy()
    sel = arr["is_pseudo"].astype(bool) & (arr["object_type"] == OBJ_NODE)
    return {
        (int(r["object_id"]), int(r["measurement_type"])): float(r["variance"])
        for r in arr[sel]
        if int(r["measurement_type"]) in (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q)
    }


def _run(m, **kw):
    add_pseudo_measurements(m, boundary_inj_loose_factor=1.0e4, **kw)
    return _prior_variance(m)


P, Q = KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q


def test_loosens_prior_fixed_by_flows_and_voltages() -> None:
    base = _run(_model())
    var = _run(_model(), loosen_unknown_load=True)
    # node 2: P flows on both branches; Q flow on 1-2, V at both ends of 2-3
    assert var[(2, P)] == pytest.approx(base[(2, P)] * 1.0e4)
    assert var[(2, Q)] == pytest.approx(base[(2, Q)] * 1.0e4)
    # node 3: P flow on its only branch, V at both ends
    assert var[(3, P)] == pytest.approx(base[(3, P)] * 1.0e4)
    assert var[(3, Q)] == pytest.approx(base[(3, Q)] * 1.0e4)
    # node 1 has a load box: its prior is data
    assert var[(1, P)] == pytest.approx(base[(1, P)])


def test_q_prior_kept_without_q_flow_or_voltages() -> None:
    base = _run(_model(v_at_3=False))
    var = _run(_model(v_at_3=False), loosen_unknown_load=True)
    assert var[(3, P)] == pytest.approx(base[(3, P)] * 1.0e4)
    assert var[(3, Q)] == pytest.approx(base[(3, Q)])
    assert var[(2, Q)] == pytest.approx(base[(2, Q)])


def test_prior_with_data_is_kept() -> None:
    # load box declared on node 2, a materialized load on node 3
    base = _run(_model(box_at_2=True, load_at_3=25.0))
    var = _run(_model(box_at_2=True, load_at_3=25.0), loosen_unknown_load=True)
    assert var[(2, P)] == pytest.approx(base[(2, P)])
    assert var[(3, P)] == pytest.approx(base[(3, P)])


def test_off_by_default() -> None:
    assert _run(_model()) == _run(_model(), loosen_unknown_load=False)
