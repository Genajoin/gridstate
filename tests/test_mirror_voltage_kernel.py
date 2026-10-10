"""Kernel of ``mirror_voltage`` over contract arrays.

Unlike the cascade functions (which mutate an existing column and then
``update_from_array``), this pseudo-measurement layer ADDS measurement rows.
``_mirror_voltage_on_arrays`` reads ONLY the contract columns of
nodes/branches/measurements and RETURNS ``new_rows: list[dict]``; a thin adapter
calls ``model.measurements.add()`` row by row. These tests check (a) the kernel
on bare ``SE_INPUT`` arrays (without a full model), (b) the exact
collision-skipping id assignment, (c) that the kernel plan matches the actual
effect of the adapter on a model.
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.preprocessing.mirror_voltage import (
    _mirror_voltage_on_arrays,
    mirror_voltage_through_unit_tap_links,
)
from gridstate.preprocessing.pseudo_measurements import (
    PseudoMeasConfig,
    _add_pseudo_measurements_on_arrays,
)
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_VOLTAGE,
    OBJ_NODE,
)


_MID = 200_000_000


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _branches(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.branches.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _meas(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.measurements.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _real_v(mid: int, node_id: int, value: float, variance: float) -> dict:
    return {
        "id": mid,
        "object_type": OBJ_NODE,
        "object_id": node_id,
        "measurement_type": KIND_VOLTAGE,
        "value": value,
        "variance": variance,
        "status": True,
        "is_pseudo": False,
    }


# ---------------------------------------------------------------------------
# Kernel on bare contract arrays
# ---------------------------------------------------------------------------


def test_mirror_core_copies_real_v_both_directions():
    # Nodes: 1 (real V), 2 (bare), 5 (bare), 6 (real V), plus off node 4.
    nodes = _nodes(
        [
            {"id": 1, "status": True},
            {"id": 2, "status": True},
            {"id": 5, "status": True},
            {"id": 6, "status": True},
            {"id": 4, "status": False},
        ]
    )
    meas = _meas(
        [
            _real_v(10, 1, 115.0, 0.5),
            _real_v(11, 6, 121.0, 0.7),
        ]
    )
    branches = _branches(
        [
            # A: trafo tap≈1, from (1 realV) -> to (2 bare) -> mirror onto 2
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
            # B: trafo tap≈1, from (5 bare) -> to (6 realV) -> mirror onto 5 (reverse direction)
            {
                "id": 200,
                "from_node": 5,
                "to_node": 6,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
        ]
    )
    rows = _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID)
    assert len(rows) == 2
    # branch A is first in array order -> id=_MID, copy of node 1 V onto node 2
    assert rows[0]["id"] == _MID
    assert rows[0]["object_id"] == 2
    assert rows[0]["value"] == 115.0
    assert rows[0]["variance"] == 0.5
    assert rows[0]["is_pseudo"] is True
    assert rows[0]["branch_side"] == -1
    assert rows[0]["source_code"] == "mirror_unit_tap"
    # branch B -> id=_MID+1, copy of node 6 V onto node 5
    assert rows[1]["id"] == _MID + 1
    assert rows[1]["object_id"] == 5
    assert rows[1]["value"] == 121.0
    assert rows[1]["variance"] == 0.7


def test_mirror_core_gates_no_spurious_rows():
    nodes = _nodes(
        [
            {"id": 1, "status": True},
            {"id": 2, "status": True},
            {"id": 3, "status": True},
            {"id": 4, "status": False},  # off node
        ]
    )
    meas = _meas([_real_v(10, 1, 115.0, 0.5)])
    branches = _branches(
        [
            # tap≠1 -> skipped
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.05,
                "status": True,
            },
            # off branch -> skipped
            {
                "id": 200,
                "from_node": 1,
                "to_node": 3,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": False,
            },
            # not a transformer (LINE) -> skipped
            {
                "id": 300,
                "from_node": 1,
                "to_node": 3,
                "branch_type": 0,
                "tap_ratio": 1.0,
                "status": True,
            },
            # end on off node 4 -> skipped
            {
                "id": 400,
                "from_node": 1,
                "to_node": 4,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
        ]
    )
    assert _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID) == []


def test_mirror_core_does_not_chain_through_mirrored():
    # 1 has real V; 1->2 mirrors onto 2 (any_v, NOT real); 2->3 must NOT mirror
    # (source 2 is not in real_v_by_node, only in any_v_by_node).
    nodes = _nodes(
        [{"id": 1, "status": True}, {"id": 2, "status": True}, {"id": 3, "status": True}]
    )
    meas = _meas([_real_v(10, 1, 115.0, 0.5)])
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
            {
                "id": 200,
                "from_node": 2,
                "to_node": 3,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
        ]
    )
    rows = _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID)
    assert len(rows) == 1
    assert rows[0]["object_id"] == 2


def test_mirror_core_skips_node_with_existing_pseudo_v():
    # Node 2 already has a pseudo V (any_v) -> no mirroring, although there is no real V.
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": True}])
    meas = _meas(
        [
            _real_v(10, 1, 115.0, 0.5),
            {
                "id": 11,
                "object_type": OBJ_NODE,
                "object_id": 2,
                "measurement_type": KIND_VOLTAGE,
                "value": 110.0,
                "variance": 9.0,
                "status": True,
                "is_pseudo": True,  # pseudo -> goes into any_v_by_node
            },
        ]
    )
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            }
        ]
    )
    assert _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID) == []


def test_mirror_core_free_id_collision_skip():
    # An existing measurement occupies mid_start itself -> allocation starts at mid_start+1
    # (verbatim collision-skip semantics; protects the own 200M id range).
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": True}])
    meas = _meas([_real_v(_MID, 1, 115.0, 0.5)])  # real V on node 1, but id == mid_start
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            }
        ]
    )
    rows = _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID)
    assert len(rows) == 1
    assert rows[0]["id"] == _MID + 1  # _MID is taken -> next free id
    assert rows[0]["object_id"] == 2


# ---------------------------------------------------------------------------
# Kernel plan == adapter effect on the model
# ---------------------------------------------------------------------------


def _build_model():
    from gridstate.constants import BranchType
    from gridstate.working import Working

    m = Working.empty()
    for nid in (1, 2, 5, 6):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": 0,
            }
        )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 0.0,
            "reactance": 0.1,
            "status": True,
            "branch_type": int(BranchType.TRANSFORMER),
            "tap_ratio": 1.0,
        }
    )
    m.branches.add(
        {
            "id": 200,
            "from_node": 5,
            "to_node": 6,
            "resistance": 0.0,
            "reactance": 0.1,
            "status": True,
            "branch_type": int(BranchType.TRANSFORMER),
            "tap_ratio": 1.0,
        }
    )
    m.measurements.add(
        {
            "id": 10,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": 115.0,
            "variance": 0.5,
            "status": True,
            "is_pseudo": False,
        }
    )
    m.measurements.add(
        {
            "id": 11,
            "object_type": OBJ_NODE,
            "object_id": 6,
            "measurement_type": KIND_VOLTAGE,
            "value": 121.0,
            "variance": 0.7,
            "status": True,
            "is_pseudo": False,
        }
    )
    return m


def test_adapter_effect_matches_core_plan():
    """The adapter adds exactly the rows (id, object_id, value, variance) that the kernel returned."""
    m = _build_model()

    plan = _mirror_voltage_on_arrays(
        m.nodes.to_numpy(), m.branches.to_numpy(), m.measurements.to_numpy(), mid_start=_MID
    )
    before = {int(me.id) for me in m.measurements}

    cnt = mirror_voltage_through_unit_tap_links(m, mid_start=_MID)
    assert cnt == {"added": len(plan)}

    added = {int(me.id) for me in m.measurements} - before
    assert added == {int(r["id"]) for r in plan}

    by_id = {int(me.id): me for me in m.measurements}
    for r in plan:
        me = by_id[int(r["id"])]
        assert int(me.object_id) == r["object_id"]
        assert float(me.value) == r["value"]
        assert float(me.variance) == r["variance"]
        assert bool(me.is_pseudo) is True


# ---------------------------------------------------------------------------
# add_pseudo_measurements: kernel on bare arrays
# ---------------------------------------------------------------------------


def test_add_pseudo_core_v_and_injection_priors():
    # Node 1: active, no measurements, vn=110 vm=115, pg=10 pn=4 -> p_inj=6.
    nodes = _nodes(
        [
            {
                "id": 1,
                "status": True,
                "node_type": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 115.0,
                "generation_p": 10.0,
                "load_p": 4.0,
                "generation_q": 0.0,
                "load_q": 0.0,
            }
        ]
    )
    branches = _branches([])
    meas = _meas([])
    rows, stats = _add_pseudo_measurements_on_arrays(
        nodes, branches, meas, None, PseudoMeasConfig(mid_start=300_000_000)
    )
    assert stats["v_priors_added"] == 1
    assert stats["zero_inj_added"] == 1
    assert len(rows) == 3  # V + P_inj + Q_inj
    # V prior: value=vm=115, σ=(0.05·110)²=30.25
    assert rows[0]["measurement_type"] == KIND_VOLTAGE
    assert rows[0]["value"] == 115.0
    assert rows[0]["variance"] == (0.05 * 110.0) ** 2
    assert rows[0]["id"] == 300_000_000
    # P_inj=pg-pn=6, not a transit node -> var=zero_inj·load_loose=100·10=1000
    assert rows[1]["measurement_type"] == KIND_POWER_INJECTION_P
    assert rows[1]["value"] == 6.0
    assert rows[1]["variance"] == 1000.0
    assert rows[1]["id"] == 300_000_001
    # Q_inj=0
    assert rows[2]["measurement_type"] == KIND_POWER_INJECTION_Q
    assert rows[2]["value"] == 0.0
    assert rows[2]["id"] == 300_000_002


def test_add_pseudo_core_transit_node_tight_prior():
    # Transit node (pg=pn=qg=qn=0, no node_load_props) -> var=zero_inj (tight, not loose).
    nodes = _nodes(
        [
            {
                "id": 1,
                "status": True,
                "node_type": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 0.0,
            },
            # another loaded node so that the model is NOT considered empty
            {
                "id": 2,
                "status": True,
                "node_type": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "generation_p": 50.0,
            },
        ]
    )
    rows, stats = _add_pseudo_measurements_on_arrays(nodes, _branches([]), _meas([]), None)
    # node 1: V=vn (vm=0 -> fallback), P_inj=0 var=100 (transit); node 2: V=110, P_inj=50 var=1000
    inj1 = [
        r for r in rows if r["object_id"] == 1 and r["measurement_type"] == KIND_POWER_INJECTION_P
    ]
    assert inj1[0]["variance"] == 100.0  # transit -> tight
    assert inj1[0]["value"] == 0.0
    v1 = [r for r in rows if r["object_id"] == 1 and r["measurement_type"] == KIND_VOLTAGE]
    assert v1[0]["value"] == 110.0  # vm=0 -> fallback to vn
    assert stats["v_priors_added"] == 2
