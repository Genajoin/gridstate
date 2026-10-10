"""Balance weight of nodes whose injection range is fully set by their boxes."""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags

from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.state import StateLayout
from gridstate.units import model_to_pu
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_VOLTAGE,
    OBJ_NODE,
    MeasurementIndex,
)


N_BALANCE = 4  # all four nodes of the test network are active
DATA_SIGMA2 = np.array([1e-4, 4e-4])  # median 2.5e-4
SOFT = 2.5e-4 / 0.1  # default balance_weight_factor


def _build_line(m: Working, bid: int, f: int, t: int) -> None:
    m.branches.add(
        {
            "id": bid,
            "from_node": f,
            "to_node": t,
            "resistance": 3.0,
            "reactance": 15.0,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )


def _build_net_with_transit() -> Working:
    """4 nodes: slack (gen) - transit - load - load.

    Node 2 is a pure transit node (exist_load=0, exist_gen=0) carrying the
    whole flow to nodes 3-4.
    """
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "generation_p": 60.0,
            "generation_q": 20.0,
            "generation_p_min": 0.0,
            "generation_p_max": 200.0,
            "generation_q_min": -100.0,
            "generation_q_max": 100.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 109.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    for nid, vm in ((3, 108.5), (4, 108.0)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": vm,
                "load_p": 30.0,
                "load_q": 10.0,
                "load_p_min": 0.0,
                "load_p_max": 100.0,
                "load_q_min": 0.0,
                "load_q_max": 50.0,
                "exist_load": 1,
                "exist_gen": 0,
                "status": True,
                "node_type": int(NodeType.PQ),
            }
        )
    _build_line(m, 100, 1, 2)
    _build_line(m, 200, 2, 3)
    _build_line(m, 300, 2, 4)

    mid = 1
    for nid, vm in [(1, 110.4), (2, 109.2), (3, 108.6), (4, 108.1)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_VOLTAGE,
                "value": vm,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1
    for nid, p, q in [(3, -30.0, -10.0), (4, -30.0, -10.0)]:
        for kind, val in ((KIND_POWER_INJECTION_P, p), (KIND_POWER_INJECTION_Q, q)):
            m.measurements.add(
                {
                    "id": mid,
                    "object_type": OBJ_NODE,
                    "object_id": nid,
                    "measurement_type": kind,
                    "value": val,
                    "variance": 0.5,
                    "status": True,
                    "quality": 0,
                }
            )
            mid += 1
    return m


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
    setup = build_ipm_setup(
        model,
        network_pu,
        np.ones(2),
        diags(DATA_SIGMA2).tocsr(),
        mi,
        layout_base=layout,
        **kwargs,
    )
    sigma2 = setup.r_matrix.diagonal()[2:]
    return sigma2[:N_BALANCE], sigma2[N_BALANCE : 2 * N_BALANCE]


def test_off_by_default():
    p_rows, q_rows = _setup(_build_net_with_transit())
    assert np.allclose(p_rows, SOFT)
    assert np.allclose(q_rows, SOFT)
    assert PipelineConfig().ipm_boxed_balance_weight_factor == 0.0


def test_boxed_nodes_get_the_weighted_row():
    # Node order: 1 slack (generation box), 2 transit, 3 and 4 loads with a box.
    p_rows, q_rows = _setup(_build_net_with_transit(), boxed_balance_weight_factor=10.0)
    tight = 2.5e-4 / 10.0
    assert np.allclose(p_rows, [tight, SOFT, tight, tight])
    assert np.allclose(q_rows, [tight, SOFT, tight, tight])


def test_never_looser_than_the_soft_row():
    p_rows, q_rows = _setup(_build_net_with_transit(), boxed_balance_weight_factor=0.01)
    assert np.allclose(p_rows, SOFT)
    assert np.allclose(q_rows, SOFT)


def test_unset_bound_leaves_the_row_soft():
    m = _build_net_with_transit()
    # A 0/0 pair means "not set": the node's P freedom is not described by data.
    m.nodes.update(3, {"load_p_min": 0.0, "load_p_max": 0.0})
    p_rows, q_rows = _setup(m, boxed_balance_weight_factor=10.0)
    assert p_rows[2] == SOFT
    assert q_rows[2] < SOFT


def test_every_declared_part_must_be_bounded():
    m = _build_net_with_transit()
    # Generation declared on a load node without its bounds: P and Q stay soft.
    m.nodes.update(4, {"exist_gen": 1})
    p_rows, q_rows = _setup(m, boxed_balance_weight_factor=10.0)
    assert p_rows[3] == SOFT
    assert q_rows[3] == SOFT


def test_compensator_active_power_is_not_a_box_part():
    m = _build_net_with_transit()
    # Reactive-only generation [0, 0] P / [-20, 20] Q next to the load box.
    m.nodes.update(
        3,
        {
            "exist_gen": 1,
            "generation_p_min": 0.0,
            "generation_p_max": 0.0,
            "generation_q_min": -20.0,
            "generation_q_max": 20.0,
        },
    )
    p_rows, q_rows = _setup(m, boxed_balance_weight_factor=10.0)
    assert p_rows[2] < SOFT
    assert q_rows[2] < SOFT
