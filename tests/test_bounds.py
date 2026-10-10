"""Uniform semantics of unspecified bounds (gridstate.bounds) and its consumers:
clipping of runaway WLS estimates, aggregation of generator limits, IPM box vars."""

from __future__ import annotations

import math

import numpy as np

from gridstate.bounds import (
    SENTINEL_VALUE,
    bounds_unset,
    declared_box,
    is_reactive_only_generation,
    is_sentinel,
    resolve_bounds,
)


# ---------------------------------------------------------------------------
# resolve_bounds / is_sentinel
# ---------------------------------------------------------------------------


def test_resolve_bounds_zero_pair_is_unbounded():
    lo, hi = resolve_bounds(0.0, 0.0)
    assert lo == -math.inf and hi == math.inf


def test_resolve_bounds_sentinel_pair_is_unbounded():
    lo, hi = resolve_bounds(-9999.0, 9999.0)
    assert lo == -math.inf and hi == math.inf


def test_resolve_bounds_half_sentinel_kept_as_is():
    """A half-sentinel pair is kept (the other side is data)."""
    assert resolve_bounds(-9999.0, 120.0) == (-9999.0, 120.0)
    assert resolve_bounds(0.0, 42500.0) == (0.0, 42500.0)


def test_resolve_bounds_large_real_pair_is_data():
    """A real wide pair (equivalent node: load 10..30 GW)
    is NOT treated as a sentinel - the |v|>=9000 threshold used to break this."""
    assert resolve_bounds(10000.0, 30000.0) == (10000.0, 30000.0)


def test_resolve_bounds_inverted_pair_is_unbounded():
    lo, hi = resolve_bounds(50.0, -50.0)
    assert lo == -math.inf and hi == math.inf


def test_resolve_bounds_valid_pair_passthrough():
    assert resolve_bounds(-50.0, 100.0) == (-50.0, 100.0)
    # A half-pair with zero is valid (not both zeros).
    assert resolve_bounds(0.0, 80.0) == (0.0, 80.0)


def test_is_sentinel_exact_value():
    """The sentinel is EXACTLY ±9999, not a threshold: real bounds of the same
    magnitude (10000, 30000, 42500) are data and must not be swallowed."""
    assert is_sentinel(9999.0)
    assert is_sentinel(-9999.0)
    assert not is_sentinel(10000.0)
    assert not is_sentinel(9000.0)
    assert not is_sentinel(42500.0)
    assert SENTINEL_VALUE == 9999.0


# ---------------------------------------------------------------------------
# Clipping of runaway WLS estimates: unspecified bounds no longer zero the estimate
# ---------------------------------------------------------------------------


def _model_unbounded_load():
    """Load-only node WITHOUT specified bounds (numpy default 0,0)."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "LoadNoBounds",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
            # load_*_min/max are not set - they stay (0, 0).
        }
    )
    return m


def test_inj_split_unset_bounds_do_not_zero_estimate():
    """Previously the (0,0) pair clipped the estimate to zero - the load was lost."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _model_unbounded_load()
    m.nodes.update(1, {"p_inj_calc": -55.0, "q_inj_calc": -21.0})
    write_node_estimates_from_inj(m)
    n = m.nodes.get_by_id(1)
    assert n.load_p_estimated == 55.0
    assert n.load_q_estimated == 21.0


def test_inj_split_sentinel_bounds_do_not_clip():
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _model_unbounded_load()
    m.nodes.update(
        1,
        {
            "load_p_min": -9999.0,
            "load_p_max": 9999.0,
            "load_q_min": -9999.0,
            "load_q_max": 9999.0,
            "p_inj_calc": -300.0,
            "q_inj_calc": 90.0,
        },
    )
    write_node_estimates_from_inj(m)
    n = m.nodes.get_by_id(1)
    assert n.load_p_estimated == 300.0
    assert n.load_q_estimated == -90.0  # a capacitive load is not clipped by junk values


def test_inj_split_valid_bounds_still_clip():
    """Valid bounds still clip (behaviour regression)."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _model_unbounded_load()
    m.nodes.update(
        1,
        {
            "load_p_min": 0.0,
            "load_p_max": 80.0,
            "p_inj_calc": -200.0,
            "q_inj_calc": 0.0,
        },
    )
    write_node_estimates_from_inj(m)
    n = m.nodes.get_by_id(1)
    assert n.load_p_estimated == 80.0


# ---------------------------------------------------------------------------
# Generator limit aggregation: sentinels are not summed
# ---------------------------------------------------------------------------


def _nodes_and_gens(gen_rows):
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add({"id": 1, "name": "N1", "voltage_nominal": 110.0, "status": True})
    for g in gen_rows:
        m.generators.add(g)
    return m


def test_aggregate_sentinel_limits_poison_node_pair():
    """A single generator with sentinel Q limits -> node Q pair = ±9999,
    not a junk sum; the P pair (all valid) is the ordinary sum."""
    from gridstate.telemetry.generators import aggregate_generators_to_node

    m = _nodes_and_gens(
        [
            {
                "id": 11,
                "node_id": 1,
                "status": True,
                "power_output": 40.0,
                "reactive_output": 10.0,
                "power_min": 0.0,
                "power_max": 60.0,
                "reactive_min": -50.0,
                "reactive_max": 100.0,
            },
            {
                "id": 12,
                "node_id": 1,
                "status": True,
                "power_output": 30.0,
                "reactive_output": 5.0,
                "power_min": 10.0,
                "power_max": 40.0,
                "reactive_min": -9999.0,
                "reactive_max": 9999.0,
            },
        ]
    )
    stats = aggregate_generators_to_node(m)
    n = m.nodes.get_by_id(1)
    assert n.generation_p == 70.0
    assert n.generation_q == 15.0
    assert (n.generation_p_min, n.generation_p_max) == (10.0, 100.0)
    # Q range is unknown -> sentinel, not [-10049, 10099].
    assert (n.generation_q_min, n.generation_q_max) == (-9999.0, 9999.0)
    assert stats["sentinel_q_nodes"] == 1
    assert stats["sentinel_p_nodes"] == 0


def test_aggregate_all_valid_limits_summed():
    from gridstate.telemetry.generators import aggregate_generators_to_node

    m = _nodes_and_gens(
        [
            {
                "id": 11,
                "node_id": 1,
                "status": True,
                "power_output": 40.0,
                "reactive_output": 10.0,
                "power_min": 0.0,
                "power_max": 60.0,
                "reactive_min": -20.0,
                "reactive_max": 30.0,
            },
            {
                "id": 12,
                "node_id": 1,
                "status": True,
                "power_output": 30.0,
                "reactive_output": 5.0,
                "power_min": 10.0,
                "power_max": 40.0,
                "reactive_min": -10.0,
                "reactive_max": 15.0,
            },
        ]
    )
    aggregate_generators_to_node(m)
    n = m.nodes.get_by_id(1)
    assert (n.generation_q_min, n.generation_q_max) == (-30.0, 45.0)


# ---------------------------------------------------------------------------
# IPM box vars: full coverage of exist_* nodes
# ---------------------------------------------------------------------------


def _ipm_setup_for(node_kwargs):
    """build_ipm_setup on a minimal 2-node model (slack + test node)."""
    from scipy.sparse import csr_matrix

    from gridstate.constants import NodeType
    from gridstate.preprocessing.ipm_setup import build_ipm_setup
    from gridstate.state import StateLayout
    from gridstate.units import model_to_pu
    from gridstate.working import Working
    from gridstate.z_vector import MeasurementIndex

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "Slack",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "name": "Test",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "status": True,
            "node_type": int(NodeType.PQ),
            **node_kwargs,
        }
    )
    m.branches.add(
        {
            "id": 1,
            "from_node": 1,
            "to_node": 2,
            "resistance": 1.0,
            "reactance": 10.0,
            "status": True,
        }
    )
    network_pu = model_to_pu(m)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)
    z = np.zeros(0, dtype=np.float64)
    r = csr_matrix((0, 0), dtype=np.float64)
    mi = MeasurementIndex(
        kind=np.zeros(0, dtype=np.int64),
        object_kind=np.zeros(0, dtype=np.int64),
        object_pos=np.zeros(0, dtype=np.int64),
        branch_side=np.zeros(0, dtype=np.int64),
        meas_id=np.zeros(0, dtype=np.int64),
    )
    return build_ipm_setup(m, network_pu, z, r, mi, layout_base=layout)


def test_ipm_box_created_without_q_bounds():
    """An exist_gen node WITHOUT Q bounds still gets a Qgen box var
    (previously it did not, and the balance-Q pinned its reactive power to zero)."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_p": 50.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
            # generation_q_min/max are not set (0, 0).
        }
    )
    assert setup.layout.pgen_node_pos.size == 1
    assert setup.layout.qgen_node_pos.size == 1  # the key point: the box var exists
    # Default box of ±50 p.u.
    q_idx = setup.layout.pgen_node_pos.size  # the qgen section follows pgen
    assert setup.box_lo[q_idx] == -50.0
    assert setup.box_hi[q_idx] == 50.0


def test_ipm_box_created_for_load_without_bounds():
    setup = _ipm_setup_for({"exist_load": 1, "load_p": 30.0})
    assert setup.layout.pnag_node_pos.size == 1
    assert setup.layout.qnag_node_pos.size == 1


def test_ipm_default_box_no_bus_equiv_prior():
    """The default wide box does NOT get a tight bus-equivalent prior
    (width 100 p.u. >= threshold, but it means "no data", not an equivalent)."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_p": 50.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
        }
    )
    # prior measurements in z come after balance: no prior row
    # may appear (prior_sigma2_normal_pu=0 and the default box is not a bus equivalent).
    n_balance = 2 * 2  # 2 active nodes x (P+Q)
    assert setup.z.shape[0] == n_balance  # z was empty; only balance rows


def test_ipm_real_bus_equiv_still_gets_prior():
    """A real wide box from DATA is still a bus equivalent."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_p": 500.0,
            "generation_p_min": 0.0,
            "generation_p_max": 42500.0,
            "generation_q_min": -1000.0,
            "generation_q_max": 1000.0,
        }
    )
    n_balance = 2 * 2
    # One prior row per pgen (width 425 p.u. > 100) - bus-equiv tight.
    assert setup.z.shape[0] == n_balance + 1


# ---------------------------------------------------------------------------
# Compensators: P exactly [0, 0] with Q specified is data, not "unspecified"
# ---------------------------------------------------------------------------


def test_reactive_only_generation_detection():
    from gridstate.bounds import is_reactive_only_generation

    assert is_reactive_only_generation(0.0, 0.0, -63.0, 0.0)  # controlled shunt reactor
    assert is_reactive_only_generation(0.0, 0.0, -66.0, 66.0)  # STATCOM
    assert not is_reactive_only_generation(0.0, 100.0, -50.0, 50.0)  # machine
    assert not is_reactive_only_generation(0.0, 0.0, 0.0, 0.0)  # nothing specified
    assert not is_reactive_only_generation(0.0, 0.0, 9999.0, -9999.0)  # sentinels


def test_ipm_compensator_gets_no_active_power_box():
    """Controlled shunt reactor (P [0, 0], Q [-63, 0]): no P variable, the Q variable has its own box."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_q_min": -63.0,
            "generation_q_max": 0.0,
        }
    )
    assert setup.layout.pgen_node_pos.size == 0
    assert setup.layout.qgen_node_pos.size == 1
    assert setup.box_lo[0] == -0.63
    assert setup.box_hi[0] == 0.0


_PAIRS = [
    (0.0, 0.0),
    (-SENTINEL_VALUE, SENTINEL_VALUE),
    (-SENTINEL_VALUE, 100.0),
    (10.0, 10.0),
    (10.0, 5.0),
    (-50.0, 120.0),
    (float("-inf"), 5.0),
    (5.0, float("inf")),
    (float("inf"), float("inf")),
    (float("nan"), 5.0),
    (float("nan"), float("nan")),
]


def test_declared_box_matches_inline_checks():
    for lo_raw, hi_raw in _PAIRS:
        lo, hi = resolve_bounds(lo_raw, hi_raw)
        strict = bool(np.isfinite(lo) and np.isfinite(hi) and hi > lo)
        finite = bool(np.isfinite(lo) and np.isfinite(hi))
        assert (declared_box(lo_raw, hi_raw) is not None) is strict
        assert (declared_box(lo_raw, hi_raw, allow_point=True) is not None) is finite
        if strict:
            assert declared_box(lo_raw, hi_raw) == (lo, hi)
        assert bounds_unset(lo_raw, hi_raw) is (not (np.isfinite(lo) or np.isfinite(hi)))


def test_reactive_only_generation_matches_former_rule():
    for q_min, q_max in _PAIRS:
        q_lo, q_hi = resolve_bounds(q_min, q_max)
        former = q_lo != float("-inf") and q_hi != float("inf") and q_hi > q_lo
        assert is_reactive_only_generation(0.0, 0.0, q_min, q_max) is former
