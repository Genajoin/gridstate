"""SE result post-processing: splitting injections into load/generation_*_estimated
+ applying static load characteristics (synthetic unit tests)."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Unit tests of write_node_estimates_from_inj.
# ---------------------------------------------------------------------------


def _toy_model_for_inj_split():
    """Minimal model: 5 nodes with different exist_load/exist_gen patterns.

    Node 1 (slack/gen-only), 2 (load-only), 3 (transit, status=False -
    checks that an inactive node is not written), 4 (both), 5 (transit).
    """
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "GenOnly",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "generation_p": 80.0,
            "generation_q": 30.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
            "generation_q_min": -50.0,
            "generation_q_max": 50.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "name": "LoadOnly",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "load_p": 40.0,
            "load_q": 15.0,
            "load_p_min": 0.0,
            "load_p_max": 80.0,
            "load_q_min": 0.0,
            "load_q_max": 30.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "name": "TransitOff",
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": False,  # inactive - the function must skip it
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 4,
            "name": "Both",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "generation_p": 60.0,
            "generation_q": 20.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
            "generation_q_min": -30.0,
            "generation_q_max": 30.0,
            "load_p": 20.0,
            "load_q": 10.0,
            "load_p_min": 0.0,
            "load_p_max": 80.0,
            "load_q_min": 0.0,
            "load_q_max": 30.0,
            "exist_gen": 1,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PV),
        }
    )
    m.nodes.add(
        {
            "id": 5,
            "name": "Transit",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    return m


def _set_inj(model, node_id: int, p: float, q: float) -> None:
    model.nodes.update(node_id, {"p_inj_calc": p, "q_inj_calc": q})


def test_write_node_estimates_from_inj_transit():
    """A node without exist_load and exist_gen -> load/gen_estimated = 0."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 5, 25.0, 10.0)  # nonzero injection at a transit node
    stats = write_node_estimates_from_inj(m)
    n5 = m.nodes.get_by_id(5)
    assert n5.load_p_estimated == 0.0
    assert n5.load_q_estimated == 0.0
    assert n5.generation_p_estimated == 0.0
    assert n5.generation_q_estimated == 0.0
    assert stats["transit"] == 1  # only node 5 - status=True transit


def test_write_node_estimates_from_inj_gen_only():
    """exist_gen=1, exist_load=0 → generation_p_estimated=p_inj, clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 1, 75.0, 25.0)
    write_node_estimates_from_inj(m)
    n1 = m.nodes.get_by_id(1)
    assert n1.generation_p_estimated == 75.0
    assert n1.generation_q_estimated == 25.0
    assert n1.load_p_estimated == 0.0
    assert n1.load_q_estimated == 0.0


def test_write_node_estimates_from_inj_gen_only_clipped():
    """gen_only with p_inj outside [gen_p_min, gen_p_max] -> clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 1, 150.0, 80.0)  # > gen_p_max=100, > gen_q_max=50
    stats = write_node_estimates_from_inj(m)
    n1 = m.nodes.get_by_id(1)
    assert n1.generation_p_estimated == 100.0  # clipped to gen_p_max
    assert n1.generation_q_estimated == 50.0  # clipped to gen_q_max
    assert stats["clipped"] >= 1


def test_write_node_estimates_from_inj_load_only():
    """exist_load=1, exist_gen=0 → load_p_estimated = -p_inj, clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    # p_inj=-50 (the node consumes 50 MW) -> load_p_est = 50
    _set_inj(m, 2, -50.0, -20.0)
    write_node_estimates_from_inj(m)
    n2 = m.nodes.get_by_id(2)
    assert n2.load_p_estimated == 50.0
    assert n2.load_q_estimated == 20.0
    assert n2.generation_p_estimated == 0.0
    assert n2.generation_q_estimated == 0.0


def test_write_node_estimates_from_inj_load_only_clipped():
    """load-only with -p_inj outside [load_p_min, load_p_max] -> clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 2, -200.0, -100.0)  # -p_inj=200 > load_p_max=80
    stats = write_node_estimates_from_inj(m)
    n2 = m.nodes.get_by_id(2)
    assert n2.load_p_estimated == 80.0  # clipped to load_p_max
    assert n2.load_q_estimated == 30.0
    assert stats["clipped"] >= 1


def test_write_node_estimates_from_inj_both_proportional():
    """exist_load=1 + exist_gen=1: the binding to cur_pg / cur_pn is preserved.

    For node 4: cur_gen_p=60, cur_load_p=20 -> p_inj_meas = 40 MW.
    If p_inj_calc = 50 (flow up by 10 MW): gen covers inj+load,
    i.e. gen_est = 50+20 = 70, load_est stays = 20.
    """
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 4, 50.0, 25.0)  # gen surplus
    write_node_estimates_from_inj(m)
    n4 = m.nodes.get_by_id(4)
    assert n4.generation_p_estimated == 70.0  # 50 + 20
    assert n4.load_p_estimated == 20.0
    # q_inj=25, cur_q_gen=20, cur_q_load=10 → q_inj_meas = 10
    # q_inj_calc=25 > 0 -> gen_q_est = 25 + 10 = 35 -> clip to 30 (gen_q_max)
    assert n4.generation_q_estimated == 30.0
    assert n4.load_q_estimated == 10.0


def test_write_node_estimates_from_inj_both_negative_inj():
    """both with a negative p_inj (load > gen): load_est grows."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 4, -30.0, -10.0)  # load surplus
    write_node_estimates_from_inj(m)
    n4 = m.nodes.get_by_id(4)
    # p_inj_calc<0 -> gen=cur_gen=60, load = cur_gen - p_inj = 60+30 = 90 -> clip to 80
    assert n4.generation_p_estimated == 60.0
    assert n4.load_p_estimated == 80.0  # clipped to load_p_max
    # q_inj<0: gen=cur_gen=20, load = 20+10=30 (=load_q_max, no clip)
    assert n4.generation_q_estimated == 20.0
    assert n4.load_q_estimated == 30.0


def test_write_node_estimates_from_inj_inactive_skipped():
    """Inactive nodes (status=False) are not updated."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 3, 999.0, 999.0)
    write_node_estimates_from_inj(m)
    n3 = m.nodes.get_by_id(3)
    # default values (0.0)
    assert n3.load_p_estimated == 0.0
    assert n3.load_q_estimated == 0.0
    assert n3.generation_p_estimated == 0.0
    assert n3.generation_q_estimated == 0.0


# ---------------------------------------------------------------------------
# Unit tests of apply_load_characteristic.
# ---------------------------------------------------------------------------


def _toy_model_with_sxn():
    """Minimal model with load characteristics to check ``apply_load_characteristic``.

    Canonical ``load_characteristics`` table - 3 rows (id = 0-based position):
        id 0 (load_model_id=0): a0=0.5, a1=0.5, a2=0, b0=1, b1=0, b2=0
        id 1 (load_model_id=1): a0=1.0, a1=0.0, a2=0 (PQ-const)
        id 2 (load_model_id=2): a0=0.0, a1=0.0, a2=1.0 (quadratic)

    Nodes (linked via 0-based ``load_model_id``, -1 = no characteristic):
        10 - exist_load=1, load_model_id=0, V=Vn -> actual P multiplier = 0.5+0.5 = 1.0
        11 - exist_load=1, load_model_id=1, V=0.9*Vn -> PQ-const, actual = load_p
        12 - exist_load=1, load_model_id=2, V=1.1*Vn -> quadratic multiplier
        13 - exist_load=1, load_model_id=-1 (no characteristic) -> skipped_no_sxn
        14 — exist_load=0, load_model_id=1 → skipped_no_load
        15 - exist_load=1, load_model_id=98 (out of range) -> skipped_bad_sxn
        16 - status=False, any fields -> skipped
    """
    from gridstate.constants import NodeType
    from gridstate.contract import SE_INPUT
    from gridstate.working import Working, _ArrayCollection

    m = Working.empty()
    # Canonical load_characteristics: id = 0-based row position.
    lc = np.zeros(3, dtype=SE_INPUT.load_characteristics.input_dtype())
    lc["id"] = np.arange(3, dtype=lc["id"].dtype)
    # id 0 — load_model_id=0: a0=0.5 a1=0.5 a2=0; b: b0=1
    lc[0]["coeff_p_a0"] = 0.5
    lc[0]["coeff_p_a1"] = 0.5
    lc[0]["coeff_p_a2"] = 0.0
    lc[0]["coeff_q_b0"] = 1.0
    lc[0]["coeff_q_b1"] = 0.0
    lc[0]["coeff_q_b2"] = 0.0
    # id 1 — load_model_id=1: PQ-const
    lc[1]["coeff_p_a0"] = 1.0
    lc[1]["coeff_q_b0"] = 1.0
    # id 2 - load_model_id=2: purely quadratic
    lc[2]["coeff_p_a2"] = 1.0
    lc[2]["coeff_q_b2"] = 1.0
    m.load_characteristics = _ArrayCollection(lc)

    for nid, kwargs in [
        (
            10,
            {
                "load_model_id": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,  # V_pu=1.0
                "load_p": 100.0,
                "load_q": 50.0,
                "exist_load": 1,
            },
        ),
        (
            11,
            {
                "load_model_id": 1,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 99.0,  # V_pu=0.9
                "load_p": 80.0,
                "load_q": 40.0,
                "exist_load": 1,
            },
        ),
        (
            12,
            {
                "load_model_id": 2,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 121.0,  # V_pu=1.1
                "load_p": 200.0,
                "load_q": 100.0,
                "exist_load": 1,
            },
        ),
        (
            13,
            {
                "load_model_id": -1,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "load_p": 50.0,
                "load_q": 25.0,
                "exist_load": 1,
            },
        ),
        (
            14,
            {
                "load_model_id": 1,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "exist_load": 0,
            },
        ),
        (
            15,
            {
                "load_model_id": 98,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "load_p": 10.0,
                "load_q": 5.0,
                "exist_load": 1,
            },
        ),
        (
            16,
            {
                "load_model_id": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "load_p": 999.0,
                "load_q": 999.0,
                "exist_load": 1,
                "status": False,
            },
        ),
    ]:
        defaults = {
            "id": nid,
            "name": f"N{nid}",
            "status": True,
            "node_type": int(NodeType.PQ),
        }
        defaults.update(kwargs)
        m.nodes.add(defaults)
    return m


def test_apply_load_characteristic_linear():
    """sxn_id=1 (a0=0.5, a1=0.5) at V_pu=1.0 -> factor=1.0 -> load_p_est = load_p_nom."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    stats = apply_load_characteristic(m)
    n10 = m.nodes.get_by_id(10)
    assert abs(n10.load_p_estimated - 100.0) < 1e-9, n10.load_p_estimated
    # Q: b0=1, b1=b2=0 → factor_q=1.0 → load_q_est = 50.0
    assert abs(n10.load_q_estimated - 50.0) < 1e-9, n10.load_q_estimated
    assert stats["updated"] >= 1


def test_apply_load_characteristic_pq_const():
    """sxn_id=2 (PQ-const): load_*_est = load_*_nom at any V_pu."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    apply_load_characteristic(m)
    n11 = m.nodes.get_by_id(11)
    # V_pu=0.9 has no effect: factor=1.0
    assert abs(n11.load_p_estimated - 80.0) < 1e-9
    assert abs(n11.load_q_estimated - 40.0) < 1e-9


def test_apply_load_characteristic_quadratic():
    """sxn_id=3 (a2=1, b2=1): factor = V_pu²."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    apply_load_characteristic(m)
    n12 = m.nodes.get_by_id(12)
    # V_pu=1.1 → factor = 1.21
    assert abs(n12.load_p_estimated - 200.0 * 1.21) < 1e-9
    assert abs(n12.load_q_estimated - 100.0 * 1.21) < 1e-9


def test_apply_load_characteristic_counters():
    """skipped_no_sxn / skipped_no_load / skipped_bad_sxn must be present."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    stats = apply_load_characteristic(m)
    # 3 nodes updated (10, 11, 12); 13 sxn=0; 14 exist_load=0; 15 bad sxn; 16 status=False
    assert stats["updated"] == 3
    assert stats["skipped_no_sxn"] == 1
    assert stats["skipped_no_load"] == 1
    assert stats["skipped_bad_sxn"] == 1


def test_apply_load_characteristic_no_load_models():
    """If the model has no load_models, the function returns no_load_models=1."""
    from gridstate.constants import NodeType
    from gridstate.post_processing import apply_load_characteristic
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "load_p": 100.0,
            "exist_load": 1,
            "load_model_id": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    stats = apply_load_characteristic(m)
    assert stats["no_load_models"] == 1
    assert stats["updated"] == 0


def test_write_node_estimates_from_inj_stats_counters():
    """Counters per category match the actual breakdown."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 1, 50.0, 20.0)
    _set_inj(m, 2, -30.0, -10.0)
    _set_inj(m, 4, 10.0, 5.0)
    stats = write_node_estimates_from_inj(m)
    # 4 active nodes: 1=gen_only, 2=load_only, 4=both, 5=transit; 3=off
    assert stats["gen_only"] == 1
    assert stats["load_only"] == 1
    assert stats["both"] == 1
    assert stats["transit"] == 1
    assert stats["updated"] == 4


# ---------------------------------------------------------------------------
# NaN semantics of estimated_si/residual for unestimated measurements.
# ---------------------------------------------------------------------------


def test_measurement_estimates_nan_for_unestimated():
    """A measurement outside meas_index -> estimated_si/residual = NaN; an estimated one -> a number.

    After SE, active measurements carry h(x)/residual in original units, while an
    inactive one (status=False, not in the z vector) stays NaN -
    distinguishable from a true zero. Regression for silent zero-fill zeros.
    """
    from gridstate.api import estimate
    from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE
    from tests.test_bad_data import _three_bus_with_clean_measurements

    m, _, _ = _three_bus_with_clean_measurements()
    # A deliberately inactive measurement - will NOT enter meas_index.
    inactive_id = 999_001
    m.measurements.add(
        {
            "id": inactive_id,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": 108.0,
            "variance": 0.01,
            "status": False,
            "quality": 0,
        }
    )

    res = estimate(m, tolerance=1e-10)
    assert res.success

    by_id = {int(me.id): me for me in res.model.measurements}

    # Unestimated (status=False) -> NaN in both OUTPUT fields.
    inactive = by_id[inactive_id]
    assert np.isnan(float(inactive.estimated_si))
    assert np.isnan(float(inactive.residual))

    # At least one estimated measurement carries a finite number.
    estimated = [
        me
        for mid, me in by_id.items()
        if mid != inactive_id and np.isfinite(float(me.estimated_si))
    ]
    assert estimated, "expected estimated measurements with a finite estimated_si"
    for me in estimated:
        assert np.isfinite(float(me.residual))
