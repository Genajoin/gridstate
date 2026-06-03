"""Тесты жёсткого V-якоря ненаблюдаемых узлов (Цех-3 #2p).

``add_pseudo_measurements(unobservable_v_sigma_frac=…)`` якорит pseudo-V
ненаблюдаемого узла жёстко (вместо 5 %), НО только когда прайор —
нетривиальная рабочая точка (``|vm-vn| ≥ min_vm_deviation·vn``) и узел не
наблюдаем (нет real-V-соседа / инцидентного flow по гейтам). Default
(``None``) — поведение не меняется (no-op). См. memory cex3-pseudov-vm-anchor.
"""

from __future__ import annotations

import pytest

from gridstate.preprocessing import add_pseudo_measurements
from gridstate.z_vector import (
    KIND_POWER_P,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


def _build_model():
    """5 узлов: разные комбинации vm/vn, real-V, соседей и flow.

    * N1 vn=500 vm=520 (+4 %), ненаблюдаем (нет real-V/соседа/flow).
    * N2 vn=110 vm=110 (==vn), ненаблюдаем — но прайор тривиален.
    * N3 vn=220 vm=230, ЕСТЬ real-V (skip целиком).
    * N4 vn=110 vm=115, сосед N3 (real-V neighbor).
    * N5 vn=35  vm=37,  есть инцидентный real branch-P (flow).

    Ветви (line, branch_type=0): N1-N2, N3-N4, N1-N5. real-flow на N1-N5.
    """
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    spec = [
        (1, 500.0, 520.0),
        (2, 110.0, 110.0),
        (3, 220.0, 230.0),
        (4, 110.0, 115.0),
        (5, 35.0, 37.0),
    ]
    for nid, vn, vm in spec:
        m.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": vn,
                "voltage_magnitude": vm,
                "status": True,
                "node_type": int(NodeType.PQ),
                "shunt_b": 0.0,
                "shunt_g": 0.0,
            }
        )
    for bid, f, t in [(10, 1, 2), (11, 3, 4), (12, 1, 5)]:
        m.branches.add(
            {
                "id": bid,
                "name": f"B{bid}",
                "from_node": f,
                "to_node": t,
                "status": True,
                "branch_type": 0,
                "tap_ratio": 1.0,
                "resistance": 0.01,
                "reactance": 0.1,
                "susceptance": 0.0,
            }
        )
    # real V-замер на N3
    m.measurements.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 3,
            "measurement_type": KIND_VOLTAGE,
            "value": 230.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
            "is_pseudo": False,
        }
    )
    # real branch-P на ветви N1-N5 (id=12)
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_BRANCH,
            "object_id": 12,
            "measurement_type": KIND_POWER_P,
            "value": 10.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
            "is_pseudo": False,
        }
    )
    return m


def _pseudo_v_sigma_frac(model, nid: int, vn: float):
    """σ-фракция pseudo-V узла (None если pseudo-V не добавлен)."""
    for me in model.measurements:
        if (
            int(me.object_type) == OBJ_NODE
            and int(me.measurement_type) == KIND_VOLTAGE
            and bool(me.is_pseudo)
            and int(me.object_id) == nid
        ):
            return (float(me.variance) ** 0.5) / vn
    return None


def test_default_is_noop():
    """Без kwarg все ненаблюдаемые узлы получают loose 5 % (нет tight)."""
    m = _build_model()
    add_pseudo_measurements(m, add_zero_injections=False)
    # N1/N2/N4/N5 без real-V → pseudo-V 5 %; N3 (real-V) — без pseudo.
    assert _pseudo_v_sigma_frac(m, 1, 500.0) == pytest.approx(0.05)
    assert _pseudo_v_sigma_frac(m, 2, 110.0) == pytest.approx(0.05)
    assert _pseudo_v_sigma_frac(m, 5, 35.0) == pytest.approx(0.05)
    assert _pseudo_v_sigma_frac(m, 3, 220.0) is None


def test_unobservable_node_tightened():
    """N1 (vm≠vn, нет real-V/соседа/flow) → жёсткий σ=2 %."""
    m = _build_model()
    add_pseudo_measurements(
        m,
        add_zero_injections=False,
        unobservable_v_sigma_frac=0.02,
        unobservable_v_min_vm_deviation=0.01,
    )
    assert _pseudo_v_sigma_frac(m, 1, 500.0) == pytest.approx(0.02)


def test_trivial_vm_not_tightened():
    """N2 (vm==vn) — прайор тривиален → НЕ якорим (остаётся 5 %)."""
    m = _build_model()
    add_pseudo_measurements(
        m,
        add_zero_injections=False,
        unobservable_v_sigma_frac=0.02,
        unobservable_v_min_vm_deviation=0.01,
    )
    assert _pseudo_v_sigma_frac(m, 2, 110.0) == pytest.approx(0.05)


def test_real_v_neighbor_excluded():
    """N4 (сосед N3 с real-V) → исключён из tighten (остаётся 5 %)."""
    m = _build_model()
    add_pseudo_measurements(
        m,
        add_zero_injections=False,
        unobservable_v_sigma_frac=0.02,
        unobservable_v_min_vm_deviation=0.01,
    )
    assert _pseudo_v_sigma_frac(m, 4, 110.0) == pytest.approx(0.05)


def test_incident_flow_default_tightened():
    """N5 (инцидентный real-flow): default exclude_incident_flow=False → tight."""
    m = _build_model()
    add_pseudo_measurements(
        m,
        add_zero_injections=False,
        unobservable_v_sigma_frac=0.02,
        unobservable_v_min_vm_deviation=0.01,
    )
    assert _pseudo_v_sigma_frac(m, 5, 35.0) == pytest.approx(0.02)


def test_incident_flow_excluded_when_flag_on():
    """N5 с exclude_incident_flow=True → НЕ якорим (остаётся 5 %)."""
    m = _build_model()
    add_pseudo_measurements(
        m,
        add_zero_injections=False,
        unobservable_v_sigma_frac=0.02,
        unobservable_v_min_vm_deviation=0.01,
        unobservable_v_exclude_incident_flow=True,
    )
    assert _pseudo_v_sigma_frac(m, 5, 35.0) == pytest.approx(0.05)


def test_min_vm_deviation_threshold():
    """min_vm_deviation=0 якорит даже тривиальный vm (N2 vm==vn → tight)."""
    m = _build_model()
    add_pseudo_measurements(
        m,
        add_zero_injections=False,
        unobservable_v_sigma_frac=0.02,
        unobservable_v_min_vm_deviation=0.0,
    )
    # N2 vm==vn, |vm-vn|=0 >= 0 → теперь якорится
    assert _pseudo_v_sigma_frac(m, 2, 110.0) == pytest.approx(0.02)
    # N1 (vm≠vn) тоже tight
    assert _pseudo_v_sigma_frac(m, 1, 500.0) == pytest.approx(0.02)
