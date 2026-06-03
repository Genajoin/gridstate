"""Тесты анализа наблюдаемости (``gridstate.validation.observability``).

Проверки:
    - переопределённая 3-узловая сеть полностью наблюдаема;
    - ``n_meas < n_state`` → ``is_observable == False``;
    - узел без измерений → попадает в ``unobservable_buses``;
    - пустая коллекция измерений → отчёт об отсутствии измерений.
"""

from __future__ import annotations

from gridstate.validation.observability import (
    ObservabilityReport,
    analyze_observability,
)
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_VOLTAGE,
    OBJ_NODE,
)


# ----------------------------------------------------------- helpers
def _build_two_bus_observable():
    """2-узловая сеть, переопределённая системой измерений."""
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
        }
    )
    # 3 измерения на 2-узлов: V на каждом узле + P_inj на узле 2.
    # n_state = 2*2 - 1 = 3.
    m.measurements.add(
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
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": 109.0,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 3,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": -30.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 4,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_Q,
            "value": -10.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    return m


# --------------------------------------------------------- positive cases
def test_two_bus_observable_when_fully_covered() -> None:
    m = _build_two_bus_observable()
    rep = analyze_observability(m)
    assert isinstance(rep, ObservabilityReport)
    assert rep.is_observable
    assert rep.n_state_vars == 3
    assert rep.n_measurements == 4
    assert rep.rank_H == 3
    assert rep.unobservable_buses == []


def test_three_bus_with_full_measurements_is_observable() -> None:
    """3-узловая с переопределённой системой — наблюдаема."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid in (1, 2, 3):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(NodeType.SLACK if nid == 1 else NodeType.PQ),
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
        }
    )
    # V на всех + P/Q-инъекции на 2 и 3
    next_id = 1
    for node_id in (1, 2, 3):
        m.measurements.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": node_id,
                "measurement_type": KIND_VOLTAGE,
                "value": 110.0,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
    for node_id in (2, 3):
        for kind in (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q):
            m.measurements.add(
                {
                    "id": next_id,
                    "object_type": OBJ_NODE,
                    "object_id": node_id,
                    "measurement_type": kind,
                    "value": -30.0 if kind == KIND_POWER_INJECTION_P else -10.0,
                    "variance": 0.5,
                    "status": True,
                    "quality": 0,
                }
            )
            next_id += 1
    rep = analyze_observability(m)
    assert rep.is_observable
    assert rep.n_state_vars == 5  # 2*3 - 1
    assert rep.rank_H == 5


# --------------------------------------------------------- negative cases
def test_too_few_measurements_unobservable() -> None:
    """``m < n_state`` → не наблюдаема."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    m.nodes.add({"id": 2, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.PQ)})
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
        }
    )
    # Только одно V на slack → m=1, n_state=3
    m.measurements.add(
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
    rep = analyze_observability(m)
    assert not rep.is_observable
    assert rep.n_measurements == 1
    assert rep.rank_H <= 1
    # Узел 2 — δ и V не покрыты.
    assert 2 in rep.unobservable_buses


def test_empty_measurements_returns_no_observability() -> None:
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    m.nodes.add({"id": 2, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.PQ)})
    rep = analyze_observability(m)
    assert not rep.is_observable
    assert rep.n_measurements == 0
    assert rep.rank_H == 0
    assert set(rep.unobservable_buses) == {1, 2}
    assert "Нет ни одного" in rep.diagnostics


def test_observable_island_unmeasured_bus_in_diagnostics() -> None:
    """3 узла + V на узлах 1 и 2; узел 3 не покрыт измерением V."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid in (1, 2, 3):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "status": True,
                "node_type": int(NodeType.SLACK if nid == 1 else NodeType.PQ),
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
        }
    )
    # Только V на 1 и 2 → измерения есть только на части сети.
    m.measurements.add(
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
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": 110.0,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    rep = analyze_observability(m)
    assert not rep.is_observable
    # Узел 3 точно неосвещён (нет вообще никакой инъекции/перетока, что зависело
    # бы от его δ или V).
    assert 3 in rep.unobservable_buses
