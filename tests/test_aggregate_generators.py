"""Unit-тесты ``aggregate_generators_to_node``.

Проверяет агрегацию параметров генераторов в node-таблицу с учётом
статуса и нескольких генераторов на узле.
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import aggregate_generators_to_node


@pytest.fixture
def model_two_nodes_multi_gen():
    """Модель из 2 узлов: на узле 1 три генератора (mix статусов),
    на узле 2 один off-генератор."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    # Узел 1: G1 ON, G2 ON, G3 OFF
    m.generators.add(
        {
            "id": 11,
            "node_id": 1,
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 10.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": True,
        }
    )
    m.generators.add(
        {
            "id": 12,
            "node_id": 1,
            "power_output": 30.0,
            "reactive_output": 10.0,
            "power_min": 5.0,
            "power_max": 60.0,
            "reactive_min": -20.0,
            "reactive_max": 40.0,
            "status": True,
        }
    )
    m.generators.add(
        {
            "id": 13,
            "node_id": 1,
            "power_output": 999.0,
            "reactive_output": 999.0,  # игнор
            "power_min": 999.0,
            "power_max": 999.0,
            "reactive_min": 999.0,
            "reactive_max": 999.0,
            "status": False,  # OFF — должен быть исключён
        }
    )
    # Узел 2: только OFF
    m.generators.add(
        {
            "id": 21,
            "node_id": 2,
            "power_output": 200.0,
            "reactive_output": 100.0,
            "power_min": 50.0,
            "power_max": 300.0,
            "reactive_min": -100.0,
            "reactive_max": 150.0,
            "status": False,  # OFF
        }
    )
    return m


def test_aggregates_only_active_generators(model_two_nodes_multi_gen) -> None:
    """Σ по active генераторам, off-генераторы исключены."""
    m = model_two_nodes_multi_gen
    stats = aggregate_generators_to_node(m)

    assert stats["active_gens"] == 2
    assert stats["off_gens"] == 2
    assert stats["updated_nodes"] == 1

    n1 = m.nodes.get_by_id(1)
    # G1 + G2 (G3 off исключён)
    assert n1.generation_p == pytest.approx(50.0 + 30.0)
    assert n1.generation_q == pytest.approx(20.0 + 10.0)
    assert n1.generation_p_min == pytest.approx(10.0 + 5.0)
    assert n1.generation_p_max == pytest.approx(100.0 + 60.0)
    assert n1.generation_q_min == pytest.approx(-50.0 + (-20.0))
    assert n1.generation_q_max == pytest.approx(80.0 + 40.0)


def test_node_with_only_off_generators_untouched(model_two_nodes_multi_gen) -> None:
    """Узлы где все генераторы off — не обновляются.

    Узел 2 имеет только OFF-ген: aggregate не должна трогать поля
    ``generation_*`` — что было в NODE_DTYPE (после ``nodes.add()``)
    остаётся как есть.
    """
    m = model_two_nodes_multi_gen
    n2_before = m.nodes.get_by_id(2)
    p_before = n2_before.generation_p
    q_before = n2_before.generation_q
    p_max_before = n2_before.generation_p_max
    q_max_before = n2_before.generation_q_max

    aggregate_generators_to_node(m)

    n2 = m.nodes.get_by_id(2)
    assert n2.generation_p == p_before
    assert n2.generation_q == q_before
    assert n2.generation_p_max == p_max_before
    assert n2.generation_q_max == q_max_before


def test_idempotent(model_two_nodes_multi_gen) -> None:
    """Повторный вызов даёт тот же результат."""
    m = model_two_nodes_multi_gen
    aggregate_generators_to_node(m)
    n1_after_first = (
        m.nodes.get_by_id(1).generation_p,
        m.nodes.get_by_id(1).generation_p_max,
        m.nodes.get_by_id(1).generation_q_max,
    )
    aggregate_generators_to_node(m)
    n1_after_second = (
        m.nodes.get_by_id(1).generation_p,
        m.nodes.get_by_id(1).generation_p_max,
        m.nodes.get_by_id(1).generation_q_max,
    )
    assert n1_after_first == n1_after_second


def test_empty_generators() -> None:
    """Модель без генераторов: stats нулевые, узел не тронут."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    stats = aggregate_generators_to_node(m)
    assert stats["active_gens"] == 0
    assert stats["off_gens"] == 0
    assert stats["updated_nodes"] == 0


def test_missing_node_in_generators() -> None:
    """Генератор с node_id, которого нет в model.nodes — учтён в stats."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.generators.add(
        {
            "id": 99,
            "node_id": 999,  # сирота
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 50.0,
            "status": True,
        }
    )
    stats = aggregate_generators_to_node(m)
    assert stats["missing_node"] == 1
    assert stats["updated_nodes"] == 0
