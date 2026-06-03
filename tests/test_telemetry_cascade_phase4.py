"""Ф4.1 (слайс 2): ядра статус-каскада/агрегации/шунтов на контрактных массивах.

Функции ``apply_generator_status_from_node`` / ``aggregate_generators_to_node``
(telemetry/generators.py) и ``apply_reactors_to_node_shunt`` /
``normalize_breaker_reactance`` (telemetry/shunts.py) расщеплены на
``_*_on_arrays``-ядро (PSC-free, мутирует контрактные numpy-массивы) + адаптер.
Эти ядра — НОВАЯ способность Ф4: исполняются на голых ``SE_INPUT``-массивах без
``PowerSystemModel`` и читают только контрактные колонки. Бит-в-бит модели для
адаптеров стережёт существующий telemetry-сьют; здесь — корректность ядер.
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.telemetry.generators import (
    _aggregate_generators_on_arrays,
    _apply_generator_status_on_arrays,
)
from gridstate.telemetry.shunts import (
    _BREAKER_X_SENTINEL_OHM,
    _apply_reactors_on_arrays,
    _normalize_breaker_reactance_on_arrays,
)
from gridstate.units import BASE_MVA


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _gens(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.generators.input_dtype())
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


def _reactors(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.raw_table("reactors").numpy_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


# ---------------------------------------------------------------------------
# apply_generator_status: node off ⇒ gen off
# ---------------------------------------------------------------------------


def test_generator_status_core_cascade_off():
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": False}])
    gens = _gens(
        [
            {"id": 10, "node_id": 1, "status": True},  # узел on → не трогаем
            {"id": 11, "node_id": 2, "status": True},  # узел off → гасим
            {"id": 12, "node_id": 2, "status": False},  # уже off → не считаем
            {"id": 13, "node_id": 99, "status": True},  # узла нет → missing
        ]
    )
    stats = _apply_generator_status_on_arrays(nodes, gens)
    assert stats == {"applied_off": 1, "missing_node": 1}
    assert bool(gens[0]["status"]) is True
    assert bool(gens[1]["status"]) is False
    assert bool(gens[2]["status"]) is False
    assert bool(gens[3]["status"]) is True  # missing-node ген не трогается


# ---------------------------------------------------------------------------
# aggregate_generators: сумма active-генов по узлу, off исключены
# ---------------------------------------------------------------------------


def test_aggregate_generators_core_sums_active_only():
    nodes = _nodes([{"id": 1}, {"id": 2}])
    gens = _gens(
        [
            {
                "id": 10,
                "node_id": 1,
                "status": True,
                "power_output": 10.0,
                "reactive_output": 1.0,
                "power_min": 0.0,
                "power_max": 15.0,
                "reactive_min": -5.0,
                "reactive_max": 5.0,
            },
            {
                "id": 11,
                "node_id": 1,
                "status": True,
                "power_output": 20.0,
                "reactive_output": 2.0,
                "power_min": 0.0,
                "power_max": 25.0,
                "reactive_min": -7.0,
                "reactive_max": 7.0,
            },
            {
                "id": 12,
                "node_id": 1,
                "status": False,
                "power_output": 100.0,
                "reactive_output": 50.0,
                "power_max": 999.0,
            },  # off → исключён
            {"id": 13, "node_id": 99, "status": True, "power_output": 5.0},  # узла нет
        ]
    )
    stats = _aggregate_generators_on_arrays(nodes, gens)
    assert stats == {"updated_nodes": 1, "active_gens": 2, "off_gens": 1, "missing_node": 1}
    assert float(nodes[0]["generation_p"]) == 30.0  # 10+20, off-100 исключён
    assert float(nodes[0]["generation_q"]) == 3.0
    assert float(nodes[0]["generation_p_max"]) == 40.0  # 15+25
    assert float(nodes[0]["generation_q_min"]) == -12.0
    assert float(nodes[1]["generation_p"]) == 0.0  # узел 2 без генов


# ---------------------------------------------------------------------------
# apply_reactors: sign·B/G (мкСм→См) в shunt узла; off-узлы/реакторы пропуск
# ---------------------------------------------------------------------------


def test_reactors_core_adds_to_node_shunt():
    nodes = _nodes(
        [
            {"id": 1, "status": True, "shunt_b": 0.0, "shunt_g": 0.0},
            {"id": 2, "status": False, "shunt_b": 0.0, "shunt_g": 0.0},  # off-узел
        ]
    )
    reac = _reactors(
        [
            {"node_id": 1, "status": True, "susceptance": 272109.0, "conductance": 1000.0},
            {"node_id": 1, "status": False, "susceptance": 999999.0},  # off-реактор
            {"node_id": 2, "status": True, "susceptance": 500000.0},  # узел off
            {"node_id": 0, "status": True, "susceptance": 500000.0},  # nid==0
        ]
    )
    stats = _apply_reactors_on_arrays(nodes, reac, sign=1)
    assert stats["applied"] == 1
    assert float(nodes[0]["shunt_b"]) == 272109.0 * 1e-6
    assert float(nodes[0]["shunt_g"]) == 1000.0 * 1e-6
    assert float(nodes[1]["shunt_b"]) == 0.0


def test_reactors_core_sign_flip():
    nodes = _nodes([{"id": 1, "status": True, "shunt_b": 0.0, "shunt_g": 0.0}])
    reac = _reactors([{"node_id": 1, "status": True, "susceptance": 272109.0}])
    stats = _apply_reactors_on_arrays(nodes, reac, sign=-1)
    assert stats["applied"] == 1
    assert float(nodes[0]["shunt_b"]) == -272109.0 * 1e-6


def test_reactors_core_empty_noop():
    nodes = _nodes([{"id": 1, "status": True}])
    assert _apply_reactors_on_arrays(nodes, None) == {
        "applied": 0,
        "sum_b_added_S": 0.0,
        "sum_g_added_S": 0.0,
    }
    assert _apply_reactors_on_arrays(nodes, _reactors([])) == {
        "applied": 0,
        "sum_b_added_S": 0.0,
        "sum_g_added_S": 0.0,
    }


# ---------------------------------------------------------------------------
# normalize_breaker_reactance: сентинел (R=0, X=1.0 Ом) → X_pu=eps_pu
# ---------------------------------------------------------------------------


def test_normalize_breaker_core_volt_aware():
    nodes = _nodes([{"id": 1, "voltage_nominal": 10.5}, {"id": 2, "voltage_nominal": 0.0}])
    branches = _branches(
        [
            # сентинел на 10.5 кВ → нормализуем
            {
                "id": 1,
                "from_node": 1,
                "to_node": 2,
                "resistance": 0.0,
                "reactance": _BREAKER_X_SENTINEL_OHM,
            },
            # реальная ветвь (R>0) → не трогаем
            {"id": 2, "from_node": 1, "to_node": 2, "resistance": 6.05, "reactance": 30.25},
            # сентинел, но Vn недоступен (узел 2 vn=0, fallback на to=2 тоже 0) → пропуск
            {
                "id": 3,
                "from_node": 2,
                "to_node": 2,
                "resistance": 0.0,
                "reactance": _BREAKER_X_SENTINEL_OHM,
            },
        ]
    )
    eps = 1e-3
    n = _normalize_breaker_reactance_on_arrays(branches, nodes, eps_pu=eps)
    assert n == 1
    assert float(branches[0]["reactance"]) == eps * (10.5 * 10.5 / BASE_MVA)
    assert float(branches[1]["reactance"]) == 30.25  # реальная не тронута
    assert float(branches[2]["reactance"]) == _BREAKER_X_SENTINEL_OHM  # Vn=0 пропуск


def test_normalize_breaker_core_fallback_to_node():
    # from-узел vn=0, to-узел vn=110 → берём vn=110 из to.
    nodes = _nodes([{"id": 1, "voltage_nominal": 0.0}, {"id": 2, "voltage_nominal": 110.0}])
    branches = _branches(
        [
            {
                "id": 1,
                "from_node": 1,
                "to_node": 2,
                "resistance": 0.0,
                "reactance": _BREAKER_X_SENTINEL_OHM,
            },
        ]
    )
    n = _normalize_breaker_reactance_on_arrays(branches, nodes, eps_pu=1e-3)
    assert n == 1
    assert float(branches[0]["reactance"]) == 1e-3 * (110.0 * 110.0 / BASE_MVA)
