"""float-free ядро ON_LINE-топологии над контрактными массивами.

``apply_topology_from_xml`` расщеплена на двухслойный шов: АДАПТЕР
резолвит ON_LINE-формулы по snapshot через ``_eval_status_formula`` + читает
``spec.args`` (GENERATOR-guard) → план ``resolved: list[(tag, parent_id, status|None,
eval_skip|None)]`` в порядке ``specs.items()``; ЯДРО ``_apply_topology_on_arrays``
матчит ``parent_tag`` → целевой массив (nodes/branches/generators), пишет
``status``-колонку. Чистый статус-каскад без float → строгий бит-в-бит. Здесь —
корректность ядра на голых массивах; бит-в-бит публичного API — canon transitively +
поведенческий new-vs-HEAD дифф на реальном регионе.
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.telemetry.on_line import _apply_topology_on_arrays


def _arr(table: str, ids: list[int], status: bool = True) -> np.ndarray:
    dt = (
        SE_INPUT.nodes.input_dtype()
        if table == "nodes"
        else SE_INPUT.branches.input_dtype()
        if table == "branches"
        else SE_INPUT.generators.input_dtype()
    )
    arr = np.zeros(len(ids), dtype=dt)
    for i, oid in enumerate(ids):
        arr[i]["id"] = oid
        arr[i]["status"] = status
    return arr


def _nodes(ids, status=True):
    return _arr("nodes", ids, status)


def _branches(ids, status=True):
    return _arr("branches", ids, status)


def _gens(ids, status=True):
    return _arr("generators", ids, status)


def test_topo_core_node_status_write():
    nodes = _nodes([1, 2])
    resolved = [("NODE", 1, False, None), ("NODE", 2, True, None)]
    stats = _apply_topology_on_arrays(nodes, _branches([]), _gens([]), resolved)
    assert bool(nodes[0]["status"]) is False
    assert bool(nodes[1]["status"]) is True
    assert stats["applied_off"] == 1 and stats["applied_on"] == 1
    assert stats["total_specs"] == 2


def test_topo_core_line_status_write():
    branches = _branches([7])
    stats = _apply_topology_on_arrays(
        _nodes([]), branches, _gens([]), [("LINE", 7, False, None)]
    )
    assert bool(branches[0]["status"]) is False
    assert stats["applied_off"] == 1


def test_topo_core_generator_status_write():
    gens = _gens([3])
    stats = _apply_topology_on_arrays(
        _nodes([]), _branches([]), gens, [("GENERATOR", 3, False, None)]
    )
    assert bool(gens[0]["status"]) is False
    assert stats["applied_off"] == 1


def test_topo_core_reactor_tag_skipped_no_object():
    # Тег REACTOR больше не имеет целевого массива → дефолтная ветка → skipped_no_object.
    stats = _apply_topology_on_arrays(
        _nodes([]), _branches([]), _gens([]), [("REACTOR", 9, True, None)]
    )
    assert stats["skipped_no_object"] == 1
    assert stats["applied_on"] == 0


def test_topo_core_skipped_no_object_unknown_id():
    nodes = _nodes([1])
    stats = _apply_topology_on_arrays(
        nodes, _branches([]), _gens([]), [("NODE", 999, True, None)]
    )
    assert stats["skipped_no_object"] == 1
    assert bool(nodes[0]["status"]) is True  # не тронут


def test_topo_core_unknown_tag_skipped_no_object():
    stats = _apply_topology_on_arrays(
        _nodes([1]), _branches([]), _gens([]), [("WIDGET", 1, True, None)]
    )
    assert stats["skipped_no_object"] == 1


def test_topo_core_eval_skip_counters_routed():
    # status=None с разными eval_skip → соответствующий счётчик, без записи.
    nodes = _nodes([1, 2, 3])
    resolved = [
        ("NODE", 1, None, "skipped_no_value"),
        ("NODE", 2, None, "skipped_partial_args"),
        ("NODE", 3, None, "skipped_formula_error"),
    ]
    stats = _apply_topology_on_arrays(nodes, _branches([]), _gens([]), resolved)
    assert stats["skipped_no_value"] == 1
    assert stats["skipped_partial_args"] == 1
    assert stats["skipped_formula_error"] == 1
    assert stats["applied_on"] == 0 and stats["applied_off"] == 0
    assert all(bool(nodes[i]["status"]) for i in range(3))  # никого не тронули


def test_topo_core_no_object_precedes_eval_skip():
    # Несуществующий объект со status=None → skipped_no_object (объект-чек первичен).
    stats = _apply_topology_on_arrays(
        _nodes([1]), _branches([]), _gens([]), [("NODE", 42, None, "skipped_no_value")]
    )
    assert stats["skipped_no_object"] == 1
    assert stats["skipped_no_value"] == 0


def test_topo_core_total_specs_equals_resolved_len():
    resolved = [
        ("NODE", 1, True, None),
        ("LINE", 7, None, "skipped_formula_error"),
        ("GENERATOR", 3, True, None),
    ]
    stats = _apply_topology_on_arrays(
        _nodes([1]), _branches([7]), _gens([3]), resolved
    )
    assert stats["total_specs"] == 3
    assert stats["applied_on"] == 2  # NODE 1 + GENERATOR 3
    assert stats["skipped_formula_error"] == 1


def test_topo_core_mixed_targets_independent():
    nodes = _nodes([1])
    branches = _branches([7])
    gens = _gens([3])
    resolved = [
        ("NODE", 1, False, None),
        ("LINE", 7, True, None),
        ("GENERATOR", 3, False, None),
    ]
    stats = _apply_topology_on_arrays(nodes, branches, gens, resolved)
    assert bool(nodes[0]["status"]) is False
    assert bool(branches[0]["status"]) is True
    assert bool(gens[0]["status"]) is False
    assert stats["applied_on"] == 1 and stats["applied_off"] == 2
