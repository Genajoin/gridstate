"""Float-free kernel that applies a resolved on-line status plan to contract arrays.

The producer of the input resolves the element statuses into a plan
``resolved: list[(object_kind, object_id, status|None, skip_reason|None)]``; the kernel
``_apply_topology_on_arrays`` matches ``object_kind`` to the target array
(nodes/branches/generators) and writes the ``status`` column. A pure status
cascade without floats, so results are compared exactly. These tests check the
kernel on bare arrays.
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
    resolved = [("node", 1, False, None), ("node", 2, True, None)]
    stats = _apply_topology_on_arrays(nodes, _branches([]), _gens([]), resolved)
    assert bool(nodes[0]["status"]) is False
    assert bool(nodes[1]["status"]) is True
    assert stats["applied_off"] == 1 and stats["applied_on"] == 1
    assert stats["total_specs"] == 2


def test_topo_core_line_status_write():
    branches = _branches([7])
    stats = _apply_topology_on_arrays(_nodes([]), branches, _gens([]), [("branch", 7, False, None)])
    assert bool(branches[0]["status"]) is False
    assert stats["applied_off"] == 1


def test_topo_core_generator_status_write():
    gens = _gens([3])
    stats = _apply_topology_on_arrays(
        _nodes([]), _branches([]), gens, [("generator", 3, False, None)]
    )
    assert bool(gens[0]["status"]) is False
    assert stats["applied_off"] == 1


def test_topo_core_reactor_tag_skipped_no_object():
    # An object kind without a target array -> skipped_no_object.
    stats = _apply_topology_on_arrays(
        _nodes([]), _branches([]), _gens([]), [("shunt", 9, True, None)]
    )
    assert stats["skipped_no_object"] == 1
    assert stats["applied_on"] == 0


def test_topo_core_skipped_no_object_unknown_id():
    nodes = _nodes([1])
    stats = _apply_topology_on_arrays(nodes, _branches([]), _gens([]), [("node", 999, True, None)])
    assert stats["skipped_no_object"] == 1
    assert bool(nodes[0]["status"]) is True  # untouched


def test_topo_core_unknown_tag_skipped_no_object():
    stats = _apply_topology_on_arrays(
        _nodes([1]), _branches([]), _gens([]), [("WIDGET", 1, True, None)]
    )
    assert stats["skipped_no_object"] == 1


def test_topo_core_skip_reason_counters_routed():
    # status=None with different skip reasons -> the matching counter, no write.
    nodes = _nodes([1, 2, 3])
    resolved = [
        ("node", 1, None, "no_value"),
        ("node", 2, None, "partial_inputs"),
        ("node", 3, None, "evaluation_error"),
    ]
    stats = _apply_topology_on_arrays(nodes, _branches([]), _gens([]), resolved)
    assert stats["skipped_no_value"] == 1
    assert stats["skipped_partial_args"] == 1
    assert stats["skipped_formula_error"] == 1
    assert stats["applied_on"] == 0 and stats["applied_off"] == 0
    assert all(bool(nodes[i]["status"]) for i in range(3))  # nobody touched


def test_topo_core_no_object_precedes_skip_reason():
    # Non-existent object with status=None -> skipped_no_object (the object check comes first).
    stats = _apply_topology_on_arrays(
        _nodes([1]), _branches([]), _gens([]), [("node", 42, None, "no_value")]
    )
    assert stats["skipped_no_object"] == 1
    assert stats["skipped_no_value"] == 0


def test_topo_core_total_specs_equals_resolved_len():
    resolved = [
        ("node", 1, True, None),
        ("branch", 7, None, "evaluation_error"),
        ("generator", 3, True, None),
    ]
    stats = _apply_topology_on_arrays(_nodes([1]), _branches([7]), _gens([3]), resolved)
    assert stats["total_specs"] == 3
    assert stats["applied_on"] == 2  # NODE 1 + GENERATOR 3
    assert stats["skipped_formula_error"] == 1


def test_topo_core_mixed_targets_independent():
    nodes = _nodes([1])
    branches = _branches([7])
    gens = _gens([3])
    resolved = [
        ("node", 1, False, None),
        ("branch", 7, True, None),
        ("generator", 3, False, None),
    ]
    stats = _apply_topology_on_arrays(nodes, branches, gens, resolved)
    assert bool(nodes[0]["status"]) is False
    assert bool(branches[0]["status"]) is True
    assert bool(gens[0]["status"]) is False
    assert stats["applied_on"] == 1 and stats["applied_off"] == 2
