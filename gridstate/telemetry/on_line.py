"""Topology: apply the derived in-service statuses to the model.

Contract core ``_apply_topology_on_arrays`` plus a thin wrapper ``apply_topology_resolved``
(applies a ready numeric status plan to the ``status`` of model objects).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gridstate.contract.derived import TopologyItem
from gridstate.utils import id_to_pos_map


# Status plan item: ``(object_kind, object_id, status|None, skip_reason|None)``
# (vocabulary: gridstate.contract.derived_tables).
ResolvedItem = TopologyItem

# skip_reason of an item without a status → statistics counter.
_SKIP_COUNTERS = {
    "no_value": "skipped_no_value",
    "partial_inputs": "skipped_partial_args",
    "evaluation_error": "skipped_formula_error",
}


def apply_topology_resolved(model: Any, resolved: Sequence[ResolvedItem]) -> dict[str, int]:
    """Apply a ready status plan ``resolved`` to the model ``status``.

    Pure application (no formulas): snapshot of the contract arrays → core
    :func:`_apply_topology_on_arrays` → write-back. ``resolved`` is a ready status plan
    ``(object_kind, object_id, status|None, skip_reason|None)``.
    """
    arr_nodes = model.nodes.to_numpy().copy()
    arr_branches = model.branches.to_numpy().copy()
    arr_gens = model.generators.to_numpy().copy()
    stats = _apply_topology_on_arrays(arr_nodes, arr_branches, arr_gens, resolved)
    model.nodes.update_from_array(arr_nodes)
    model.branches.update_from_array(arr_branches)
    model.generators.update_from_array(arr_gens)
    return stats


def _apply_topology_on_arrays(
    arr_nodes: np.ndarray,
    arr_branches: np.ndarray,
    arr_gens: np.ndarray,
    resolved: Sequence[ResolvedItem],
) -> dict[str, int]:
    """CORE: apply derived statuses over the contract arrays.

    Pure status cascade: takes a ready ``resolved`` plan (item
    ``(object_kind, object_id, status|None, skip_reason|None)``), matches
    ``object_kind`` (``node``/``branch``/``generator``) → target array, writes the
    ``status`` column. Items of other object kinds count as ``skipped_no_object``.
    Mutates the arrays in place. No external dependencies, no float arithmetic.
    """
    by_id_nodes = id_to_pos_map(arr_nodes["id"])
    by_id_branches = id_to_pos_map(arr_branches["id"])
    by_id_gens = id_to_pos_map(arr_gens["id"])

    stats = {
        "applied_on": 0,
        "applied_off": 0,
        "skipped_no_value": 0,
        "skipped_partial_args": 0,
        "skipped_no_object": 0,
        "skipped_formula_error": 0,
        "total_specs": len(resolved),
    }

    for kind, object_id, status, skip_reason in resolved:
        if kind == "node":
            idx = by_id_nodes.get(object_id)
            target = arr_nodes
        elif kind == "branch":
            idx = by_id_branches.get(object_id)
            target = arr_branches
        elif kind == "generator":
            idx = by_id_gens.get(object_id)
            target = arr_gens
        else:
            idx = None
            target = None
        if idx is None or target is None:
            stats["skipped_no_object"] += 1
            continue
        if status is None:
            # Contract: when ``status is None`` the producer always sets the skip reason.
            assert skip_reason is not None
            stats[_SKIP_COUNTERS[skip_reason]] += 1
            continue

        target[idx]["status"] = status
        if status:
            stats["applied_on"] += 1
        else:
            stats["applied_off"] += 1

    return stats
