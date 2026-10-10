"""Generators: status cascade from the node + aggregation of multiple generators into node generation.

Split out of the former monolithic telemetry module.

**Decomposition:** each function is split into an ``_*_on_arrays`` **core over
contract numpy arrays** (mutates the passed arrays in place, returns stats) plus
a thin wrapper (``to_numpy().copy()`` -> core -> ``update_from_array``). The core
does not depend on how results are written back: the wrapper only takes the
array, calls the core and writes the result back.
"""

from __future__ import annotations

from typing import Any

from gridstate._arrays import id_value_map
from gridstate.bounds import SENTINEL_VALUE, is_sentinel
from gridstate.utils import id_to_pos_map


def _apply_generator_status_on_arrays(nodes_arr: Any, gens_arr: Any) -> dict[str, int]:
    """Cascade ``node off ⇒ gen off`` over contract arrays (mutates ``gens_arr``).

    Reads ``node.{id,status}`` and ``gen.{node_id,status}``; sets
    ``gens_arr[i]["status"]=False`` for active generators on off nodes.
    Returns ``{"applied_off": N, "missing_node": N}``.
    """
    node_status = id_value_map(nodes_arr, "status", bool)

    stats = {"applied_off": 0, "missing_node": 0}
    for i in range(len(gens_arr)):
        nid = int(gens_arr[i]["node_id"])
        node_active = node_status.get(nid)
        if node_active is None:
            stats["missing_node"] += 1
            continue
        if not node_active and bool(gens_arr[i]["status"]):
            gens_arr[i]["status"] = False
            stats["applied_off"] += 1
    return stats


def apply_generator_status_from_node(model: Any) -> dict[str, int]:
    """Cascade disconnection: if a node is off, all generators on it are off.

    This is **cascade-off only**: when ``node.status=False`` all generators
    with that ``node_id`` are forced to ``status=False``. If the node is on,
    the generator status is **left untouched**: a single node may carry
    several generators with different statuses (one in service, another on
    reserve), which is normal.

    Enforces the physical invariant ``(node off ⇒ gen off)``, which the node
    cascade (``disable_isolated_nodes`` etc.) did not propagate to the
    generators by itself.

    Must be called **after** all node cascade functions.

    Returns:
        ``{"applied_off": N, "missing_node": N}``.
    """
    nodes_arr = model.nodes.to_numpy()
    arr = model.generators.to_numpy().copy()
    stats = _apply_generator_status_on_arrays(nodes_arr, arr)
    model.generators.update_from_array(arr)
    return stats


def _aggregate_generators_on_arrays(nodes_arr: Any, gens_arr: Any) -> dict[str, int]:
    """Sum of the parameters of a node's active generators -> gen fields of ``nodes_arr`` (mutates).

    Reads ``gen.{node_id,status,power_output,reactive_output,power_min,power_max,
    reactive_min,reactive_max}`` and ``node.id``; writes ``node.{generation_p,
    generation_q,generation_p_min,generation_p_max,generation_q_min,generation_q_max}``
    and raises ``node.exist_gen``.

    ``exist_gen`` is mandatory: a node with a running machine has generation by
    definition, and the flag is the only gate of the ``pgen``/``qgen`` box
    variable in the IPM (``preprocessing/ipm_setup._collect_box_sections``). The
    schema may declare generation as a separate object and not repeat it on the
    node: a unit's output bus arrives with a zero ``exist_gen`` and a narrow load
    box (auxiliary consumption), while the machine's capability sits only in the
    generator catalog record. Without raising the flag the node bounds were
    filled correctly but no variable was created for them, and the measured
    output ran into the load box: a node with a 1080 MW bound and 1070 MW output
    handed zero generation to the estimate, and the injection remainder that the
    dispatch could not place piled up in ``sum_unclosed_p_mw``. On a grid where
    most plants are declared that way, this understated the total generation of
    the regime threefold.

    Limits: only **valid** pairs are summed; if at least one active generator of
    the node carries a sentinel (|limit| >= ``SENTINEL_ABS``, "no data"), the
    node pair is marked with the sentinel ±9999 -- the node range is unknown.
    Previously sentinels were summed as numbers: a real range [-50, 100] plus a
    sentinel [-9999, 9999] gave garbage [-10049, 10099] -- the node either lost
    its box variable in the IPM or was falsely classified as a BUS equivalent
    with a tight prior toward zero.

    Returns ``{"updated_nodes":N,"active_gens":N,"off_gens":N,
    "missing_node":N,"sentinel_p_nodes":N,"sentinel_q_nodes":N}``.
    """
    node_pos = id_to_pos_map(nodes_arr["id"])

    stats = {
        "updated_nodes": 0,
        "active_gens": 0,
        "off_gens": 0,
        "missing_node": 0,
        "sentinel_p_nodes": 0,
        "sentinel_q_nodes": 0,
        "exist_gen_raised": 0,
        "closed_nodes": 0,
    }

    # First zero the gen fields on all nodes so that a repeated call gives
    # a stable result (off generators stop being counted).
    touched: set[int] = set()
    aggregates: dict[int, dict[str, float]] = {}
    # Nodes where at least one generator has no valid P/Q limit pair.
    p_unknown: set[int] = set()
    q_unknown: set[int] = set()

    # Nodes with generators in the catalog -- to close the generation of those
    # whose machines are all off (see below).
    with_gens: set[int] = set()
    for i in range(len(gens_arr)):
        nid = int(gens_arr[i]["node_id"])
        if nid not in node_pos:
            stats["missing_node"] += 1
            continue
        with_gens.add(nid)
        if not bool(gens_arr[i]["status"]):
            stats["off_gens"] += 1
            continue
        stats["active_gens"] += 1
        agg = aggregates.setdefault(
            nid,
            {
                "p": 0.0,
                "q": 0.0,
                "p_min": 0.0,
                "p_max": 0.0,
                "q_min": 0.0,
                "q_max": 0.0,
            },
        )
        agg["p"] += float(gens_arr[i]["power_output"])
        agg["q"] += float(gens_arr[i]["reactive_output"])

        p_min = float(gens_arr[i]["power_min"])
        p_max = float(gens_arr[i]["power_max"])
        if is_sentinel(p_min) or is_sentinel(p_max):
            p_unknown.add(nid)
        else:
            agg["p_min"] += p_min
            agg["p_max"] += p_max

        q_min = float(gens_arr[i]["reactive_min"])
        q_max = float(gens_arr[i]["reactive_max"])
        if is_sentinel(q_min) or is_sentinel(q_max):
            q_unknown.add(nid)
        else:
            agg["q_min"] += q_min
            agg["q_max"] += q_max

    for nid, agg in aggregates.items():
        i = node_pos[nid]
        nodes_arr[i]["generation_p"] = agg["p"]
        nodes_arr[i]["generation_q"] = agg["q"]
        if nid in p_unknown:
            nodes_arr[i]["generation_p_min"] = -SENTINEL_VALUE
            nodes_arr[i]["generation_p_max"] = SENTINEL_VALUE
        else:
            nodes_arr[i]["generation_p_min"] = agg["p_min"]
            nodes_arr[i]["generation_p_max"] = agg["p_max"]
        if nid in q_unknown:
            nodes_arr[i]["generation_q_min"] = -SENTINEL_VALUE
            nodes_arr[i]["generation_q_max"] = SENTINEL_VALUE
        else:
            nodes_arr[i]["generation_q_min"] = agg["q_min"]
            nodes_arr[i]["generation_q_max"] = agg["q_max"]
        if not bool(nodes_arr[i]["exist_gen"]):
            nodes_arr[i]["exist_gen"] = 1
            stats["exist_gen_raised"] += 1
        touched.add(nid)

    # Generation of a node with a generator catalog = the sum of its machines.
    # If all are off (by a status formula or by the cascade from the node), the
    # node has no generation: without this it kept the box and exist_gen from the
    # schema, and the estimate produced generation on it for machines that are
    # not running.
    for nid in with_gens - set(aggregates):
        i = node_pos[nid]
        for col in (
            "generation_p",
            "generation_q",
            "generation_p_min",
            "generation_p_max",
            "generation_q_min",
            "generation_q_max",
        ):
            nodes_arr[i][col] = 0.0
        if bool(nodes_arr[i]["exist_gen"]):
            nodes_arr[i]["exist_gen"] = 0
            stats["closed_nodes"] += 1

    stats["updated_nodes"] = len(touched)
    stats["sentinel_p_nodes"] = len(p_unknown)
    stats["sentinel_q_nodes"] = len(q_unknown)
    return stats


def aggregate_generators_to_node(model: Any) -> dict[str, int]:
    """Sum of the parameters of a node's active generators -> ``model.nodes``.

    For each node sums over the **active** (``status=True``) generators
    with that ``node_id``:

    * ``power_output``    → ``node.generation_p``;
    * ``reactive_output`` → ``node.generation_q``;
    * ``power_min``       → ``node.generation_p_min``;
    * ``power_max``       → ``node.generation_p_max``;
    * ``reactive_min``    → ``node.generation_q_min``;
    * ``reactive_max``    → ``node.generation_q_max``;

    and raises ``node.exist_gen`` -- a node with an active machine has
    generation, and without the flag the IPM creates no box variable for it (see
    ``_aggregate_generators_on_arrays``). The flag is only ever raised: a node
    whose machines are all off does not get here and keeps its previous values.

    Off generators are excluded from all sums (their Q limits are not merged into
    the node's Q range). Nodes that have no generators in the catalog at all keep
    their previous generation_* values (generation is declared on the node
    itself). Sentinel limits (±9999, "no data") are not summed: a node with at
    least one such generator gets a sentinel pair -- "range unknown" (see
    ``_aggregate_generators_on_arrays``).

    Idempotent: a repeated call gives the same result, because the sums are
    overwritten completely (not accumulated).

    Must be called **after** the ON_LINE topology has been applied and after
    ``apply_generator_status_from_node`` -- once the final generator statuses are
    set.

    Returns:
        ``{"updated_nodes": N, "active_gens": N, "off_gens": N,
        "missing_node": N, "exist_gen_raised": N}``.
    """
    nodes_arr = model.nodes.to_numpy().copy()
    gens = model.generators.to_numpy()
    stats = _aggregate_generators_on_arrays(nodes_arr, gens)
    model.nodes.update_from_array(nodes_arr)
    return stats
