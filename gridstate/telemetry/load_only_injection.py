"""Release node injection measurements that carry only the load component.

A node injection measurement is assembled from whatever node components arrive
in the telemetry (``PG``/``PN``/``QG``/``QN``, generator ``PG_G*``/``QG_G*``;
see :func:`gridstate.telemetry.apply_resolved._build_node_injection_rows`). When
a node with generation reports only its load (typically the auxiliary load of a
power unit) and no generation, the sum is ``-load`` and, with its tight sigma,
states that the node generates nothing. The unit output then has nowhere to go:
the estimator keeps the injection near ``-load`` and violates the measured flows
of the unit transformer instead.

Such a measurement is partial, not a net injection. This module finds the nodes
whose injection was built from load components only and, on nodes that do have
generation, deactivates it: the generation is left free within its box (IPM) or
determined by the surrounding flows (WLS). The released nodes are reported so the
pseudo-injection prior can be widened to the generation range instead of pulling
the unmeasured generation back to its (usually zero) materialized value.

The mirror case is an injection built from generation components only on a node
that also has load: ``+PG`` states that the load is zero. The generation reading
is valid there, so the measurement is kept but turned into what it actually
constrains: with the load known only as its box ``[lo, hi]``, the net injection
is ``PG - load`` with ``load`` anywhere in the box. The value is shifted by the
box centre and the variance grows by that of a load uniform in the box,
``(hi - lo)**2 / 12``. Both algorithms use it as is: WLS gets an honest net
injection, IPM lets the load settle within its box while the generation reading
still holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridstate.bounds import resolve_bounds
from gridstate.constants import FilterFlag
from gridstate.telemetry._specs import _INJ_MT, _NODE_INJ_MAP
from gridstate.telemetry.quality import QUALITY_BAD


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = [
    "generation_only_injection_nodes",
    "load_only_injection_nodes",
    "release_load_only_injections",
    "widen_generation_only_injections",
]


def _kind_base(kind: str) -> str:
    """``PG_G3`` -> ``PG``; other kinds unchanged."""
    return kind.split("_G", 1)[0] if kind.startswith(("PG_G", "QG_G")) else kind


def _injection_signs(
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
) -> dict[tuple[int, str], set[int]]:
    """``(node, "P"|"Q") -> {component signs}``: +1 generation, -1 load.

    Mirrors the component filter of the telemetry core: a component counts when
    it has a value and its quality is not BAD.
    """
    parts: dict[tuple[int, str], set[int]] = {}
    for obj_id, kind in arg_keys:
        base = _kind_base(kind)
        if base not in _NODE_INJ_MAP:
            continue
        value, _n_res, _guid, quality = resolved[(obj_id, kind)]
        if value is None or quality == QUALITY_BAD:
            continue
        pq, sign = _NODE_INJ_MAP[base]
        parts.setdefault((int(obj_id), pq), set()).add(sign)
    return parts


def _nodes_with_signs(
    parts: dict[tuple[int, str], set[int]], signs: set[int]
) -> dict[str, frozenset[int]]:
    out: dict[str, set[int]] = {"P": set(), "Q": set()}
    for (node, pq), node_signs in parts.items():
        if node_signs == signs:
            out[pq].add(node)
    return {pq: frozenset(nodes) for pq, nodes in out.items()}


def load_only_injection_nodes(
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
) -> dict[str, frozenset[int]]:
    """Nodes whose injection measurement is built from load components only.

    Returns ``{"P": nodes, "Q": nodes}`` where the P (Q) injection of the node
    consists of ``PN`` (``QN``) alone.
    """
    return _nodes_with_signs(_injection_signs(resolved, arg_keys), {-1})


def generation_only_injection_nodes(
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
) -> dict[str, frozenset[int]]:
    """Nodes whose injection measurement is built from generation components only.

    Returns ``{"P": nodes, "Q": nodes}`` where the P (Q) injection of the node
    consists of ``PG``/``PG_G*`` (``QG``/``QG_G*``) alone.
    """
    return _nodes_with_signs(_injection_signs(resolved, arg_keys), {+1})


def release_load_only_injections(
    model: Working,
    load_only: dict[str, frozenset[int]],
) -> tuple[dict, dict[str, frozenset[int]]]:
    """Deactivate load-only injection measurements on nodes with generation.

    Must run after the generator aggregation: ``exist_gen`` and the generation
    ranges of a node declared only through its generator catalogue are filled
    there. Only nodes with a declared active generation range (any sign) are
    touched.

    Returns ``(stats, released)``: counts ``released_p``/``released_q`` and the
    released node sets ``{"P": ..., "Q": ...}`` (for the pseudo-injection prior).
    """
    # Generation counts only with a declared, non-degenerate active range of any
    # sign: pumped storage may be declared ``[-200, 0]`` (pumping only) or
    # ``[-1200, 1200]``. A node flagged ``exist_gen`` with an unset range
    # (``[0, 0]`` or sentinels: boundary equivalents, compensators) is left alone.
    nodes = model.nodes.to_numpy()
    gen_nodes: set[int] = set()
    for row in nodes:
        if not bool(row["status"]) or not bool(row["exist_gen"]):
            continue
        lo, hi = resolve_bounds(float(row["generation_p_min"]), float(row["generation_p_max"]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            gen_nodes.add(int(row["id"]))

    targets = {
        _INJ_MT["P"]: load_only.get("P", frozenset()) & gen_nodes,
        _INJ_MT["Q"]: load_only.get("Q", frozenset()) & gen_nodes,
    }
    meas = model.measurements.to_numpy().copy()
    base = (
        meas["status"].astype(bool) & ~meas["is_pseudo"].astype(bool) & (meas["object_type"] == 0)
    )
    released: dict[int, set[int]] = {}
    for mt, node_set in targets.items():
        if not node_set:
            released[mt] = set()
            continue
        sel = base & (meas["measurement_type"] == mt) & np.isin(meas["object_id"], list(node_set))
        meas["status"][sel] = False
        meas["filter_flag"][sel] = int(FilterFlag.LOAD_ONLY_INJECTION)
        released[mt] = {int(x) for x in meas["object_id"][sel]}
    if any(released.values()):
        model.measurements.update_from_array(meas)
    nodes_p = frozenset(released[_INJ_MT["P"]])
    nodes_q = frozenset(released[_INJ_MT["Q"]])
    return {"released_p": len(nodes_p), "released_q": len(nodes_q)}, {"P": nodes_p, "Q": nodes_q}


def widen_generation_only_injections(
    model: Working,
    gen_only: dict[str, frozenset[int]],
) -> dict:
    """Account for the unmeasured load in generation-only injection measurements.

    On an active node with load and a declared load box ``[lo, hi]`` the P (Q)
    injection built from generation alone gets ``value -= (lo + hi) / 2`` and
    ``variance += (hi - lo)**2 / 12``. Nodes without load or with an unset box
    (``[0, 0]``, sentinels) are left alone.
    """
    nodes = model.nodes.to_numpy()
    box: dict[tuple[int, int], tuple[float, float]] = {}
    for row in nodes:
        if not bool(row["status"]) or not bool(row["exist_load"]):
            continue
        nid = int(row["id"])
        for pq, lo_col, hi_col in (
            ("P", "load_p_min", "load_p_max"),
            ("Q", "load_q_min", "load_q_max"),
        ):
            if nid not in gen_only.get(pq, frozenset()):
                continue
            lo, hi = resolve_bounds(float(row[lo_col]), float(row[hi_col]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                box[(nid, _INJ_MT[pq])] = (lo, hi)

    counts = {_INJ_MT["P"]: 0, _INJ_MT["Q"]: 0}
    if box:
        meas = model.measurements.to_numpy().copy()
        sel = (
            meas["status"].astype(bool)
            & ~meas["is_pseudo"].astype(bool)
            & (meas["object_type"] == 0)
            & np.isin(meas["measurement_type"], list(counts))
        )
        for k in np.flatnonzero(sel):
            mt = int(meas["measurement_type"][k])
            bounds = box.get((int(meas["object_id"][k]), mt))
            if bounds is None:
                continue
            lo, hi = bounds
            meas["value"][k] -= 0.5 * (lo + hi)
            meas["variance"][k] += (hi - lo) ** 2 / 12.0
            meas["weight"][k] = 1.0 / meas["variance"][k]
            counts[mt] += 1
        if any(counts.values()):
            model.measurements.update_from_array(meas)
    return {"widened_p": counts[_INJ_MT["P"]], "widened_q": counts[_INJ_MT["Q"]]}
