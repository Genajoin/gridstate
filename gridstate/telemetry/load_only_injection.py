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


__all__ = ["load_only_injection_nodes", "release_load_only_injections"]


def _kind_base(kind: str) -> str:
    """``PG_G3`` -> ``PG``; other kinds unchanged."""
    return kind.split("_G", 1)[0] if kind.startswith(("PG_G", "QG_G")) else kind


def load_only_injection_nodes(
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
) -> dict[str, frozenset[int]]:
    """Nodes whose injection measurement is built from load components only.

    Mirrors the component filter of the telemetry core: a component counts when
    it has a value and its quality is not BAD. Returns ``{"P": nodes, "Q": nodes}``
    where the P (Q) injection of the node consists of ``PN`` (``QN``) alone.
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
    out: dict[str, set[int]] = {"P": set(), "Q": set()}
    for (node, pq), signs in parts.items():
        if signs == {-1}:
            out[pq].add(node)
    return {pq: frozenset(nodes) for pq, nodes in out.items()}


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
