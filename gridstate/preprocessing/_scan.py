"""Shared read-only scans over contract measurement/branch arrays.

Collects the "gather per-node measurements by kind" loops that were copy-pasted
across the preprocessing steps. All helpers are pure reads and preserve the exact filtering semantics of the former
inline loops (bit-for-bit): active (``status``) rows only, NODE-object filtering,
and the ``is_pseudo`` guard behaviour where it applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


if TYPE_CHECKING:
    import numpy as np


__all__ = [
    "scan_node_meas_by_kind",
    "scan_node_voltage",
]


def scan_node_meas_by_kind(
    meas_arr: np.ndarray,
    kinds: tuple[int, ...],
    *,
    real_only: bool = False,
) -> dict[int, set[int]]:
    """Map each requested kind to the set of node ids carrying such a measurement.

    Iterates ``meas_arr`` once. Considers only active (``status``) NODE-object
    measurements whose ``measurement_type`` is in ``kinds``. When ``real_only`` is
    set, rows flagged ``is_pseudo`` are skipped (the ``is_pseudo`` column is probed
    once via ``dtype.names``; absent, every row counts as real).
    """
    out: dict[int, set[int]] = {k: set() for k in kinds}
    names = meas_arr.dtype.names
    has_pseudo = names is not None and "is_pseudo" in names
    for r in meas_arr:
        if not r["status"]:
            continue
        if int(r["object_type"]) != OBJ_NODE:
            continue
        if real_only and has_pseudo and bool(r["is_pseudo"]):
            continue
        kind = int(r["measurement_type"])
        bucket = out.get(kind)
        if bucket is not None:
            bucket.add(int(r["object_id"]))
    return out


def scan_node_voltage(
    meas_arr: np.ndarray,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """Return ``(all_v, real_v)`` value/variance maps for active NODE voltage meas.

    ``all_v`` maps every node with an active VOLTAGE measurement to its last
    ``(value, variance)`` (real or pseudo); ``real_v`` restricts that to non-pseudo
    rows. Last-write-wins on duplicate node ids (build order), matching the former
    inline loops in the chain/mirror voltage steps.
    """
    all_v: dict[int, tuple[float, float]] = {}
    real_v: dict[int, tuple[float, float]] = {}
    names = meas_arr.dtype.names
    has_pseudo = names is not None and "is_pseudo" in names
    for r in meas_arr:
        if not r["status"]:
            continue
        if int(r["object_type"]) != OBJ_NODE:
            continue
        if int(r["measurement_type"]) != KIND_VOLTAGE:
            continue
        nid = int(r["object_id"])
        vv = (float(r["value"]), float(r["variance"]))
        all_v[nid] = vv
        if not (has_pseudo and bool(r["is_pseudo"])):
            real_v[nid] = vv
    return all_v, real_v
