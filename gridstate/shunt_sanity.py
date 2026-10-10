"""Shunt-sanity: try-off/flip of suspicious shunts, gated by ΔΣrn² (research).

Defective shunt records (inverted sign of B, wrong status) are statically
indistinguishable from honest ones: type/sign/status in the source data are
consistent, and the defect shows up only behaviourally — the solved node V
systematically disagrees with its V measurement, and the network "prefers"
a different element.

Mechanism (a two-pass family, see ``pipeline.passes._refine_two_pass``, but with
per-candidate trial runs):

1. Candidates: active shunts on nodes whose node V measurement disagrees with the
   solution, ``|z − h| > v_frac · Vnom`` (an internal signal; no outside solution
   is needed).
2. For each candidate: snapshot → edit (``off``: status=False;
   ``flip``: −B) → warm re-solve → ΔΣrn² over the live real measurements.
3. Gate: a variant is accepted only if ``ΔΣrn² < −gate_drop`` (substantially
   better agreement with its own measurements); otherwise the snapshot is
   fully rolled back.

The gate is selective both ways (validation on four large models:
one candidate accepted, dVmax −40.7 %; 7 false candidates on three other
models rejected). ⚠️ Research flag:
the step EDITS the network based on measurement statistics — on live
telemetry the solution may flicker between slices; for a production fix an
offline audit of the data source is preferable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import id_value_map, meas_mask
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


if TYPE_CHECKING:
    from gridstate.working import Working

__all__ = [
    "ShuntSanityPlan",
    "classify_shunt_candidates",
    "edit_shunt",
    "sum_rn2",
]


@dataclass(frozen=True)
class ShuntSanityPlan:
    """Candidate nodes (each has an active shunt and a large V residual)."""

    candidates: tuple[int, ...]

    @property
    def empty(self) -> bool:
        return not self.candidates


def sum_rn2(measurements: np.ndarray) -> float:
    """Σ((z−h)²/σ²) over the live real measurements — a scalar of solution/measurement agreement."""
    sel = meas_mask(measurements, pseudo=False, finite_estimate=True)
    z = measurements["value"][sel]
    h = measurements["estimated_si"][sel]
    var = np.maximum(measurements["variance"][sel], 1e-12)
    return float((((z - h) ** 2) / var).sum())


def classify_shunt_candidates(
    measurements: np.ndarray,
    nodes: np.ndarray,
    shunts: np.ndarray,
    *,
    v_frac: float,
    max_candidates: int,
) -> ShuntSanityPlan:
    """Nodes with active shunts where the node V measurement disagrees with the solution.

    Candidate: an active shunt (``shunts.status``) on a node that has a real
    node V measurement in a plausible range (0.5–1.5 Vnom) with
    ``|z − h| > v_frac · Vnom`` (h = the node's solved ``voltage_magnitude``).
    Returns at most ``max_candidates`` nodes (by descending residual) —
    each candidate costs up to two warm re-solves.
    """
    vn = id_value_map(nodes, "voltage_nominal")
    vh = id_value_map(nodes, "voltage_magnitude")
    shunt_nodes = {
        int(n) for n, st in zip(shunts["node_id"], shunts["status"], strict=True) if bool(st)
    }
    # No status filter: an inactive real V reading still nominates its node.
    selv = meas_mask(measurements, kinds=KIND_VOLTAGE, obj=OBJ_NODE, active=False, pseudo=False)
    worst: dict[int, float] = {}
    for j in np.where(selv)[0]:
        nid = int(measurements["object_id"][j])
        if nid not in shunt_nodes:
            continue
        vnom = vn.get(nid, 0.0)
        z = float(measurements["value"][j])
        h = vh.get(nid, float("nan"))
        if vnom <= 0 or not (0.5 * vnom < z < 1.5 * vnom) or not np.isfinite(h):
            continue
        dev = abs(z - h)
        if dev > v_frac * vnom:
            worst[nid] = max(worst.get(nid, 0.0), dev)
    ranked = sorted(worst, key=lambda n: -worst[n])[:max_candidates]
    return ShuntSanityPlan(candidates=tuple(ranked))


def edit_shunt(model: Working, node_id: int, mode: str) -> int:
    """Apply an edit to ALL shunts of the node: ``off`` | ``flip``. Returns the number of rows."""
    coll = model.shunts
    arr = coll.to_numpy()
    sel = arr["node_id"] == node_id
    n = int(sel.sum())
    if n == 0:
        return 0
    if mode == "off":
        arr["status"][sel] = False
    elif mode == "flip":
        arr["susceptance"][sel] = -arr["susceptance"][sel]
    else:  # pragma: no cover — guarded by the caller
        raise ValueError(f"unknown shunt edit mode: {mode!r}")
    coll.update_from_array(arr)
    return n
