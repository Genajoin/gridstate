"""Mirror V measurements across ktr=1.0 transformer links.

In the input model of a 3-winding autotransformer the wye point (the midpoint) is a
phantom node connected to the real HV bus by a ``branch_type=1`` (transformer) branch
with ``tap_ratio=1.0``. Physically there is no real bus at that point; V and δ on it
equal V/δ on the HV side (a topological identity of the model, not a power-flow
result).

If the HV bus has a **real** V measurement, the wye node should carry the same V
observation, with the same ``value`` and the same ``variance``.
This function adds the corresponding pseudo V measurements **before**
``add_pseudo_measurements``, so that it skips the wye node instead of substituting
the default pseudo-V=Vn.

It works symmetrically in both directions: a real V measurement on the branch's
to-node is copied to its from-node, and vice versa.

Effect: HV-bus voltages are pulled toward the real V measurement, and so are the
wye nodes of 3-winding autotransformers. This cures the asymmetry between
autotransformers with a telemetered HV leg (V_wye=V_HV through the power flow) and
those without (V_wye floated freely; now it is pulled to the real V).

**Decomposition:** the function is split into the
``_mirror_voltage_on_arrays`` **kernel over contract numpy arrays** (it reads
ONLY the contract columns of nodes/branches/measurements) and a thin adapter.
**Append** pattern: the core does not mutate
existing rows but RETURNS a list of new measurement rows
(``list[dict]``), and the adapter adds them via ``model.measurements.add()``
row by row. The logic is sequential (branch traversal order, within-pass
``any_v_by_node`` dedup, monotonic id allocation): ``(value, variance)``
are COPIED without arithmetic; the only float operation is the threshold gate
``|tap-1|>=tol`` (not rounding). The free-id search scans
``meas_arr['id']`` (taken before any ``add``); the collision-skip semantics
are preserved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gridstate._arrays import active_ids
from gridstate.constants import BranchType
from gridstate.measurement_ids import MIRROR_V_ID_BASE, IdAllocator
from gridstate.preprocessing._scan import scan_node_voltage
from gridstate.preprocessing.meas_rows import pseudo_node_measurement
from gridstate.z_vector import KIND_VOLTAGE, SIDE_NONE


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)

__all__ = ["mirror_voltage_through_unit_tap_links"]


def _mirror_voltage_on_arrays(
    nodes_arr: Any,
    branches_arr: Any,
    meas_arr: Any,
    *,
    tap_tolerance: float = 1e-3,
    mid_start: int = MIRROR_V_ID_BASE,
) -> list[dict]:
    """Build new mirror-V rows on the contract arrays (does NOT mutate the inputs).

    Reads ``node.{id,status}``, ``branch.{status,branch_type,tap_ratio,from_node,
    to_node}`` and ``measurement.{status,object_type,measurement_type,object_id,value,
    variance,is_pseudo,id}``; returns ``new_rows: list[dict]`` (ready for
    ``measurements.add``). The free id is found by scanning ``meas_arr['id']`` (the same
    set as the live collection at the moment the array was taken), then increment-only,
    which is the verbatim collision-skip semantics of the original.

    **Must remain a sequential Python loop** over ``branches_arr`` in the
    original order, with local mutation of ``any_v_by_node``/``new_id`` inside the pass
    (within-pass dedup and id allocation are load-bearing; vectorizing would change the result).
    """
    all_v_by_node, real_v_by_node = scan_node_voltage(meas_arr)
    any_v_by_node: set[int] = set(all_v_by_node)

    active_node_ids = active_ids(nodes_arr)

    # Free id: scan the set of existing ids (== the live collection at the moment
    # meas_arr was taken) once at the start, then increment-only inside the pass
    # (a collision further along the range is an error, not a skip).
    ids = IdAllocator(meas_arr["id"], mid_start, name="mirror_voltage", skip_taken_start=True)

    new_rows: list[dict] = []

    def _mirror(src: int, dst: int) -> None:
        """Copy the real V-meas on ``src`` onto the bare node ``dst``.

        ``(value, variance)`` are copied verbatim (no arithmetic); ``dst`` is
        recorded in ``any_v_by_node`` so a later branch does not re-mirror it.
        """
        val, var = real_v_by_node[src]
        new_rows.append(
            pseudo_node_measurement(
                ids.next(),
                dst,
                KIND_VOLTAGE,
                val,
                var,
                branch_side=SIDE_NONE,
                source_code="mirror_unit_tap",
            )
        )
        any_v_by_node.add(dst)

    for r in branches_arr:
        if not r["status"]:
            continue
        if int(r["branch_type"]) != BranchType.TRANSFORMER:
            continue
        if abs(float(r["tap_ratio"]) - 1.0) >= tap_tolerance:
            continue
        f = int(r["from_node"])
        t = int(r["to_node"])
        if f not in active_node_ids or t not in active_node_ids:
            continue
        if f in real_v_by_node and t not in any_v_by_node:
            _mirror(f, t)
        elif t in real_v_by_node and f not in any_v_by_node:
            _mirror(t, f)

    return new_rows


def mirror_voltage_through_unit_tap_links(
    model: Working,
    *,
    tap_tolerance: float = 1e-3,
    mid_start: int = MIRROR_V_ID_BASE,
) -> dict[str, int]:
    """Copy real V measurements across ktr=1.0 transformer links.

    For every active ``branch_type=1`` branch with ``|tap_ratio - 1| <
    tap_tolerance``: if one end has an active real V measurement
    (``object_type=NODE``, ``measurement_type=VOLTAGE``,
    ``is_pseudo=False``), and the other end has neither a real nor a pseudo V measurement,
    add a pseudo V measurement to the "bare" node with the same ``(value,
    variance)``.

    Returns:
        Step statistics ``{"added": <number of mirror-V measurements added>}``.
    """
    new_rows = _mirror_voltage_on_arrays(
        model.nodes.to_numpy(),
        model.branches.to_numpy(),
        model.measurements.to_numpy(),
        tap_tolerance=tap_tolerance,
        mid_start=mid_start,
    )
    # Batch insertion in a single concatenation (per-row .add() = O(n²)).
    model.measurements.add_many(new_rows)

    logger.info("mirror_voltage_through_unit_tap_links: added %d pseudo V-meas", len(new_rows))
    return {"added": len(new_rows)}
