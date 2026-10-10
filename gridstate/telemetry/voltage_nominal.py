"""Node Vnom: apply U_NOM to ``model.nodes.voltage_nominal`` (prep adapter).

``apply_voltage_nominal_resolved`` applies a ready ``{node_id → vn}`` map to the nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gridstate.working import Working


def apply_voltage_nominal_resolved(model: Working, vn_by_id: dict[int, float]) -> dict[str, int]:
    """Fill nodes with ``voltage_nominal=0`` from the ready ``vn_by_id`` map.

    Contract application: writes ``voltage_nominal`` only where it is 0
    (counted as ``already_set`` otherwise). ``vn_by_id`` is a ready ``{node_id → vn}`` map.
    """
    arr = model.nodes.to_numpy().copy()
    applied = 0
    already_set = 0
    missing = 0
    for i in range(len(arr)):
        cur = float(arr[i]["voltage_nominal"])
        if cur > 0:
            already_set += 1
            continue
        nid = int(arr[i]["id"])
        new_vn = vn_by_id.get(nid)
        if new_vn is None:
            missing += 1
            continue
        arr[i]["voltage_nominal"] = new_vn
        applied += 1

    model.nodes.update_from_array(arr)
    return {
        "applied": applied,
        "already_set": already_set,
        "missing": missing,
        "total": len(arr),
    }
