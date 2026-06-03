"""Vnom узлов: применение U_NOM к ``model.nodes.voltage_nominal`` (prep-адаптер).

``apply_voltage_nominal_resolved`` — применяет готовую карту ``{node_id → vn}`` к узлам.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gridstate.working import Working


def apply_voltage_nominal_resolved(model: Working, vn_by_id: dict[int, float]) -> dict[str, int]:
    """Заполнить ``voltage_nominal=0`` узлов из готовой карты ``vn_by_id``.

    Контрактное применение: пишет ``voltage_nominal`` только там, где он 0
    (``already_set`` иначе). ``vn_by_id`` — готовая карта ``{node_id → vn}``.
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
