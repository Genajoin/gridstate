"""Deactivation of measurements whose object is switched off or missing.

Split out of the former monolithic telemetry module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridstate._arrays import active_ids
from gridstate.working import table_copy, table_replace


if TYPE_CHECKING:
    import numpy as np

    from gridstate.working import Working


def deactivate_orphan_measurements(model: Working) -> dict[str, int]:
    """Деактивировать measurements, чьи объекты `status=False` или отсутствуют.

    Когда ветвь/узел/генератор отключается (через ON_LINE-формулу,
    `disable_orphan_branches`, cascade и т.д.), associated measurements
    остаются `status=True`. ``z_vector`` пропускает их через
    ``branch_id_to_pos`` / ``bus_id_to_pos``-фильтр, но это засоряет
    логи warning-ами и завышает `len(active_measurements)` в audit.

    Делает универсальную пост-очистку: для каждой active measurement
    проверяет, что её object активен, иначе ставит ``status=False``.

    Args:
        model: ``Working`` после всех topology-операций.

    Returns:
        ``{"branch_meas": N, "node_meas": N, "gen_meas": N,
        "orphan_object_id": N}``.
    """
    meas_arr = table_copy(model.measurements)
    stats = _deactivate_orphan_on_arrays(
        meas_arr,
        model.nodes.to_numpy(),
        model.branches.to_numpy(),
        model.generators.to_numpy(),
    )
    table_replace(model.measurements, meas_arr)
    return stats


def _deactivate_orphan_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    gens_arr: np.ndarray,
) -> dict[str, int]:
    """ЯДРО: деактивация measurements осиротевших объектов над контрактом.

    Строит множества active-id из ``branches_arr``/``nodes_arr``/``gens_arr``, для
    каждой active-меры с неактивным/отсутствующим объектом ставит ``status=False``.
    Mutates ``meas_arr`` in place; no external dependencies.
    """
    active_branches = active_ids(branches_arr)
    active_nodes = active_ids(nodes_arr)
    active_gens = active_ids(gens_arr)

    stats = {"branch_meas": 0, "node_meas": 0, "gen_meas": 0, "orphan_object_id": 0}
    for i in range(len(meas_arr)):
        if not bool(meas_arr[i]["status"]):
            continue
        ot = int(meas_arr[i]["object_type"])
        oid = int(meas_arr[i]["object_id"])
        if ot == 0:  # NODE
            if oid not in active_nodes:
                meas_arr[i]["status"] = False
                stats["node_meas"] += 1
        elif ot == 1:  # BRANCH
            if oid not in active_branches:
                meas_arr[i]["status"] = False
                stats["branch_meas"] += 1
        elif ot == 2:  # GENERATOR
            if oid not in active_gens:
                meas_arr[i]["status"] = False
                stats["gen_meas"] += 1
        else:
            stats["orphan_object_id"] += 1
    return stats
