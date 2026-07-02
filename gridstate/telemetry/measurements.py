"""Слияние/деактивация дубль-измерений z-вектора (F-step эталонной OC).

Выделено из telemetry/topology.py (раскол по концернам).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from gridstate.working import Working


def resolve_merged_measurement_conflicts(model: Working) -> dict[str, int]:
    """Слить дубликаты measurements на одном объекте.

    Контекст: исторически использовалось после
    :func:`apply_breaker_node_merge` (удалена), но логика общая —
    несколько измерений на одном объекте могут возникнуть в любой
    топологической свёртке. Сейчас функция нейтральна на типовом XML
    (после удаления merge дублей нет), но оставлена как защитный
    шаг pipeline.

    Стратегия:

    * V (mt=2): weighted average ``V_new = Σ(V_i·w_i)/Σ w_i``,
      новая variance ``var_new = 1/Σ w_i`` (closed-form maximum-
      likelihood для нормальных независимых).
    * P_inj (mt=4) / Q_inj (mt=5): ``v_new = Σ v_i``,
      ``var_new = Σ var_i`` (аддитивные величины).
    * Остальные кроме одного — деактивируются (status=False).

    Не трогает branch-level измерения (mt=0/1, side=0/1).

    Args:
        model: ``Working``.

    Returns:
        ``{"resolved_v": N, "resolved_p_inj": N, "resolved_q_inj": N,
        "deactivated": N}``.
    """
    meas_arr = model.measurements.to_numpy().copy()
    stats = _resolve_merged_on_arrays(meas_arr)
    model.measurements.update_from_array(meas_arr)
    return stats


def _resolve_merged_on_arrays(meas_arr: np.ndarray) -> dict[str, int]:
    """ЯДРО: слияние дубль-measurements над контрактным массивом.

    Группирует active NODE-меры (``object_type==0``, ``measurement_type∈{2,4,5}``)
    по ключу ``(ot, object_id, mt, branch_side)``; для групп >1: V (mt=2) —
    weighted-average по ``1/variance`` (var clamp ``1e-12``), P/Q_inj (mt=4/5) —
    сумма value и variance. Оставляет первую строку (по порядку массива), остальные
    ``status=False``. ``weight`` первой строки = ``1/var`` (бывш. приватный
    ``me._weight``). Мутирует ``meas_arr`` in place. БЕЗ внешних зависимостей и XML. Порядок строк =
    порядок объектов.
    """
    keys: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for i in range(len(meas_arr)):
        if not bool(meas_arr[i]["status"]):
            continue
        ot = int(meas_arr[i]["object_type"])
        mt = int(meas_arr[i]["measurement_type"])
        # Нас интересуют только NODE-level measurements
        # (V=2, P_inj=4, Q_inj=5). Branch (ot=1) skip.
        if ot != 0:
            continue
        if mt not in (2, 4, 5):
            continue
        k = (ot, int(meas_arr[i]["object_id"]), mt, int(meas_arr[i]["branch_side"]))
        keys[k].append(i)

    stats = {
        "resolved_v": 0,
        "resolved_p_inj": 0,
        "resolved_q_inj": 0,
        "deactivated": 0,
    }
    for k, idxs in keys.items():
        if len(idxs) <= 1:
            continue
        _ot, _oid, mt, _side = k

        if mt == 2:
            # V — weighted average по 1/variance.
            total_w = 0.0
            wsum = 0.0
            for i in idxs:
                v = max(float(meas_arr[i]["variance"]), 1e-12)
                w = 1.0 / v
                total_w += w
                wsum += float(meas_arr[i]["value"]) * w
            new_value = wsum / total_w
            new_var = 1.0 / total_w
            stats["resolved_v"] += 1
        else:
            # P_inj/Q_inj — сумма (аддитивные величины), variance — сумма.
            new_value = sum(float(meas_arr[i]["value"]) for i in idxs)
            new_var = sum(float(meas_arr[i]["variance"]) for i in idxs)
            if mt == 4:
                stats["resolved_p_inj"] += 1
            else:
                stats["resolved_q_inj"] += 1

        # Оставляем первый, остальные деактивируем.
        first = idxs[0]
        meas_arr[first]["value"] = float(new_value)
        meas_arr[first]["variance"] = float(new_var)
        meas_arr[first]["weight"] = 1.0 / float(new_var) if new_var > 0 else 0.0
        for i in idxs[1:]:
            meas_arr[i]["status"] = False
            stats["deactivated"] += 1

    return stats


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
    meas_arr = model.measurements.to_numpy().copy()
    stats = _deactivate_orphan_on_arrays(
        meas_arr,
        model.nodes.to_numpy(),
        model.branches.to_numpy(),
        model.generators.to_numpy(),
    )
    model.measurements.update_from_array(meas_arr)
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
    Мутирует ``meas_arr`` in place. БЕЗ внешних зависимостей и XML.
    """
    active_branches = {int(b["id"]) for b in branches_arr if b["status"]}
    active_nodes = {int(n["id"]) for n in nodes_arr if n["status"]}
    active_gens = {int(g["id"]) for g in gens_arr if g["status"]}

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
