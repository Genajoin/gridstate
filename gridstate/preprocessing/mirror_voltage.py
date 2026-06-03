"""Mirror V-измерений через ktr=1.0 трансформаторные связи.

Во входной модели 3-обмоточных АТ wye-точка (средняя точка) — это
фантомный узел, соединённый с реальной HV-шиной ветвью
``branch_type=1`` (transformer) с ``tap_ratio=1.0``. Физически в такой
точке нет реальной шины, V и δ на ней равны V/δ на HV-стороне (это
топологическое тождество модели, не результат power-flow).

Если на HV-шине есть **реальное** V-измерение, то на wye-узле должно
быть такое же V-наблюдение — с тем же ``value`` и той же ``variance``.
Эта функция добавляет соответствующие pseudo V-измерения **до**
``add_pseudo_measurements``, чтобы тот пропустил wye-узел при подстановке
дефолтного pseudo-V=Vn.

Симметрично работает в обе стороны: если real V-meas на узле конца
ветви, копируется на узел начала.

Эффект: V_HV-шины подтягиваются к real V-measurement, wye-узлы
3-обмоточных АТ — тоже. Лечит asymmetry между АТ с TI на ВВ-плече
(V_wye=V_HV через power flow) и без (V_wye плавал свободно, теперь —
к real V).

**(CLASS-1 pseudo-слой):** функция расщеплена на
``_mirror_voltage_on_arrays``-**ядро над контрактными numpy-массивами**
(vendor-free, читает ТОЛЬКО контрактные колонки nodes/branches/measurements,
XML не трогает) + тонкий адаптер. В отличие от уже-мигрированных
каскадных функций (мутируют существующую колонку → ``update_from_array``),
здесь паттерн **append**: ядро не мутирует существующие строки, а
ВОЗВРАЩАЕТ список новых measurement-строк (``list[dict]``), а адаптер
добавляет их через ``model.measurements.add()`` построчно (батч-append из
массива в ``_ArrayCollection`` отсутствует). Логика дословно прежняя
(тот же последовательный порядок обхода ветвей, тот же within-pass
``any_v_by_node`` dedup, та же раздача id) → **строгий бит-в-бит 1e-9**:
``(value, variance)`` КОПИРУЮТСЯ без арифметики, единственная float-операция
— порог-гейт ``|tap-1|>=tol`` (не округление). Единственное изменение
механики — поиск свободного id скан-ит ``meas_arr['id']`` вместо итерации
live-коллекции: множество id идентично (``meas_arr`` снят до любого
``add``), collision-skip-семантика сохранена дословно.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


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
    mid_start: int = 200_000_000,
) -> list[dict]:
    """Построить новые mirror-V-строки на контрактных массивах (НЕ мутирует входы).

    Читает ``node.{id,status}``, ``branch.{status,branch_type,tap_ratio,from_node,
    to_node}`` и ``measurement.{status,object_type,measurement_type,object_id,value,
    variance,is_pseudo,id}``; возвращает ``new_rows: list[dict]`` (готовые к
    ``measurements.add``). Свободный id ищется скан-ом ``meas_arr['id']`` (то же
    множество, что live-коллекция на момент снятия массива), затем инкремент-only —
    дословная collision-skip-семантика оригинала.

    **Должно оставаться последовательным Python-циклом** по ``branches_arr`` в
    исходном порядке с локальной мутацией ``any_v_by_node``/``new_id`` внутри прохода
    (within-pass dedup и раздача id — load-bearing; векторизация разрушит бит-в-бит).
    """
    real_v_by_node: dict[int, tuple[float, float]] = {}
    any_v_by_node: set[int] = set()
    for r in meas_arr:
        if not r["status"]:
            continue
        if int(r["object_type"]) != OBJ_NODE:
            continue
        if int(r["measurement_type"]) != KIND_VOLTAGE:
            continue
        nid = int(r["object_id"])
        any_v_by_node.add(nid)
        if not bool(r["is_pseudo"]):
            real_v_by_node[nid] = (float(r["value"]), float(r["variance"]))

    active_node_ids = {int(r["id"]) for r in nodes_arr if r["status"]}

    # Свободный id: скан множества существующих id (== live-коллекция на момент
    # снятия meas_arr) + инкремент-only внутри прохода. Дословно повторяет
    # `while any(int(m.id)==new_id ...): new_id += 1` оригинала.
    existing_ids = {int(x) for x in meas_arr["id"]}
    new_id = mid_start
    while new_id in existing_ids:
        new_id += 1

    new_rows: list[dict] = []
    for r in branches_arr:
        if not r["status"]:
            continue
        if int(r["branch_type"]) != 1:
            continue
        if abs(float(r["tap_ratio"]) - 1.0) >= tap_tolerance:
            continue
        f = int(r["from_node"])
        t = int(r["to_node"])
        if f not in active_node_ids or t not in active_node_ids:
            continue
        if f in real_v_by_node and t not in any_v_by_node:
            val, var = real_v_by_node[f]
            new_rows.append(
                {
                    "id": new_id,
                    "object_type": OBJ_NODE,
                    "object_id": t,
                    "measurement_type": KIND_VOLTAGE,
                    "value": val,
                    "variance": var,
                    "status": True,
                    "quality": 0,
                    "is_pseudo": True,
                    "branch_side": -1,
                    "source_code": "mirror_unit_tap",
                }
            )
            new_id += 1
            any_v_by_node.add(t)
        elif t in real_v_by_node and f not in any_v_by_node:
            val, var = real_v_by_node[t]
            new_rows.append(
                {
                    "id": new_id,
                    "object_type": OBJ_NODE,
                    "object_id": f,
                    "measurement_type": KIND_VOLTAGE,
                    "value": val,
                    "variance": var,
                    "status": True,
                    "quality": 0,
                    "is_pseudo": True,
                    "branch_side": -1,
                    "source_code": "mirror_unit_tap",
                }
            )
            new_id += 1
            any_v_by_node.add(f)

    return new_rows


def mirror_voltage_through_unit_tap_links(
    model: Working,
    *,
    tap_tolerance: float = 1e-3,
    mid_start: int = 200_000_000,
) -> int:
    """Скопировать real V-измерения через ktr=1.0 trafo-связи.

    Для каждой активной ветви ``branch_type=1`` с ``|tap_ratio - 1| <
    tap_tolerance``: если у одного конца есть active real V-meas
    (``object_type=NODE``, ``measurement_type=VOLTAGE``,
    ``is_pseudo=False``), а у другого нет ни real, ни pseudo V-meas,
    добавить pseudo V-meas на «голый» узел с тем же ``(value,
    variance)``.

    Returns:
        Количество добавленных mirror-V измерений.
    """
    new_rows = _mirror_voltage_on_arrays(
        model.nodes.to_numpy(),
        model.branches.to_numpy(),
        model.measurements.to_numpy(),
        tap_tolerance=tap_tolerance,
        mid_start=mid_start,
    )
    for row in new_rows:
        model.measurements.add(row)

    logger.info("mirror_voltage_through_unit_tap_links: добавлено %d pseudo V-meas", len(new_rows))
    return len(new_rows)
