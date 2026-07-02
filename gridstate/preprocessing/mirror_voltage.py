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

**Декомпозиция:** функция расщеплена на
``_mirror_voltage_on_arrays``-**ядро над контрактными numpy-массивами**
(читает ТОЛЬКО контрактные колонки nodes/branches/measurements,
XML не трогает) + тонкий адаптер. Паттерн **append**: ядро не мутирует
существующие строки, а ВОЗВРАЩАЕТ список новых measurement-строк
(``list[dict]``), а адаптер добавляет их через ``model.measurements.add()``
построчно. Логика последовательная (порядок обхода ветвей, within-pass
``any_v_by_node`` dedup, монотонная раздача id): ``(value, variance)``
КОПИРУЮТСЯ без арифметики, единственная float-операция — порог-гейт
``|tap-1|>=tol`` (не округление). Поиск свободного id скан-ит
``meas_arr['id']`` (снят до любого ``add``), collision-skip-семантика
сохранена.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gridstate.preprocessing._scan import scan_node_voltage
from gridstate.preprocessing.meas_rows import pseudo_node_measurement
from gridstate.z_vector import KIND_VOLTAGE


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
    (within-pass dedup и раздача id — load-bearing; векторизация изменит результат).
    """
    all_v_by_node, real_v_by_node = scan_node_voltage(meas_arr)
    any_v_by_node: set[int] = set(all_v_by_node)

    active_node_ids = {int(r["id"]) for r in nodes_arr if r["status"]}

    # Свободный id: скан множества существующих id (== live-коллекция на момент
    # снятия meas_arr) + инкремент-only внутри прохода. Дословно повторяет
    # `while any(int(m.id)==new_id ...): new_id += 1` оригинала.
    existing_ids = {int(x) for x in meas_arr["id"]}
    new_id = mid_start
    while new_id in existing_ids:
        new_id += 1

    new_rows: list[dict] = []

    def _mirror(src: int, dst: int) -> None:
        """Copy the real V-meas on ``src`` onto the bare node ``dst``.

        ``(value, variance)`` are copied verbatim (no arithmetic); ``dst`` is
        recorded in ``any_v_by_node`` so a later branch does not re-mirror it.
        """
        nonlocal new_id
        val, var = real_v_by_node[src]
        new_rows.append(
            pseudo_node_measurement(
                new_id,
                dst,
                KIND_VOLTAGE,
                val,
                var,
                branch_side=-1,
                source_code="mirror_unit_tap",
            )
        )
        new_id += 1
        any_v_by_node.add(dst)

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
            _mirror(f, t)
        elif t in real_v_by_node and f not in any_v_by_node:
            _mirror(t, f)

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
    # Пакетная вставка за одну конкатенацию (per-row .add() = O(n²)).
    model.measurements.add_many(new_rows)

    logger.info("mirror_voltage_through_unit_tap_links: добавлено %d pseudo V-meas", len(new_rows))
    return len(new_rows)
