"""Генераторы: статус от узла (каскад) + агрегация multi-gen в узловую генерацию.

Выделено из telemetry/topology.py (раскол по концернам).

**Декомпозиция:** каждая функция расщеплена на ``_*_on_arrays``-**ядро над
контрактными numpy-массивами** (мутирует переданные массивы in-place, возвращает
stats) + тонкая обёртка (``to_numpy().copy()`` → ядро → ``update_from_array``).
Ядро не зависит от способа записи: обёртка лишь снимает массив, зовёт ядро и
пишет результат обратно.
"""

from __future__ import annotations

from typing import Any

from gridstate.bounds import is_sentinel
from gridstate.utils import id_to_pos_map


def _apply_generator_status_on_arrays(nodes_arr: Any, gens_arr: Any) -> dict[str, int]:
    """Каскад ``node off ⇒ gen off`` на контрактных массивах (мутирует ``gens_arr``).

    Читает ``node.{id,status}`` и ``gen.{node_id,status}``; ставит
    ``gens_arr[i]["status"]=False`` для активных генераторов на off-узлах.
    Возвращает ``{"applied_off": N, "missing_node": N}``.
    """
    node_status = {
        int(i): bool(s) for i, s in zip(nodes_arr["id"], nodes_arr["status"], strict=True)
    }

    stats = {"applied_off": 0, "missing_node": 0}
    for i in range(len(gens_arr)):
        nid = int(gens_arr[i]["node_id"])
        node_active = node_status.get(nid)
        if node_active is None:
            stats["missing_node"] += 1
            continue
        if not node_active and bool(gens_arr[i]["status"]):
            gens_arr[i]["status"] = False
            stats["applied_off"] += 1
    return stats


def apply_generator_status_from_node(model: Any) -> dict[str, int]:
    """Каскадное отключение: если узел off → все генераторы на нём off.

    Это **только cascade-off**: при ``node.status=False`` все генераторы
    с этим ``node_id`` принудительно ставятся ``status=False``. Если
    узел on — статус генератора **не трогается**: на одном узле
    возможно несколько ген. с разным статусом (один в работе, другой
    в резерве), это нормально.

    Закрывает астра-инвариант ``(node off ⇒ gen off)`` который наш
    каскад на узлы (``disable_isolated_nodes`` и т.д.) не передавал
    автоматически на генераторы.

    Должна вызываться **после** всех каскад-функций на узлы.

    Returns:
        ``{"applied_off": N, "missing_node": N}``.
    """
    nodes_arr = model.nodes.to_numpy()
    arr = model.generators.to_numpy().copy()
    stats = _apply_generator_status_on_arrays(nodes_arr, arr)
    model.generators.update_from_array(arr)
    return stats


def _aggregate_generators_on_arrays(nodes_arr: Any, gens_arr: Any) -> dict[str, int]:
    """Сумма параметров active-генераторов узла → gen-поля ``nodes_arr`` (мутирует).

    Читает ``gen.{node_id,status,power_output,reactive_output,power_min,power_max,
    reactive_min,reactive_max}`` и ``node.id``; пишет ``node.{generation_p,
    generation_q,generation_p_min,generation_p_max,generation_q_min,generation_q_max}``.

    Лимиты: суммируются только **валидные** пары; если хотя бы один
    активный генератор узла несёт сентинел (|лимит| ≥ ``SENTINEL_ABS``,
    «нет данных»), узловая пара помечается сентинелом ±9999 — диапазон
    узла неизвестен. Раньше сентинелы суммировались как числа: реальный
    диапазон [-50, 100] + сентинельный [-9999, 9999] давал мусор
    [-10049, 10099] — узел либо терял box-var в IPM, либо ложно
    классифицировался BUS-эквивалентом с tight-prior к нулю.

    Возвращает ``{"updated_nodes":N,"active_gens":N,"off_gens":N,
    "missing_node":N,"sentinel_p_nodes":N,"sentinel_q_nodes":N}``.
    """
    node_pos = id_to_pos_map(nodes_arr["id"])

    stats = {
        "updated_nodes": 0,
        "active_gens": 0,
        "off_gens": 0,
        "missing_node": 0,
        "sentinel_p_nodes": 0,
        "sentinel_q_nodes": 0,
    }

    # Сначала обнуляем gen-поля у всех узлов, чтобы повторный вызов
    # давал стабильный результат (off-генераторы перестают учитываться).
    touched: set[int] = set()
    aggregates: dict[int, dict[str, float]] = {}
    # Узлы, где хотя бы один генератор без валидной P/Q-пары лимитов.
    p_unknown: set[int] = set()
    q_unknown: set[int] = set()

    for i in range(len(gens_arr)):
        nid = int(gens_arr[i]["node_id"])
        if nid not in node_pos:
            stats["missing_node"] += 1
            continue
        if not bool(gens_arr[i]["status"]):
            stats["off_gens"] += 1
            continue
        stats["active_gens"] += 1
        agg = aggregates.setdefault(
            nid,
            {
                "p": 0.0,
                "q": 0.0,
                "p_min": 0.0,
                "p_max": 0.0,
                "q_min": 0.0,
                "q_max": 0.0,
            },
        )
        agg["p"] += float(gens_arr[i]["power_output"])
        agg["q"] += float(gens_arr[i]["reactive_output"])

        p_min = float(gens_arr[i]["power_min"])
        p_max = float(gens_arr[i]["power_max"])
        if is_sentinel(p_min) or is_sentinel(p_max):
            p_unknown.add(nid)
        else:
            agg["p_min"] += p_min
            agg["p_max"] += p_max

        q_min = float(gens_arr[i]["reactive_min"])
        q_max = float(gens_arr[i]["reactive_max"])
        if is_sentinel(q_min) or is_sentinel(q_max):
            q_unknown.add(nid)
        else:
            agg["q_min"] += q_min
            agg["q_max"] += q_max

    for nid, agg in aggregates.items():
        i = node_pos[nid]
        nodes_arr[i]["generation_p"] = agg["p"]
        nodes_arr[i]["generation_q"] = agg["q"]
        if nid in p_unknown:
            nodes_arr[i]["generation_p_min"] = -9999.0
            nodes_arr[i]["generation_p_max"] = 9999.0
        else:
            nodes_arr[i]["generation_p_min"] = agg["p_min"]
            nodes_arr[i]["generation_p_max"] = agg["p_max"]
        if nid in q_unknown:
            nodes_arr[i]["generation_q_min"] = -9999.0
            nodes_arr[i]["generation_q_max"] = 9999.0
        else:
            nodes_arr[i]["generation_q_min"] = agg["q_min"]
            nodes_arr[i]["generation_q_max"] = agg["q_max"]
        touched.add(nid)

    stats["updated_nodes"] = len(touched)
    stats["sentinel_p_nodes"] = len(p_unknown)
    stats["sentinel_q_nodes"] = len(q_unknown)
    return stats


def aggregate_generators_to_node(model: Any) -> dict[str, int]:
    """Сумма параметров активных генераторов узла → ``model.nodes``.

    Для каждого узла суммирует по **активным** (``status=True``) генераторам
    с этим ``node_id``:

    * ``power_output``    → ``node.generation_p``;
    * ``reactive_output`` → ``node.generation_q``;
    * ``power_min``       → ``node.generation_p_min``;
    * ``power_max``       → ``node.generation_p_max``;
    * ``reactive_min``    → ``node.generation_q_min``;
    * ``reactive_max``    → ``node.generation_q_max``.

    Off-генераторы исключаются из всех сумм (их Q-лимиты не объединяются
    в Q-диапазон узла). Узлы без active-генераторов остаются с теми
    значениями generation_p_min/max и generation_q_min/max что были до
    вызова (NODE_DTYPE-default = 0). Сентинельные лимиты (±9999, «нет
    данных») не суммируются: узел с хотя бы одним таким генератором
    получает сентинельную пару — «диапазон неизвестен» (см.
    ``_aggregate_generators_on_arrays``).

    Идемпотентна: повторный вызов даёт тот же результат, потому что
    суммы перезаписываются полностью (не аккумулируются).

    Должна вызываться **после** применения ON_LINE-топологии и
    ``apply_generator_status_from_node`` — когда финальные статусы
    генераторов уже выставлены.

    Returns:
        ``{"updated_nodes": N, "active_gens": N, "off_gens": N,
        "missing_node": N}``.
    """
    nodes_arr = model.nodes.to_numpy().copy()
    gens = model.generators.to_numpy()
    stats = _aggregate_generators_on_arrays(nodes_arr, gens)
    model.nodes.update_from_array(nodes_arr)
    return stats
