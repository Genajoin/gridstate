"""ON_LINE-топология: применение ON_LINE-статусов к модели (prep-адаптер + ядро).

Контрактное ядро ``_apply_topology_on_arrays`` + адаптер ``apply_topology_resolved``
(применяет готовый числовой план статусов к ``status`` объектов модели).
"""

from __future__ import annotations


def apply_topology_resolved(model, resolved) -> dict[str, int]:
    """Применить готовый ``resolved``-план ON_LINE к ``status`` модели.

    Чистое применение (без snapshot/формул): снимок контрактных массивов → ядро
    :func:`_apply_topology_on_arrays` → write-back. Зовётся шагом ``run()`` на своей
    позиции (до каскада статусов). ``resolved`` — готовый план статусов
    ``(tag, parent_id, status|None, eval_skip|None)``.
    """
    arr_nodes = model.nodes.to_numpy().copy()
    arr_branches = model.branches.to_numpy().copy()
    arr_gens = model.generators.to_numpy().copy()
    arr_reactors = (
        model.raw_tables.get("reactors").copy()
        if model.raw_tables.get("reactors") is not None
        else None
    )
    stats = _apply_topology_on_arrays(arr_nodes, arr_branches, arr_gens, arr_reactors, resolved)
    model.nodes.update_from_array(arr_nodes)
    model.branches.update_from_array(arr_branches)
    model.generators.update_from_array(arr_gens)
    if arr_reactors is not None:
        model.raw_tables["reactors"] = arr_reactors

    return stats


def _apply_topology_on_arrays(arr_nodes, arr_branches, arr_gens, arr_reactors, resolved):
    """Ф4.1-ЯДРО (слайс 7a): применение ON_LINE-статусов над контрактными массивами.

    Чистый статус-каскад: получает готовый план ``resolved`` (Блокатор-4 — eval
    ON_LINE-формул по snapshot + чтение ``spec.args`` — выполнен в адаптере; элемент
    ``(tag, parent_id, status|None, eval_skip|None)`` в порядке ``specs.items()``),
    матчит ``parent_tag`` → целевой массив (nodes/branches/generators/reactors),
    пишет ``status``-колонку. ``reactors`` — raw-таблица (мутируемая status). Мутирует
    массивы in place. Без PSC/XML/snapshot, без float → строгий бит-в-бит 1e-9.
    """
    by_id_nodes: dict[int, int] = {int(arr_nodes[i]["id"]): i for i in range(len(arr_nodes))}
    by_id_branches: dict[int, int] = {
        int(arr_branches[i]["id"]): i for i in range(len(arr_branches))
    }
    by_id_gens: dict[int, int] = {int(arr_gens[i]["id"]): i for i in range(len(arr_gens))}
    by_id_reacs: dict[int, int] = {}
    if arr_reactors is not None and "id" in arr_reactors.dtype.names:
        by_id_reacs = {int(arr_reactors[i]["id"]): i for i in range(len(arr_reactors))}

    stats = {
        "applied_on": 0,
        "applied_off": 0,
        "skipped_no_value": 0,
        "skipped_partial_args": 0,
        "skipped_no_object": 0,
        "skipped_formula_error": 0,
        "total_specs": len(resolved),
    }

    for tag, parent_id, status, eval_skip in resolved:
        if tag == "NODE":
            idx = by_id_nodes.get(parent_id)
            target = arr_nodes
        elif tag == "LINE":
            idx = by_id_branches.get(parent_id)
            target = arr_branches
        elif tag == "GENERATOR":
            idx = by_id_gens.get(parent_id)
            target = arr_gens
        elif tag == "REACTOR":
            if arr_reactors is None:
                idx = None
                target = None
            else:
                idx = by_id_reacs.get(parent_id)
                target = arr_reactors
        else:
            idx = None
            target = None
        if idx is None or target is None:
            stats["skipped_no_object"] += 1
            continue
        if status is None:
            stats[eval_skip] += 1
            continue

        target[idx]["status"] = status
        if status:
            stats["applied_on"] += 1
        else:
            stats["applied_off"] += 1

    return stats
