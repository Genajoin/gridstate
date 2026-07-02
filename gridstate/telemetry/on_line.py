"""ON_LINE-топология: применение ON_LINE-статусов к модели.

Контрактное ядро ``_apply_topology_on_arrays`` + тонкая обёртка ``apply_topology_resolved``
(применяет готовый числовой план статусов к ``status`` объектов модели).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gridstate.utils import id_to_pos_map


# Элемент плана статусов ON_LINE: ``(tag, parent_id, status|None, eval_skip|None)``.
ResolvedItem = tuple[str, int, bool | None, str | None]


def apply_topology_resolved(model: Any, resolved: Sequence[ResolvedItem]) -> dict[str, int]:
    """Применить готовый ``resolved``-план ON_LINE к ``status`` модели.

    Чистое применение (без формул): снимок контрактных массивов → ядро
    :func:`_apply_topology_on_arrays` → write-back. ``resolved`` — готовый план
    статусов ``(tag, parent_id, status|None, eval_skip|None)``.
    """
    arr_nodes = model.nodes.to_numpy().copy()
    arr_branches = model.branches.to_numpy().copy()
    arr_gens = model.generators.to_numpy().copy()
    stats = _apply_topology_on_arrays(arr_nodes, arr_branches, arr_gens, resolved)
    model.nodes.update_from_array(arr_nodes)
    model.branches.update_from_array(arr_branches)
    model.generators.update_from_array(arr_gens)
    return stats


def _apply_topology_on_arrays(
    arr_nodes: np.ndarray,
    arr_branches: np.ndarray,
    arr_gens: np.ndarray,
    resolved: Sequence[ResolvedItem],
) -> dict[str, int]:
    """ЯДРО: применение ON_LINE-статусов над контрактными массивами.

    Чистый статус-каскад: получает готовый план ``resolved`` (элемент
    ``(tag, parent_id, status|None, eval_skip|None)``), матчит ``tag`` → целевой
    массив (nodes/branches/generators), пишет ``status``-колонку. Мутирует массивы
    in place. Без внешних зависимостей, без float-арифметики.
    """
    by_id_nodes = id_to_pos_map(arr_nodes["id"])
    by_id_branches = id_to_pos_map(arr_branches["id"])
    by_id_gens = id_to_pos_map(arr_gens["id"])

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
        else:
            idx = None
            target = None
        if idx is None or target is None:
            stats["skipped_no_object"] += 1
            continue
        if status is None:
            # Контракт: при ``status is None`` producer всегда задаёт ключ eval_skip.
            assert eval_skip is not None
            stats[eval_skip] += 1
            continue

        target[idx]["status"] = status
        if status:
            stats["applied_on"] += 1
        else:
            stats["applied_off"] += 1

    return stats
