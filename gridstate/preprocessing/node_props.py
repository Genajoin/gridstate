"""NODE-properties для pseudo-measurements: чтение из ``model.nodes``.

`exist_load`, `exist_gen`, `vzd`, `pn_min/max` и т.п. — контрактные поля
``NODE_DTYPE`` (входные атрибуты ``EXIST_PN/QN/PG/QG``, ``U_ZAD``,
``PN_MIN/MAX`` и т.д.).

Используется ``add_pseudo_measurements`` для:

* классификации узла как **transit** vs **load/gen** (порог жёсткости ZIB);
* привязки V-прайора PV-узла к ``vzd`` (астровская семантика).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = [
    "extract_boundary_node_ids_from_model",
    "extract_node_load_props_from_model",
]


def extract_node_load_props_from_model(model: Working) -> dict[int, dict]:
    """``id → {exist_load, exist_gen, vzd, pn_min/max, qn_min/max,
    pg_min/max, umin, umax, na, tip}`` из ``model.nodes``.

    Поля живут прямо в ``NODE_DTYPE``: ``exist_load``, ``exist_gen``,
    ``load_p_min/max``, ``load_q_min/max``, ``generation_p_min/max``,
    ``voltage_setpoint``, ``voltage_min/max``, ``area_id``, ``node_type``.

    Имена ключей в возврате (``pn_min``, ``vzd``, ``na`` и т.п.) сохранены
    из астровской терминологии для ``add_pseudo_measurements``.
    """
    arr = model.nodes.to_numpy()
    out: dict[int, dict] = {}
    for row in arr:
        out[int(row["id"])] = {
            "exist_load": int(row["exist_load"]),
            "exist_gen": int(row["exist_gen"]),
            "pn_min": float(row["load_p_min"]),
            "pn_max": float(row["load_p_max"]),
            "qn_min": float(row["load_q_min"]),
            "qn_max": float(row["load_q_max"]),
            "pg_min": float(row["generation_p_min"]),
            "pg_max": float(row["generation_p_max"]),
            "vzd": float(row["voltage_setpoint"]),
            "umin": float(row["voltage_min"]),
            "umax": float(row["voltage_max"]),
            "na": int(row["area_id"]),
            "tip": int(row["node_type"]),
        }
    return out


def extract_boundary_node_ids_from_model(
    model: Working,
    *,
    boundary_area_ids: set[int] | None = None,
    pn_range_threshold_mw: float | None = None,
) -> set[int]:
    """Эвристическое определение boundary-узлов из ``model.nodes``.

    Поддерживает два признака:

    1. ``area_id ∈ boundary_area_ids`` — явный список граничных area-id;
    2. ``|load_p_max − load_p_min| ≥ pn_range_threshold_mw`` или
       ``|load_q_max − load_q_min| ≥ pn_range_threshold_mw`` — широкий
       physical-limit диапазон (≥1000 МВт типично) указывает на
       эквивалент с заданными границами обмена.

    Используется как helper для ``add_pseudo_measurements(..., boundary_node_ids=...)``
    при ослаблении σ² P_inj/Q_inj-pseudo-priors. Контекст —
    в ``docs/audit/audit_se_boundary_nodes.md``.
    """
    arr = model.nodes.to_numpy()
    out: set[int] = set()
    if boundary_area_ids:
        for nid in arr["id"][np.isin(arr["area_id"], list(boundary_area_ids))]:
            out.add(int(nid))
    if pn_range_threshold_mw is not None:
        rp = np.abs(arr["load_p_max"] - arr["load_p_min"])
        rq = np.abs(arr["load_q_max"] - arr["load_q_min"])
        for nid in arr["id"][(rp >= pn_range_threshold_mw) | (rq >= pn_range_threshold_mw)]:
            out.add(int(nid))
    return out
