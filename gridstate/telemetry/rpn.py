"""РПН/ПБВ: применение выбранных отпаек к tap_ratio/phase_shift/шунту.

Ядро ``_apply_tap_steps_on_arrays`` применяет готовую таблицу ``tap_steps`` к
ветвям модели через обёртку ``apply_rpn_resolved``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridstate.utils import id_to_pos_map


if TYPE_CHECKING:
    from gridstate.working import Working


def _apply_tap_steps_on_arrays(
    branches_arr: np.ndarray, tap_steps_arr: np.ndarray
) -> dict[str, int | float]:
    """Применить контракт ``tap_steps`` к ветвям (tap/phase + H30-шунт). Мутирует ``branches_arr``.

    Выбор отпайки сделан вне ядра — ``tap_steps`` несёт целевые ``tap_ratio``/
    ``phase_shift`` + ``shunt_factor`` (=(tap_new/tap_old)², 1.0=без пересчёта).
    Ядро лишь применяет: пишет tap/phase и H30-факторит шунт (поля ``!=0.0``).
    Это формат-агностичная физика трансформатора.
    """
    by_id = id_to_pos_map(branches_arr["id"])
    stats: dict[str, int | float] = {"applied": 0, "shunt_recalc": 0}
    for ts in tap_steps_arr:
        idx = by_id.get(int(ts["branch_id"]))
        if idx is None:
            continue
        branches_arr[idx]["tap_ratio"] = float(ts["tap_ratio"])
        branches_arr[idx]["phase_shift"] = float(ts["phase_shift"])
        stats["applied"] += 1

        factor = float(ts["shunt_factor"])
        if factor != 1.0:  # 1.0 — sentinel «без H30» (адаптер); умножение было бы no-op
            recalc_done = False
            for field in (
                "conductance",
                "susceptance",
                "conductance_from",
                "susceptance_from",
                "conductance_to",
                "susceptance_to",
            ):
                val = float(branches_arr[idx][field])
                if val != 0.0:
                    branches_arr[idx][field] = val * factor
                    recalc_done = True
            if recalc_done:
                stats["shunt_recalc"] += 1
    return stats


def apply_rpn_resolved(model: Working) -> dict[str, int | float]:
    """Применить выбранные отпайки (таблица ``tap_steps``) к ветвям модели.

    Выбор отпайки сделан выше по pipeline; ядро :func:`_apply_tap_steps_on_arrays`
    пишет tap/phase + H30-факторит шунт. Зовётся шагом ``run()`` до
    ``apply_telemetry`` (tap влияет на variance/Y-bus). Пустая ``tap_steps`` →
    no-op.
    """
    tap_steps_coll = getattr(model, "tap_steps", None)
    tap_steps_arr = tap_steps_coll.to_numpy() if tap_steps_coll is not None else None
    if tap_steps_arr is None or len(tap_steps_arr) == 0:
        return {"applied": 0, "shunt_recalc": 0}
    branches_arr = model.branches.to_numpy().copy()
    stats = _apply_tap_steps_on_arrays(branches_arr, tap_steps_arr)
    model.branches.update_from_array(branches_arr)
    return stats
