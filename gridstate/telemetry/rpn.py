"""РПН/ПБВ: применение №отпаек к tap_ratio/phase_shift/шунту (prep-адаптер + ядро).

float-ядро ``_apply_rpn_on_arrays`` + адаптер ``apply_rpn_resolved`` (применяет готовый
числовой план №отпаек ``resolved_taps`` к ветвям модели).
"""

from __future__ import annotations

import math


def _apply_rpn_on_arrays(
    branches_arr,
    shema_ktr_arr,
    resolved_taps: list[tuple[int, int, int, int]],
    *,
    skipped_no_branch: int = 0,
    skipped_no_tm: int = 0,
) -> dict[str, int | float]:
    """Применить разрешённые №отпаек к ``tap_ratio``/``phase_shift``/шунту (мутирует ``branches_arr``).

    PSC-free float-ядро Ф4.1 (CLASS-2): читает ТОЛЬКО контракт — ``branch.{id,tap_ratio,
    conductance*,susceptance*}`` + raw ``shema_ktr.{type_rpn,num_a,num_r,ktr_a,ktr_r,
    ktr_a_vc,ktr_r_vc}``; пишет ``branch.{tap_ratio,phase_shift,conductance*,susceptance*}``.
    Вся float-математика (ktr_lookup, main-vs-vc выбор по близости к xml_tap, hypot/atan2
    tap+phase, shunt-rescale (tap_new/tap_old)², diff-stats) — здесь.

    ``resolved_taps`` — список ``(branch_id, type_rpn, num_a, num_r)`` в порядке
    ``specs.items()`` (адаптер уже отфильтровал spec без ветви / без TM и передал их
    счётчики ``skipped_no_branch``/``skipped_no_tm`` для бит-в-бит сводки).

    **Должно оставаться последовательным циклом** в порядке ``resolved_taps``: выбор
    NUM_PBV — data-dependent argmin по xml_tap (тай-брейк строгий ``<``), main==vc —
    ``<=``; shunt-rescale только при ``|Δtap|>1e-12`` И поле ``!=0.0`` (точное сравнение).
    Округление hypot/atan2/деления допускает <1e-3 (open-q #3), но порядок сохраняем →
    фактически бит-в-бит.
    """
    sk = shema_ktr_arr
    ktr_lookup: dict[tuple[int, int, int], list[tuple[float, float, float, float]]] = {}
    if sk is not None and len(sk) > 0:
        for r in sk:
            key = (int(r["type_rpn"]), int(r["num_a"]), int(r["num_r"]))
            ktr_lookup.setdefault(key, []).append(
                (
                    float(r["ktr_a"]),
                    float(r["ktr_r"]),
                    float(r["ktr_a_vc"]),
                    float(r["ktr_r_vc"]),
                )
            )

    arr = branches_arr
    by_id: dict[int, int] = {int(arr[i]["id"]): i for i in range(len(arr))}

    stats: dict[str, int | float] = {
        "applied": 0,
        "skipped_no_tm": skipped_no_tm,
        "skipped_no_ktr": 0,
        "skipped_empty_ktr": 0,
        "skipped_no_branch": skipped_no_branch,
        "max_diff_pct": 0.0,
        "median_diff_pct": 0.0,
        "shunt_recalc": 0,
    }
    diffs: list[float] = []
    for branch_id, type_rpn, num_a, num_r in resolved_taps:
        idx = by_id[branch_id]
        candidates = ktr_lookup.get((type_rpn, num_a, num_r))
        if not candidates and num_r != 0:
            candidates = ktr_lookup.get((type_rpn, num_a, 0))
        if not candidates:
            stats["skipped_no_ktr"] += 1
            continue

        # Initial xml_tap (до изменения) — используется и для выбора
        # main vs vc на 3-обмоточном АТ, и для выбора NUM_PBV-позиции
        # среди нескольких строк SHEMA_KT с одинаковым (type, na, nr).
        xml_tap = float(arr[idx]["tap_ratio"])

        # Для каждой строки-кандидата вычисляем потенциальный
        # (use_a, use_r) с разрешением main vs vc по близости к xml_tap.
        # Затем выбираем ту строку, у которой результирующий tap
        # ближе всего к xml_tap — это даёт правильный NUM_PBV.
        best_use_a = 0.0
        best_use_r = 0.0
        best_diff = float("inf")
        any_nonempty = False
        for ktr_a, ktr_r, ktr_a_vc, ktr_r_vc in candidates:
            main_nz = abs(ktr_a) > 1e-12 or abs(ktr_r) > 1e-12
            vc_nz = abs(ktr_a_vc) > 1e-12 or abs(ktr_r_vc) > 1e-12
            if main_nz and vc_nz:
                tap_main = 1.0 / math.hypot(ktr_a, ktr_r)
                tap_vc = 1.0 / math.hypot(ktr_a_vc, ktr_r_vc)
                if abs(tap_main - xml_tap) <= abs(tap_vc - xml_tap):
                    use_a, use_r = ktr_a, ktr_r
                else:
                    use_a, use_r = ktr_a_vc, ktr_r_vc
            elif main_nz:
                use_a, use_r = ktr_a, ktr_r
            elif vc_nz:
                use_a, use_r = ktr_a_vc, ktr_r_vc
            else:
                continue
            any_nonempty = True
            cand_tap = 1.0 / math.hypot(use_a, use_r)
            diff = abs(cand_tap - xml_tap)
            if diff < best_diff:
                best_diff = diff
                best_use_a, best_use_r = use_a, use_r
        if not any_nonempty:
            # Все строки-кандидаты с пустыми парами — для этой ступени
            # РПН в БД нет коэффициента. Оставляем XML-tap (значение на
            # момент выгрузки) — лучшее приближение к актуальному.
            stats["skipped_empty_ktr"] += 1
            continue
        use_a, use_r = best_use_a, best_use_r

        # XmlFormat-конвенция: tap_ratio = 1 / hypot(ktr_a, ktr_r),
        # phase_shift = arctan2(-ktr_r, ktr_a). Re-применяем на новых
        # коэффициентах. См. xml_format.py:1414-1419.
        z2 = use_a * use_a + use_r * use_r
        re = use_a / z2
        im = -use_r / z2
        new_tap = float(math.hypot(re, im))
        new_phase = float(math.atan2(im, re))

        old_tap = float(arr[idx]["tap_ratio"])
        if old_tap > 1e-9:
            diffs.append(abs(new_tap - old_tap) / old_tap * 100.0)

        arr[idx]["tap_ratio"] = new_tap
        arr[idx]["phase_shift"] = new_phase
        stats["applied"] += 1

        # H30: пересчёт шунта намагничивания ХХ при изменении tap.
        #
        # Входной формат хранит ``conductance``/``susceptance`` трансформатора
        # как **приведённую** к стороне with-tap измеренную проводимость
        # ХХ; при изменении №отпайки физический шунт железа неизменен,
        # но приведённое значение должно масштабироваться как
        # ``B_new = B_old * (tap_new/tap_old)^2`` (то же для G).
        # Без пересчёта на 3-обм АТ с заметным шунтом ХХ это даёт
        # смещение потерь ~1 % от p_loss_total.
        # См. issue ``rpn_shunt_recalc`` + memory ``rpn_dynamic_tap_ratio``.
        if old_tap > 1e-9 and abs(new_tap - old_tap) > 1e-12:
            factor = (new_tap / old_tap) ** 2
            recalc_done = False
            for field in (
                "conductance",
                "susceptance",
                "conductance_from",
                "susceptance_from",
                "conductance_to",
                "susceptance_to",
            ):
                val = float(arr[idx][field])
                if val != 0.0:
                    arr[idx][field] = val * factor
                    recalc_done = True
            if recalc_done:
                stats["shunt_recalc"] += 1

    if diffs:
        stats["max_diff_pct"] = float(max(diffs))
        diffs_sorted = sorted(diffs)
        n = len(diffs_sorted)
        stats["median_diff_pct"] = float(
            diffs_sorted[n // 2]
            if n % 2 == 1
            else (diffs_sorted[n // 2 - 1] + diffs_sorted[n // 2]) / 2
        )

    return stats


def apply_rpn_resolved(
    model,
    resolved_taps: list[tuple[int, int, int, int]],
    *,
    skipped_no_branch: int = 0,
    skipped_no_tm: int = 0,
) -> dict[str, int | float]:
    """Фаза-A (релокация): применить готовый ``resolved_taps`` к ветвям модели.

    Чистое применение (без XML/snapshot/формул): снимок ``branches`` + ``shema_ktr`` →
    ядро :func:`_apply_rpn_on_arrays` → write-back. Зовётся шагом ``run()`` на своей
    позиции (до ``apply_telemetry``: tap влияет на variance/Y-bus).
    """
    branches_arr = model.branches.to_numpy().copy()
    stats = _apply_rpn_on_arrays(
        branches_arr,
        model.raw_tables.get("shema_ktr"),
        resolved_taps,
        skipped_no_branch=skipped_no_branch,
        skipped_no_tm=skipped_no_tm,
    )
    model.branches.update_from_array(branches_arr)
    return stats
