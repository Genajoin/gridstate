"""Априорная фильтрация грубых ошибок ТИ через анализ потерь по 5 формулам.

Основано на подходе эталонной SE (см. презентацию по работе с ТИ,
стр. 23): для каждой ветви считаем 5 независимых оценок потерь активной
мощности, отклонение между ними локализует ТИ-источник ошибки.

Формулы (для нашей конвенции «оба ТИ — поток ИЗ узла В линию»,
поэтому в F3, F4 берём СУММУ, а не разность):
* F1 = (P_нач² + Q_нач²) / U_нач² · R   — потери из ТИ начала
* F2 = (P_кон² + Q_кон²) / U_кон² · R   — потери из ТИ конца
* F3 = (Q_нач + Q_кон) / X · R           — потери через сумму Q
* F4 = P_нач + P_кон                      — прямые потери (сумма потоков)
* F5 = (U_нач² + U_кон² - 2·U_нач·U_кон·cos(δ)) / Z² · R — через ΔU/δ

В оригинальной презентации эталонной SE (PDF стр. 23) используется F4=P_нач-P_кон,
F3=(Q_нач-Q_кон)/X·R — это для конвенции входного формата, где P_кон — поток
ИЗ ЛИНИИ в узел конца. В нашем pipeline после `apply_telemetry` оба
P (нач и кон) — потоки В ЛИНИЮ из соответствующих узлов, поэтому P_кон
имеет противоположный знак относительно конвенции входного формата → меняем «-» на «+».

В идеале F1=F2=F3=F4=F5. Выбираем 2 наиболее отклоняющиеся формулы;
по таблице совпадения определяем виновное ТИ:

**Q-balance расчётный** (расширение, не из оригинальной эталонной SE):
:func:`compute_expected_q_imbalance_mvar` оценивает физически
ожидаемое ``|Q_beg + Q_end|`` для линии через ``V² · B`` (зарядная B)
и ``(P²+Q²)/V² · X`` (потери в X). Можно использовать как
**расчётный threshold** для ``q_inconsistency`` filter вместо
flat-default 50 МВар — на 750 кВ ВЛ с большой зарядной B
expected_charging достигает 1000+ МВар, и flat threshold их ложно
отбрасывает.

Пример для ВЛ 750 кВ:
* ``Vnom=750, |B|=0.001895 См → expected_charging = 750²·0.001895
  = 1066 МВар``;
* TM имеет |Q_beg + Q_end| ≈ 1100 МВар (close, в пределах charging);
* default threshold=50 ложно отбрасывает; charging-aware threshold
  ≥ 1.5·1066 = 1600 МВар — оставляет.

| (i, j)        | Виновное ТИ |
|---------------|-------------|
| (1, 4)        | P_нач       |
| (2, 4)        | P_кон       |
| (1, 3)        | Q_нач       |
| (2, 3)        | Q_кон       |
| (1, 5)        | U_нач       |
| (2, 5)        | U_кон       |

Действие на «виновный» ТИ: либо `deactivate` (status=False), либо
`downweight` (variance × `downweight_factor`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gridstate.constants import (
    MeasurementObjectType,
    MeasurementType,
)


if TYPE_CHECKING:
    from gridstate.working import Working


# Соответствие пары (i, j) → виновное ТИ. Ключи (i<j).
_BLAME_TABLE: dict[tuple[int, int], str] = {
    (1, 4): "PBEG",
    (2, 4): "PEND",
    (1, 3): "QBEG",
    (2, 3): "QEND",
    (1, 5): "VBEG",
    (2, 5): "VEND",
}


@dataclass
class BranchLossReport:
    branch_id: int
    f_values: dict[int, float]  # {1: F1, 2: F2, ...}
    delta_max: float  # max |Fi - Fj|
    delta_pct: float  # delta_max / max(|F1..F5|) * 100
    blamed_meas: str | None  # PBEG/PEND/QBEG/QEND/VBEG/VEND или None
    pair: tuple[int, int] | None


def compute_branch_loss_formulas(
    *,
    p_beg: float | None,
    p_end: float | None,
    q_beg: float | None,
    q_end: float | None,
    v_beg: float | None,
    v_end: float | None,
    delta_rad: float = 0.0,
    r: float,
    x: float,
    z2: float,
) -> dict[int, float]:
    """Считает F1..F5 на основании имеющихся значений (МВт).

    Возвращает dict только тех формул, которые удалось посчитать
    (требуют валидных значений соответствующих ТИ и ненулевых R/X/Z²).
    """
    out: dict[int, float] = {}
    if r is None or r <= 0:
        return out  # без R потери не оценить (для нулевого R fallback не делаем)

    if v_beg is not None and v_beg > 0 and p_beg is not None and q_beg is not None:
        out[1] = (p_beg**2 + q_beg**2) / (v_beg**2) * r

    if v_end is not None and v_end > 0 and p_end is not None and q_end is not None:
        out[2] = (p_end**2 + q_end**2) / (v_end**2) * r

    if x is not None and x != 0 and q_beg is not None and q_end is not None:
        out[3] = (q_beg + q_end) / x * r

    if p_beg is not None and p_end is not None:
        out[4] = p_beg + p_end

    if v_beg is not None and v_end is not None and v_beg > 0 and v_end > 0 and z2 and z2 > 0:
        out[5] = (v_beg**2 + v_end**2 - 2 * v_beg * v_end * math.cos(delta_rad)) / z2 * r
    return out


def compute_expected_q_imbalance_mvar(
    *,
    vn_kv: float,
    susceptance_si: float,
    reactance_si: float,
    p_typical_mw: float = 0.0,
    q_typical_mvar: float = 0.0,
) -> float:
    """Расчётное ``|Q_beg + Q_end|`` для линии через физику π-схемы.

    На активной ВЛ (без трафо) сумма Q-потоков на обоих концах =
    Q-потери в реактивности минус Q-генерация зарядной B::

        Q_beg + Q_end = (P²+Q²)/V²·X  -  V²·B

    где B — полная зарядная (хранится в нашей storage как
    ``branch.susceptance``, в Сименсах) — делится пополам в шунты
    обоих концов π-схемы (V²·B/2 + V²·B/2 = V²·B).

    Для приблизительной оценки используем ``Vnom`` вместо реального V
    (на ВЛ V обычно близок к номиналу ±5 %). Также можно опустить
    Q-потери в X если ``p_typical/q_typical = 0`` — на 750 кВ ВЛ они
    обычно << зарядной мощности.

    Args:
        vn_kv: номинальное напряжение ветви, кВ (берём из узла начала).
        susceptance_si: ``branch.susceptance``, См (полная B линии,
            хранится со знаком: +у capacitive ВЛ, ‒у indictive ШР).
        reactance_si: ``branch.reactance``, Ом.
        p_typical_mw: типичный P-поток через ветвь, МВт (если знаем —
            добавим Q-потери X-серии; default 0 = только charging).
        q_typical_mvar: типичный Q-поток.

    Returns:
        |Σ Q_beg + Q_end| в МВар (положительное число).

    Notes:
        Единицы: V [кВ], B [См] → V²·B имеет размерность (кВ)² · См =
        (1000 В)² · См = 10⁶ ВА = МВА. Поэтому V²·B напрямую в МВар.
        Аналогично (P²+Q²)/V²·X = (МВА)²/(кВ)² · Ом = МА² · Ом = МВт²/В.
        Конвертация: 1 МА = 10⁶ А; 1 МА² · Ом = 10¹² ВА = 10⁶ МВар.
        Корректировка: (P²+Q²)/V²·X где P,Q в МВт/МВар, V в кВ →
        результат в МВар напрямую (через единицы входного формата S = VI·1e-3).
    """
    charging_mvar = vn_kv * vn_kv * abs(susceptance_si)
    if vn_kv > 0 and abs(reactance_si) > 1e-12:
        s2 = p_typical_mw * p_typical_mw + q_typical_mvar * q_typical_mvar
        x_loss_mvar = s2 / (vn_kv * vn_kv) * abs(reactance_si)
    else:
        x_loss_mvar = 0.0
    return abs(charging_mvar - x_loss_mvar)


def is_branch_q_consistent_with_physics(
    *,
    q_observed_mvar: float,
    vn_kv: float,
    susceptance_si: float,
    reactance_si: float,
    p_typical_mw: float = 0.0,
    floor_mvar: float = 50.0,
    rel_pct: float = 30.0,
) -> tuple[bool, float, float, float]:
    """Sanity-check: совпадает ли |Q_beg+Q_end| с физикой π-схемы.

    Расширение flat-threshold ``|Q_beg+Q_end|>const``: использует
    расчётное ожидание через зарядную B и потери в X (V_flat=V_nom).

    Outlier-логика: ``excess = ||Q_obs| - |Q_expected||``; tolerance
    ``tol = max(floor_mvar, rel_pct/100 · |Q_expected|)``. Outlier если
    ``excess > tol``.

    Use case на 750 кВ ВЛ с большой зарядной B:
    ``|Q_obs|≈1100, |Q_exp|=V²·B=1066`` → excess=34 < tol≈max(50, 320)
    → **keep** (flat ``q>50 МВар`` ложно дропал).

    Use case на 220 кВ ВЛ с малой B:
    ``|Q_obs|=80, |Q_exp|=5`` → excess=75 > tol=max(50, 1.5) → **drop**
    (flat ``q>100 МВар`` пропускал).

    Returns:
        (is_consistent, expected_mvar, observed_mvar, excess_mvar)
    """
    q_obs_abs = abs(q_observed_mvar)
    q_exp_abs = compute_expected_q_imbalance_mvar(
        vn_kv=vn_kv,
        susceptance_si=susceptance_si,
        reactance_si=reactance_si,
        p_typical_mw=p_typical_mw,
        q_typical_mvar=q_observed_mvar / 2.0,
    )
    excess = abs(q_obs_abs - q_exp_abs)
    tol = max(floor_mvar, rel_pct / 100.0 * q_exp_abs)
    return (excess <= tol, q_exp_abs, q_obs_abs, excess)


def _identify_outlier_pair(
    f_values: dict[int, float],
    *,
    sens_err_mw: float,
    sens_err_pct: float,
) -> tuple[tuple[int, int] | None, float, float]:
    """Найти пару максимально расходящихся формул.

    Outlier если |F_i - F_j| > `sens_err_mw` (абсолютный порог в МВт)
    И delta > `sens_err_pct/100` · min(|F_i|, |F_j|) (относительный к
    меньшему). Семантика эталонной SE: 25 МВт абсолютно или 150 % от меньшего.
    """
    if len(f_values) < 2:
        return None, 0.0, 0.0
    keys = sorted(f_values.keys())
    best = (0, 0, 0.0, 0.0)  # (i, j, delta, min_abs)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = keys[a], keys[b]
            d = abs(f_values[i] - f_values[j])
            if d > best[2]:
                best = (i, j, d, min(abs(f_values[i]), abs(f_values[j])))
    i, j, delta, min_abs = best
    scale = max(abs(v) for v in f_values.values())
    delta_pct = (delta / scale * 100.0) if scale > 0 else 0.0
    rel_threshold = sens_err_pct / 100.0 * min_abs
    if delta < sens_err_mw and delta < rel_threshold:
        return None, delta, delta_pct
    return (i, j), delta, delta_pct


def analyze_branch_loss_consistency(
    model: Working,
    *,
    sens_err_mw: float = 25.0,
    sens_err_pct: float = 150.0,
    use_estimated_voltage: bool = False,
    action: str = "downweight",
    downweight_factor: float = 100.0,
) -> dict:
    """Применить F1-F5 фильтр к ветвям модели.

    Args:
        model: Working с уже применённой телеметрией.
        sens_err_mw: абсолютный порог расхождения, МВт (default эталонной SE 25).
        sens_err_pct: относительный порог в % от max(|F|) (default эталонной SE 150).
        use_estimated_voltage: если True, использовать `node.voltage_magnitude`
            (заполнено estimate-ом или начальной оценкой); иначе ищем
            активные voltage-measurements на узле.
        action: `"downweight"` × variance, либо `"deactivate"` (status=False).
        downweight_factor: множитель variance при `action="downweight"`.

    Returns:
        dict со статистикой и списком отчётов::

            {
                "total": N,
                "checked": N,
                "outliers": N,
                "blame": {"PBEG": N, "PEND": N, ...},
                "actions_taken": N,
                "reports": [BranchLossReport, ...],
            }
    """
    if action not in ("downweight", "deactivate"):
        raise ValueError(f"action must be 'downweight' or 'deactivate', got {action!r}")

    nn = model.nodes.to_numpy()
    bn = model.branches.to_numpy()
    arr = model.measurements.to_numpy().copy()

    # Index measurements: (object_type, measurement_type, branch_side, object_id) → row idx
    meas_idx: dict[tuple[int, int, int, int], int] = {}
    for i, r in enumerate(arr):
        if not bool(r["status"]):
            continue
        key = (
            int(r["object_type"]),
            int(r["measurement_type"]),
            int(r["branch_side"]),
            int(r["object_id"]),
        )
        meas_idx.setdefault(key, i)

    # node_id → V (либо из measurement, либо из node.voltage_magnitude)
    node_v: dict[int, float] = {}
    for n in nn:
        nid = int(n["id"])
        if not bool(n["status"]):
            continue
        # Active voltage measurement?
        ki = meas_idx.get((int(MeasurementObjectType.NODE), int(MeasurementType.VOLTAGE), -1, nid))
        if ki is not None:
            node_v[nid] = float(arr[ki]["value"])
        elif use_estimated_voltage:
            v = float(n["voltage_magnitude"])
            if v > 0:
                node_v[nid] = v
            else:
                vn = float(n["voltage_nominal"])
                if vn > 0:
                    node_v[nid] = vn  # фоллбек на номинал
        else:
            # без use_estimated_voltage — узлы без TM-V не получают F1/F2/F5
            pass

    OT_BR = int(MeasurementObjectType.BRANCH)
    MT_P = int(MeasurementType.POWER_P)
    MT_Q = int(MeasurementType.POWER_Q)

    blame_counts = dict.fromkeys(("PBEG", "PEND", "QBEG", "QEND", "VBEG", "VEND"), 0)
    reports: list[BranchLossReport] = []
    actions_taken = 0
    checked = 0
    outliers = 0

    for b in bn:
        if not bool(b["status"]):
            continue
        bid = int(b["id"])
        from_id = int(b["from_node"])
        to_id = int(b["to_node"])
        # Трансформаторы пропускаем — F1-F5 валидно только для линий
        # (V_нач/V_кон одного класса). Трансформатор: tap_ratio != 1.0.
        tap = float(b["tap_ratio"]) if "tap_ratio" in b.dtype.names else 1.0
        if tap != 0 and abs(tap - 1.0) > 0.001:
            continue
        # R, X в Омах (BRANCH_DTYPE: resistance/reactance в Ом).
        # Формула в СИ: ΔP[МВт] = (P[МВт]² + Q[МВар]²)/U[кВ]² · R[Ом].
        r_si = float(b["resistance"])
        x_si = float(b["reactance"])
        if r_si <= 0 and x_si == 0:
            continue
        z2_si = r_si**2 + x_si**2

        # ТИ значения ветви
        def _val(side: int, mt: int, bid: int = bid) -> float | None:
            i = meas_idx.get((OT_BR, mt, side, bid))
            return float(arr[i]["value"]) if i is not None else None

        p_beg = _val(0, MT_P)
        p_end = _val(1, MT_P)
        q_beg = _val(0, MT_Q)
        q_end = _val(1, MT_Q)
        v_beg = node_v.get(from_id)
        v_end = node_v.get(to_id)

        f_values = compute_branch_loss_formulas(
            p_beg=p_beg,
            p_end=p_end,
            q_beg=q_beg,
            q_end=q_end,
            v_beg=v_beg,
            v_end=v_end,
            r=r_si,
            x=x_si,
            z2=z2_si,
        )
        if len(f_values) < 2:
            continue
        checked += 1

        pair, delta_max, delta_pct = _identify_outlier_pair(
            f_values,
            sens_err_mw=sens_err_mw,
            sens_err_pct=sens_err_pct,
        )
        blamed: str | None = None
        if pair is not None:
            outliers += 1
            blamed = _BLAME_TABLE.get(pair)

        reports.append(
            BranchLossReport(
                branch_id=bid,
                f_values=dict(f_values),
                delta_max=delta_max,
                delta_pct=delta_pct,
                blamed_meas=blamed,
                pair=pair,
            )
        )

        if blamed is None:
            continue
        blame_counts[blamed] += 1

        # Локализуем индекс виновного ТИ
        if blamed in ("PBEG", "PEND"):
            side = 0 if blamed == "PBEG" else 1
            ki = meas_idx.get((OT_BR, MT_P, side, bid))
        elif blamed in ("QBEG", "QEND"):
            side = 0 if blamed == "QBEG" else 1
            ki = meas_idx.get((OT_BR, MT_Q, side, bid))
        else:  # VBEG/VEND
            target_id = from_id if blamed == "VBEG" else to_id
            ki = meas_idx.get(
                (int(MeasurementObjectType.NODE), int(MeasurementType.VOLTAGE), -1, target_id)
            )
        if ki is None:
            continue

        if action == "deactivate":
            arr[ki]["status"] = False
            field_names = arr.dtype.names
            arr[ki]["filter_flag"] = (
                3
                if field_names is not None and "filter_flag" in field_names
                else arr[ki]["filter_flag"]
            )
        else:
            new_var = float(arr[ki]["variance"]) * downweight_factor
            arr[ki]["variance"] = new_var
            arr[ki]["weight"] = 1.0 / new_var if new_var > 0 else 0.0
        actions_taken += 1

    # Записать обратно
    model.measurements.update_from_array(arr)

    return {
        "total": len(bn),
        "checked": checked,
        "outliers": outliers,
        "blame": blame_counts,
        "actions_taken": actions_taken,
        "reports": reports,
    }
