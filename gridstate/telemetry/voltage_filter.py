"""Фильтр V-измерений: широкий диапазон [V_lo, V_hi] с запасом.

Implements the standard plausibility check of measurements against the
admissible and nominal values. The goal is to reject only readings that are
certainly broken (V<<V_critical, V>>V_max) without losing valid edge values.

Нижний порог (приоритет — наиболее мягкий):
* `voltage_critical`, the emergency lower limit; the first constraint,
  usually supplied for all nodes;
* fallback `V_ном · (1 − fallback_pct/100)` (default 20 %).

Верхний порог:
* `voltage_max · (1 + upper_margin_pct/100)` — voltage_max само по себе
  «нормальный режим», нужен запас сверху для edge-значений;
* fallback `V_ном · (1 + fallback_pct/100)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from gridstate.bounds import is_sentinel
from gridstate.constants import (
    MeasurementObjectType,
    MeasurementQuality,
    MeasurementType,
    NodeType,
)
from gridstate.telemetry._filters import downweight_measurement, validate_downweight_action
from gridstate.utils import id_to_pos_map
from gridstate.working import table_copy, table_replace


if TYPE_CHECKING:
    from gridstate.working import Working


def apply_voltage_range_filter(
    model: Working,
    *,
    upper_margin_pct: float = 10.0,
    min_voltage_nominal_kv: float = 110.0,
    upper_fallback_factor: float = 1.4,
    action: str = "downweight",
    deactivate_nonphysical: bool = False,
    nonphysical_lower_factor: float = 0.5,
    nonphysical_upper_factor: float = 1.5,
    questionable_sigma2_multiplier: float = 100.0,
) -> dict:
    """Деактивирует/понижает вес V-измерений вне допустимого диапазона.

    The fallbacks handle placeholder limits that real inputs contain, e.g. a
    critical limit of 1.0 on 500 kV nodes, or a missing upper limit on a large
    share of high-voltage nodes:

    * Нижняя граница: ``voltage_critical`` если он ≥ ``V_ном/2``
      (валидное значение). Иначе — ``voltage_min`` если он ≥ ``V_ном/2``.
      Otherwise ``V_ном/2`` (so that a placeholder critical limit of 1.0 does
      not let any V≥1 through).
    * Верхняя граница: ``voltage_max · (1 + upper_margin_pct/100)``;
      если ``voltage_max == 0`` — fallback ``V_ном · upper_fallback_factor``.

    Args:
        model: Working с применённой телеметрией.
        upper_margin_pct: запас сверху над voltage_max (default 10 %).
            voltage_max — «допустимое в норм. режиме», для фильтра
            битых ТИ берём `voltage_max · 1.10`. На узле Vn=10.5,
            voltage_max=11.6 (=1.1·Vn) это даёт верхнюю границу 12.76.
        upper_fallback_factor: множитель V_ном при `voltage_max=0`
            (default 1.4 = +40 %). Many inputs lack the upper limit on a
            large share of nodes ≥500 kV; a wide bound is used then.
        min_voltage_nominal_kv: минимальный класс напряжения для
            проверки. Default 110 кВ — игнорируем генераторные шины
            (6-21 кВ) и низковольтные нагрузки. На них V_TI обычно
            на edge диапазона (voltage_max), а через трансформаторы
            эти ТИ — важные якоря для 110-750 кВ части. Установка
            в 0 проверяет все узлы.
        action: что делать с out-of-range measurements:
            ``"downweight"`` (default) — увеличить variance × multiplier
            (мягко, WLS сам решит); ``"deactivate"`` — `status=False`
            (жёстко, может ухудшить если V-меры — единственный якорь).
        deactivate_nonphysical: если True, физически невозможные как
            напряжение V-меры (``V<=0``, ``V < V_ном·nonphysical_lower_factor``
            или ``V > V_ном·nonphysical_upper_factor``) **деактивируются**
            (``status=False``) вне зависимости от ``action``. Мотив:
            ``downweight`` (σ×10) не гасит грубо-нефизичные значения — битый
            U-датчик с ``V=2533`` кВ на шине 20 кВ (pu=127), отрицательный
            ``V`` или ``V=0.2`` pu (глубокий коллапс/битый датчик) сохраняет
            гигантский взвешенный остаток и рушит solve (J~1e8, коллапс узлов).
            Мягкий out-of-range в [lo, hi] по-прежнему обрабатывается
            ``action``. **Default False** (сохраняет прежнее поведение бит-в-бит).
        nonphysical_lower_factor: нижний физический предел V как множитель
            V_ном (default 0.5 = 50 % V_ном, совпадает с soft-floor ``half_nom``).
            V ниже этого на запитанной шине не может быть валидным напряжением
            (симметрично верхнему пределу). ``V<=0`` деактивируется всегда,
            вне зависимости от значения фактора.
        nonphysical_upper_factor: верхний физический предел V как множитель
            V_ном для ``deactivate_nonphysical`` (default 1.5 = 150 % V_ном).
            V выше этого не может быть валидным напряжением.
        questionable_sigma2_multiplier: множитель variance для
            QUESTIONABLE measurements (default 100.0 = σ × 10).

    Returns:
        ``{"checked": N, "out_of_range": N, "hard_deactivated": N,
        "by_vnom": {kv: N}}``
    """
    # Енумы object/measurement-type резолвятся здесь (gridstate.constants),
    # ядро принимает готовые int и читает только контрактные колонки.
    validate_downweight_action(action)
    meas_arr = table_copy(model.measurements)
    nodes_arr = model.nodes.to_numpy()
    stats = _voltage_range_filter_on_arrays(
        meas_arr,
        nodes_arr,
        ot_node=int(MeasurementObjectType.NODE),
        mt_v=int(MeasurementType.VOLTAGE),
        upper_margin_pct=upper_margin_pct,
        min_voltage_nominal_kv=min_voltage_nominal_kv,
        upper_fallback_factor=upper_fallback_factor,
        action=action,
        deactivate_nonphysical=deactivate_nonphysical,
        nonphysical_lower_factor=nonphysical_lower_factor,
        nonphysical_upper_factor=nonphysical_upper_factor,
        questionable_sigma2_multiplier=questionable_sigma2_multiplier,
    )
    table_replace(model.measurements, meas_arr)
    return stats


def _voltage_range_filter_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    *,
    ot_node: int,
    mt_v: int,
    upper_margin_pct: float,
    min_voltage_nominal_kv: float,
    upper_fallback_factor: float,
    action: str,
    questionable_sigma2_multiplier: float,
    deactivate_nonphysical: bool = False,
    nonphysical_lower_factor: float = 0.5,
    nonphysical_upper_factor: float = 1.5,
) -> dict:
    """ЯДРО: V-range filter над контрактными массивами.

    Mutates ``meas_arr`` in place (``status``/``variance``/``weight``/``quality``)
    and reads ``nodes_arr`` (``voltage_nominal``/``voltage_critical``/``voltage_min``/
    ``voltage_max``). The object and measurement type enums arrive as ready ints
    from the adapter. No external dependencies; a sequential loop in input order.
    """
    node_by_id = id_to_pos_map(nodes_arr["id"])

    counters: dict[str, int] = {
        "checked": 0,
        "out_of_range": 0,
        "hard_deactivated": 0,
    }
    by_vnom: dict[int, int] = {}

    for i, m in enumerate(meas_arr):
        if not bool(m["status"]):
            continue
        if int(m["object_type"]) != ot_node or int(m["measurement_type"]) != mt_v:
            continue
        node_id = int(m["object_id"])
        ni = node_by_id.get(node_id)
        if ni is None:
            continue
        v_nom = float(nodes_arr[ni]["voltage_nominal"])
        if v_nom <= 0 or v_nom < min_voltage_nominal_kv:
            continue
        v_meas = float(m["value"])

        # Жёсткая деактивация ФИЗИЧЕСКИ НЕВОЗМОЖНЫХ V (nonphysical): V≤0,
        # V < V_ном·lower_factor или V > V_ном·upper_factor. downweight (σ×10)
        # их не гасит — битый U-датчик с pu=127 («2533 кВ» на шине 20 кВ),
        # отрицательный V или V=0.2 pu (глубокий коллапс) сохраняет гигантский
        # взвешенный остаток и рушит solve (J~1e8, коллапс). Такие значения не
        # могут быть напряжением → status=False, а не downweight. Пределы
        # симметричны (снизу и сверху), V≤0 деактивируется всегда.
        if deactivate_nonphysical and (
            v_meas <= 0
            or v_meas < v_nom * nonphysical_lower_factor
            or v_meas > v_nom * nonphysical_upper_factor
        ):
            counters["out_of_range"] += 1
            counters["hard_deactivated"] += 1
            v_nom_int = round(v_nom)
            by_vnom[v_nom_int] = by_vnom.get(v_nom_int, 0) + 1
            meas_arr[i]["status"] = False
            continue

        if v_meas <= 0:
            # V≤0 на active-узле — деактивируем как заведомо невалидное.
            # apply_telemetry уже фильтрует V<50%Vn для TM-формул, но
            # measurement мог попасть из внешнего источника (manual seed,
            # SQL-импорт, повторное apply); этот guard закрывает дыру.
            counters["out_of_range"] += 1
            v_nom_int = round(v_nom)
            by_vnom[v_nom_int] = by_vnom.get(v_nom_int, 0) + 1
            if action == "deactivate":
                meas_arr[i]["status"] = False
            else:
                downweight_measurement(
                    meas_arr, i, questionable_sigma2_multiplier, mark_questionable=True
                )
            continue

        # Нижний порог: voltage_critical → voltage_min → V_ном/2.
        # Some inputs carry a placeholder critical limit of 1.0 on nodes
        # ≥500 kV; the validity criterion "X ≥ V_nom/2" discards 1.0 as a
        # placeholder (1 ≪ 250 for 500 kV).
        half_nom = v_nom * 0.5
        v_crit = float(nodes_arr[ni]["voltage_critical"])
        v_min = (
            float(nodes_arr[ni]["voltage_min"])
            if "voltage_min" in (nodes_arr.dtype.names or ())
            else 0.0
        )
        if v_crit >= half_nom:
            lo = v_crit
        elif v_min >= half_nom:
            lo = v_min
        else:
            lo = half_nom

        # Верхний порог: voltage_max·(1+upper_margin_pct) → fallback V_ном·factor.
        # Many inputs lack the upper limit on a large share of nodes ≥500 kV;
        # fallback 1.4·V_nom.
        v_max = float(nodes_arr[ni]["voltage_max"])
        hi = (
            v_max * (1.0 + upper_margin_pct / 100.0) if v_max > 0 else v_nom * upper_fallback_factor
        )

        counters["checked"] += 1
        v_nom_int = round(v_nom)
        by_vnom[v_nom_int] = by_vnom.get(v_nom_int, 0) + 1

        if v_meas < lo or v_meas > hi:
            counters["out_of_range"] += 1
            if action == "deactivate":
                meas_arr[i]["status"] = False
            else:  # downweight
                downweight_measurement(
                    meas_arr, i, questionable_sigma2_multiplier, mark_questionable=True
                )
            continue

    return {
        "checked": counters["checked"],
        "out_of_range": counters["out_of_range"],
        "hard_deactivated": counters["hard_deactivated"],
        "by_vnom": by_vnom,
    }


def apply_voltage_meas_calibration_for_gen_nodes(
    model: Working,
    *,
    sigma2: float = 0.1,
    mode: Literal["absolute", "tighten"] = "absolute",
) -> dict:
    """Calibrate σ² of V measurements on generating and slack nodes.

    Voltage transducers on generator buses (110-750 kV) are far more accurate
    than on passive load buses (σ_V of tenths of a kV against several kV).
    Without this step the default variance leaves the V measurement a weak
    anchor; on 750 kV clusters of nuclear plants that gave a systematic
    voltage sag of a few percent.

    Targets:

    * active nodes with ``generation_p_max`` set after
      :func:`aggregate_generators_to_node`. The ±9999 sentinel means "no
      data" (:func:`gridstate.bounds.is_sentinel`), not generation, so such
      nodes are skipped;
    * slack nodes (``node_type == NodeType.SLACK``).

    Only measurements of ``quality == GOOD`` are touched. A measurement an
    earlier stage marked QUESTIONABLE (telemetry quality, the voltage range
    filter) keeps its inflated variance: calibrating it would turn a suspect
    reading into the strongest anchor of the estimate.

    Modes:

    * ``"absolute"`` (default): the variance is **set** to ``sigma2``
      regardless of its current value, and the weight to ``1 / sigma2``. This
      is a tightening only where the current variance is larger than
      ``sigma2``. On low-voltage generator buses (6-16 kV) the telemetry
      variance is already smaller (``max(1 % · Vn, 0.1)² ≈ 0.01-0.025``), so
      ``"absolute"`` *loosens* those measurements, by up to a factor of 10,
      and it discards any variance scaling applied by earlier stages.
    * ``"tighten"``: the variance becomes ``min(current, sigma2)``, i.e. it is
      never loosened and a measurement already tighter than ``sigma2`` is left
      untouched (variance and weight). The weight of a tightened measurement
      is ``1 / sigma2``.

    Args:
        model: ``Working`` after aggregate_generators_to_node
            and apply_voltage_range_filter.
        sigma2: target σ² of the V measurement, in the units of the model's
            V measurements. Default 0.1 (σ ≈ 0.32 kV).
        mode: ``"absolute"`` or ``"tighten"``, see above.

    Returns:
        ``{"updated_meas": N, "target_nodes": N}``; in ``"tighten"`` mode also
        ``"kept_tighter": N`` -- targeted GOOD measurements whose variance was
        already at or below ``sigma2`` and was left as is.

    Raises:
        ValueError: if ``mode`` is not one of the two listed.
    """
    if mode not in ("absolute", "tighten"):
        raise ValueError(f"mode must be 'absolute' or 'tighten', got {mode!r}")
    meas_arr = table_copy(model.measurements)
    nodes_arr = model.nodes.to_numpy()
    stats = _voltage_meas_calibration_on_arrays(
        meas_arr,
        nodes_arr,
        slack_type=int(NodeType.SLACK),
        ot_node=int(MeasurementObjectType.NODE),
        mt_v=int(MeasurementType.VOLTAGE),
        sigma2=sigma2,
        mode=mode,
    )
    table_replace(model.measurements, meas_arr)
    return stats


def _voltage_meas_calibration_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    *,
    slack_type: int,
    ot_node: int,
    mt_v: int,
    sigma2: float,
    mode: Literal["absolute", "tighten"] = "absolute",
) -> dict:
    """Core of :func:`apply_voltage_meas_calibration_for_gen_nodes` on arrays.

    Targets are active nodes with a non-zero, non-sentinel
    ``generation_p_max`` or ``node_type == SLACK`` (enum values as ints).
    Mutates ``variance``/``weight`` of GOOD measurements in ``meas_arr`` in
    place; reads ``nodes_arr``. ``mode`` is validated by the public wrapper.
    """
    has_gen_max = "generation_p_max" in (nodes_arr.dtype.names or ())
    target_ids: set[int] = set()
    for n in nodes_arr:
        if not bool(n["status"]):
            continue
        if int(n["node_type"]) == slack_type:
            target_ids.add(int(n["id"]))
            continue
        if has_gen_max:
            p_max = float(n["generation_p_max"])
            if p_max != 0.0 and not is_sentinel(p_max):
                target_ids.add(int(n["id"]))

    tighten = mode == "tighten"
    n_updated = 0
    n_kept = 0
    for i in range(len(meas_arr)):
        if not bool(meas_arr[i]["status"]):
            continue
        if int(meas_arr[i]["object_type"]) != ot_node:
            continue
        if int(meas_arr[i]["measurement_type"]) != mt_v:
            continue
        if int(meas_arr[i]["object_id"]) not in target_ids:
            continue
        if int(meas_arr[i]["quality"]) != int(MeasurementQuality.GOOD):
            continue
        if tighten:
            current = float(meas_arr[i]["variance"])
            if np.isfinite(current) and 0.0 < current <= sigma2:
                n_kept += 1
                continue
        meas_arr[i]["variance"] = float(sigma2)
        meas_arr[i]["weight"] = 1.0 / float(sigma2) if sigma2 > 0 else 0.0
        n_updated += 1

    stats = {"updated_meas": n_updated, "target_nodes": len(target_ids)}
    if tighten:
        stats["kept_tighter"] = n_kept
    return stats
