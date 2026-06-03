"""Фильтр V-измерений: широкий диапазон [V_lo, V_hi] с запасом.

Реализует стр. 21 пункт 1 презентации эталонной SE «Контроль допустимых и
номинальных значений измерений». Цель — отбраковка только гарантированно
битых ТИ (V<<V_critical, V>>V_max), без потери валидных edge-значений.

Нижний порог (приоритет — наиболее мягкий):
* `voltage_critical` (U_CRIT) — аварийный нижний предел; первое
  ограничение, как правило заполнено на 100 % узлов в исходном XML;
* fallback `V_ном · (1 − fallback_pct/100)` (default 20 %).

Верхний порог:
* `voltage_max · (1 + upper_margin_pct/100)` — voltage_max само по себе
  «нормальный режим», нужен запас сверху для edge-значений;
* fallback `V_ном · (1 + fallback_pct/100)`.
"""

from __future__ import annotations

from gridstate.constants import (
    MeasurementObjectType,
    MeasurementType,
)


def apply_voltage_range_filter(
    model,
    *,
    upper_margin_pct: float = 10.0,
    min_voltage_nominal_kv: float = 110.0,
    upper_fallback_factor: float = 1.4,
    action: str = "downweight",
    detect_nominal_substitution: bool = False,
    nominal_substitution_eps: float = 0.001,
    questionable_sigma2_multiplier: float = 100.0,
) -> dict:
    """Деактивирует/понижает вес V-измерений вне допустимого диапазона.

    Логика fallback'ов настроена под исходный XML, где встречаются
    «заглушки» — например `U_KRIT=1.0` на 500-кВ узлах, `U_MAX`
    отсутствует на большой доле высоковольтных узлов:

    * Нижняя граница: ``voltage_critical`` если он ≥ ``V_ном/2``
      (валидное значение). Иначе — ``voltage_min`` если он ≥ ``V_ном/2``.
      Иначе — ``V_ном/2`` (чтобы не пропустить любое V≥1 при заглушке
      `U_KRIT=1.0`).
    * Верхняя граница: ``voltage_max · (1 + upper_margin_pct/100)``;
      если ``voltage_max == 0`` — fallback ``V_ном · upper_fallback_factor``.

    Args:
        model: Working с применённой телеметрией.
        upper_margin_pct: запас сверху над voltage_max (default 10 %).
            voltage_max — «допустимое в норм. режиме», для фильтра
            битых ТИ берём `voltage_max · 1.10`. На узле Vn=10.5,
            voltage_max=11.6 (=1.1·Vn) это даёт верхнюю границу 12.76.
        upper_fallback_factor: множитель V_ном при `voltage_max=0`
            (default 1.4 = +40 %). На многих XML-выгрузках атрибут
            `U_MAX` отсутствует у большой доли узлов ≥500 кВ; при
            отсутствии берём широкую границу.
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
        detect_nominal_substitution: если True, V-измерения с
            ``|V_TI - V_ном| / V_ном < nominal_substitution_eps``
            помечаются как QUESTIONABLE (вероятная подстановка
            номинала вместо реального ТИ). **Default False** —
            на типичных XML-выгрузках многие реальные V-ТИ близки
            к V_ном (генераторные шины, режим без перетоков), поэтому
            детект даёт false-positives. Включать с осторожностью
            и низким eps (≤ 0.001 = 0.1 %).
        nominal_substitution_eps: порог для детекта подстановки
            (default 0.001 = 0.1 %).
        questionable_sigma2_multiplier: множитель variance для
            QUESTIONABLE measurements (default 100.0 = σ × 10).

    Returns:
        ``{"checked": N, "out_of_range": N,
        "downweighted_nominal_substitution": N, "by_vnom": {kv: N}}``
    """
    # Енумы object/measurement-type резолвятся здесь (gridstate.constants),
    # ядро принимает готовые int и читает только контрактные колонки.
    if action not in ("downweight", "deactivate"):
        raise ValueError(f"action must be 'downweight' or 'deactivate', got {action!r}")
    meas_arr = model.measurements.to_numpy().copy()
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
        detect_nominal_substitution=detect_nominal_substitution,
        nominal_substitution_eps=nominal_substitution_eps,
        questionable_sigma2_multiplier=questionable_sigma2_multiplier,
    )
    # NB: model.measurements.to_numpy() кеширует array; **всегда** вызываем
    # update_from_array — он перестраивает _measurements из arr и валидирует кеш.
    model.measurements.update_from_array(meas_arr)
    return stats


def _voltage_range_filter_on_arrays(
    meas_arr,
    nodes_arr,
    *,
    ot_node: int,
    mt_v: int,
    upper_margin_pct: float,
    min_voltage_nominal_kv: float,
    upper_fallback_factor: float,
    action: str,
    detect_nominal_substitution: bool,
    nominal_substitution_eps: float,
    questionable_sigma2_multiplier: float,
) -> dict:
    """ЯДРО: V-range filter над контрактными массивами.

    Мутирует ``meas_arr`` in place (``status``/``variance``/``weight``/``quality``),
    читает ``nodes_arr`` (``voltage_nominal``/``voltage_critical``/``voltage_min``/
    ``voltage_max``). Енумы object/measurement-type приходят готовыми int из
    адаптера. БЕЗ внешних зависимостей и XML. Последовательный цикл в исходном порядке → бит-в-бит.
    """
    node_by_id: dict[int, int] = {int(r["id"]): i for i, r in enumerate(nodes_arr)}

    stats = {
        "checked": 0,
        "out_of_range": 0,
        "downweighted_nominal_substitution": 0,
        "by_vnom": {},
    }

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
        if v_meas <= 0:
            # V≤0 на active-узле — деактивируем как заведомо невалидное.
            # apply_telemetry уже фильтрует V<50%Vn для TM-формул, но
            # measurement мог попасть из внешнего источника (manual seed,
            # SQL-импорт, повторное apply); этот guard закрывает дыру.
            stats["out_of_range"] += 1
            v_nom_int = round(v_nom)
            stats["by_vnom"][v_nom_int] = stats["by_vnom"].get(v_nom_int, 0) + 1
            if action == "deactivate":
                meas_arr[i]["status"] = False
            else:
                new_var = float(meas_arr[i]["variance"]) * questionable_sigma2_multiplier
                meas_arr[i]["variance"] = new_var
                meas_arr[i]["weight"] = 1.0 / new_var if new_var > 0 else 0.0
                meas_arr[i]["quality"] = 1
            continue

        # Нижний порог: voltage_critical → voltage_min → V_ном/2.
        # На некоторых XML встречается заглушка U_KRIT=1.0 на узлах
        # ≥500 кВ; valid-критерий «X ≥ V_ном/2» отсекает 1.0 как
        # заглушку (250 ≪ 1 для 500 кВ).
        half_nom = v_nom * 0.5
        v_crit = float(nodes_arr[ni]["voltage_critical"])
        v_min = (
            float(nodes_arr[ni]["voltage_min"]) if "voltage_min" in nodes_arr.dtype.names else 0.0
        )
        if v_crit >= half_nom:
            lo = v_crit
        elif v_min >= half_nom:
            lo = v_min
        else:
            lo = half_nom

        # Верхний порог: voltage_max·(1+upper_margin_pct) → fallback V_ном·factor.
        # На многих XML U_MAX отсутствует у большой доли узлов ≥500 кВ;
        # fallback 1.4·V_ном.
        v_max = float(nodes_arr[ni]["voltage_max"])
        hi = (
            v_max * (1.0 + upper_margin_pct / 100.0) if v_max > 0 else v_nom * upper_fallback_factor
        )

        stats["checked"] += 1
        v_nom_int = round(v_nom)
        stats["by_vnom"][v_nom_int] = stats["by_vnom"].get(v_nom_int, 0) + 1

        if v_meas < lo or v_meas > hi:
            stats["out_of_range"] += 1
            if action == "deactivate":
                meas_arr[i]["status"] = False
            else:  # downweight
                new_var = float(meas_arr[i]["variance"]) * questionable_sigma2_multiplier
                meas_arr[i]["variance"] = new_var
                meas_arr[i]["weight"] = 1.0 / new_var if new_var > 0 else 0.0
                meas_arr[i]["quality"] = 1
            continue

        if detect_nominal_substitution and abs(v_meas - v_nom) / v_nom < nominal_substitution_eps:
            new_var = float(meas_arr[i]["variance"]) * questionable_sigma2_multiplier
            meas_arr[i]["variance"] = new_var
            meas_arr[i]["weight"] = 1.0 / new_var if new_var > 0 else 0.0
            meas_arr[i]["quality"] = 1  # QUESTIONABLE
            stats["downweighted_nominal_substitution"] += 1

    return stats


def apply_voltage_meas_calibration_for_gen_nodes(
    model,
    *,
    sigma2: float = 0.1,
) -> dict:
    """Override σ² для V-измерений на узлах с активной генерацией + slack.

    На реальных XML-моделях V-измерения на узлах подключения
    генерации (генераторные шины 110-750 кВ) имеют существенно более
    высокую достоверность чем на пассивных нагрузочных узлах:

    * на gen-узлах стоят АЦП класса 0.1-0.2 → σ_V ≈ 0.2-0.4 кВ;
    * на нагрузочных σ исчисляется в процентах от Vnom (несколько кВ).

    Без отдельной калибровки наш default σ² (через ``Sens_Err_U_proc`` в
    ``apply_voltage_range_filter``) даёт mean σ² ≈ 50 кВ², что делает
    V-меру слабым якорем. На 750-кВ кластерах АЭС это давало
    систематическую просадку V на единицы процентов. Эталонный OC
    использует tight σ² на gen-шинах через свою калибровку; этот
    фильтр воспроизводит то же поведение.

    Применяется к:
    * узлам с ``generation_p_max != 0`` после
      :func:`aggregate_generators_to_node` (= узлы с активными ген.);
    * slack-узлам (``node_type == NodeType.SLACK``) — они почти всегда
      сборные шины с эталонным V-meas.

    При σ²=0.1 ΔV p50/p95 заметно улучшается, особенно для 750-кВ
    хвостов.

    Args:
        model: ``Working`` после aggregate_generators_to_node
            и apply_voltage_range_filter.
        sigma2: целевая σ² для V-меры (p.u. или kV² — единицы как у
            существующих V-measurements в model). Default 0.1
            (≈ σ=0.32 кВ для kV-units).

    Returns:
        ``{"updated_meas": N, "target_nodes": N}``.
    """
    from gridstate.constants import NodeType

    meas_arr = model.measurements.to_numpy().copy()
    nodes_arr = model.nodes.to_numpy()
    stats = _voltage_meas_calibration_on_arrays(
        meas_arr,
        nodes_arr,
        slack_type=int(NodeType.SLACK),
        ot_node=int(MeasurementObjectType.NODE),
        mt_v=int(MeasurementType.VOLTAGE),
        sigma2=sigma2,
    )
    model.measurements.update_from_array(meas_arr)
    return stats


def _voltage_meas_calibration_on_arrays(
    meas_arr,
    nodes_arr,
    *,
    slack_type: int,
    ot_node: int,
    mt_v: int,
    sigma2: float,
) -> dict:
    """ЯДРО: tight σ² для V-мер на gen/slack-узлах над контрактом.

    Цели — узлы с ``generation_p_max != 0`` или ``node_type == SLACK`` (енумы
    готовыми int из адаптера). Мутирует ``meas_arr`` (``variance``/``weight``)
    in place, читает ``nodes_arr``. БЕЗ внешних зависимостей и XML.
    """
    target_ids: set[int] = set()
    for n in nodes_arr:
        if not bool(n["status"]):
            continue
        if int(n["node_type"]) == slack_type or (
            "generation_p_max" in nodes_arr.dtype.names and float(n["generation_p_max"]) != 0.0
        ):
            target_ids.add(int(n["id"]))

    n_updated = 0
    for i in range(len(meas_arr)):
        if not bool(meas_arr[i]["status"]):
            continue
        if int(meas_arr[i]["object_type"]) != ot_node:
            continue
        if int(meas_arr[i]["measurement_type"]) != mt_v:
            continue
        if int(meas_arr[i]["object_id"]) not in target_ids:
            continue
        meas_arr[i]["variance"] = float(sigma2)
        meas_arr[i]["weight"] = 1.0 / float(sigma2) if sigma2 > 0 else 0.0
        n_updated += 1

    return {"updated_meas": n_updated, "target_nodes": len(target_ids)}
