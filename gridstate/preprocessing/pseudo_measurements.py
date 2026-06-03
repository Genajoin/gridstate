"""Псевдо-измерения для устранения недонаблюдаемости в WLS.

Узел без telemetry-измерений делает соответствующую часть state-vector
unobservable → Gain matrix singular. Чтобы этого избежать, для каждого
активного узла без `V/P_inj/Q_inj` измерения добавляем слабый прайор:

* `V ≈ V_nominal` (либо `V_magnitude` из загруженной модели, либо `vzd`
  для PV-узла) с σ=10% от Vnom (slack — 0.1%, PV — 5%).
* `P_inj = pg - pn`, `Q_inj = qg - qn` из node-таблицы (для transit
  узлов это даёт 0). σ зависит от типа узла и empty-model detection.

Перенесено из ``tests/_ti_loader.add_pseudo_measurements``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = ["add_pseudo_measurements"]


def _add_pseudo_measurements_on_arrays(
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    meas_arr: np.ndarray,
    node_load_props: dict[int, dict] | None,
    *,
    add_voltage_priors: bool = True,
    add_zero_injections: bool = True,
    voltage_prior_variance: float | None = None,
    slack_voltage_sigma_frac: float = 0.001,
    zero_inj_variance: float = 100.0,
    empty_model_loose_factor: float = 100.0,
    load_inj_loose_factor: float = 10.0,
    terminal_inj_tight_degree: int = 0,
    mid_start: int = 300_000_000,
    boundary_node_ids: set[int] | None = None,
    boundary_branch_p_threshold: float | None = None,
    boundary_inj_loose_factor: float = 10000.0,
    block_bus_v_sigma_frac: float | None = None,
    unobservable_v_sigma_frac: float | None = None,
    unobservable_v_exclude_real_v_neighbor: bool = True,
    unobservable_v_exclude_incident_flow: bool = False,
    unobservable_v_min_vm_deviation: float = 0.0,
) -> tuple[list[dict], dict]:
    """Построить pseudo V/P_inj/Q_inj-строки на контрактных массивах (НЕ мутирует входы).

    vendor-free ядро (CLASS-1, append-паттерн): читает ТОЛЬКО контрактные колонки
    nodes/branches/measurements + готовый ``node_load_props`` (резолвит адаптер);
    ВОЗВРАЩАЕТ ``(new_rows, stats)``. Семантика kwargs — см. адаптер
    ``add_pseudo_measurements``.

    **Должно оставаться последовательным Python-циклом** по ``nodes_arr`` в исходном
    порядке, с тем же порядком добавления строк внутри узла (V-приор → P_inj → Q_inj)
    и монотонной раздачей ``mid`` — арифметика σ²/p_inj_prior скалярная
    детерминированная, без векторизации → **строгий бит-в-бит 1e-9**.
    """
    have_v: set[int] = set()
    have_pinj: set[int] = set()
    have_qinj: set[int] = set()
    for r in meas_arr:
        if not r["status"]:
            continue  # disabled measurements не блокируют добавление prior'ов
        if int(r["object_type"]) != OBJ_NODE:
            continue
        nid = int(r["object_id"])
        kind = int(r["measurement_type"])
        if kind == KIND_VOLTAGE:
            have_v.add(nid)
        elif kind == KIND_POWER_INJECTION_P:
            have_pinj.add(nid)
        elif kind == KIND_POWER_INJECTION_Q:
            have_qinj.add(nid)

    # Авто-детект «пустой» модели: ряд снимков имеет в model.json
    # нули в pg/pn/qg/qn (режим хранится отдельно).
    # На таких моделях критерий transit pg=pn=qg=qn=0 срабатывает на
    # ВСЕХ узлах → жёсткий prior σ²=zero_inj_variance давит на TM →
    # SE расходится. При empty-model ослабляем все priors на
    # empty_model_loose_factor.
    n_active = 0
    n_loaded = 0
    for row in nodes_arr:
        if not row["status"]:
            continue
        n_active += 1
        if (
            abs(float(row["generation_p"])) > 1e-3
            or abs(float(row["load_p"])) > 1e-3
            or abs(float(row["generation_q"])) > 1e-3
            or abs(float(row["load_q"])) > 1e-3
        ):
            n_loaded += 1
    is_empty_model = n_active > 0 and n_loaded * 100 < n_active  # <1% loaded

    # Boundary-узлы: явный override + опц. авто-детект через branch-P-meas.
    # Поле NodeCollection.type_ekv существует, но его семантика в наших
    # снимках не «эквивалент»: type_ekv=1 встречается и у обычных
    # 110 кВ-узлов, автоматическая маркировка их как boundary ломает SE
    # (V_min до 0.055 и не сходится). Поэтому type_ekv для авто-detection
    # не используется.
    boundary: set[int] = set(boundary_node_ids or set())
    if boundary_branch_p_threshold is not None:
        # Real (не pseudo) branch-P-meas: id < mid_start.
        # Для каждой ветви возьмём max |P| по любым её сторонам.
        max_abs_p_per_branch: dict[int, float] = {}
        for r in meas_arr:
            if not r["status"]:
                continue
            if int(r["object_type"]) != OBJ_BRANCH:
                continue
            if int(r["measurement_type"]) != KIND_POWER_P:
                continue
            if int(r["id"]) >= mid_start:
                continue
            bid = int(r["object_id"])
            v = abs(float(r["value"]))
            if v > max_abs_p_per_branch.get(bid, 0.0):
                max_abs_p_per_branch[bid] = v
        # Узлы у которых уже есть real-P_inj-meas — точно не boundary.
        real_pinj_nodes: set[int] = set()
        for r in meas_arr:
            if not r["status"]:
                continue
            if int(r["object_type"]) != OBJ_NODE:
                continue
            if int(r["measurement_type"]) != KIND_POWER_INJECTION_P:
                continue
            if int(r["id"]) >= mid_start:
                continue
            real_pinj_nodes.add(int(r["object_id"]))
        # Концы ветвей с большим P-meas → кандидаты в boundary.
        for r in branches_arr:
            if not r["status"]:
                continue
            bid = int(r["id"])
            if max_abs_p_per_branch.get(bid, 0.0) < boundary_branch_p_threshold:
                continue
            for end in (int(r["from_node"]), int(r["to_node"])):
                if end not in real_pinj_nodes:
                    boundary.add(end)

    # Block-bus detection: одно-ветевой узел, единственная ветвь — trafo с tap≠1.
    block_buses: set[int] = set()
    if block_bus_v_sigma_frac is not None:
        active_branches = branches_arr[branches_arr["status"]]
        degree: dict[int, int] = {}
        first_branch: dict[int, np.void] = {}
        for r in active_branches:
            for end in (int(r["from_node"]), int(r["to_node"])):
                degree[end] = degree.get(end, 0) + 1
                if end not in first_branch:
                    first_branch[end] = r
        for nid_, deg in degree.items():
            if deg != 1:
                continue
            b = first_branch[nid_]
            if int(b["branch_type"]) != 1:
                continue
            if abs(float(b["tap_ratio"]) - 1.0) < 1e-3:
                continue
            block_buses.add(nid_)

    # Observability-сеты для жёсткого V-якоря ненаблюдаемых узлов.
    # Заполняются только если запрошен unobservable_v_sigma_frac.
    real_v_neighbor: set[int] = set()
    incident_real_flow: set[int] = set()
    if unobservable_v_sigma_frac is not None:
        if unobservable_v_exclude_real_v_neighbor:
            for r in branches_arr:
                if not r["status"]:
                    continue
                f, t = int(r["from_node"]), int(r["to_node"])
                if f in have_v:
                    real_v_neighbor.add(t)
                if t in have_v:
                    real_v_neighbor.add(f)
        if unobservable_v_exclude_incident_flow:
            br_ends = {int(r["id"]): (int(r["from_node"]), int(r["to_node"])) for r in branches_arr}
            for r in meas_arr:
                if not r["status"] or int(r["object_type"]) != OBJ_BRANCH:
                    continue
                if int(r["measurement_type"]) not in (KIND_POWER_P, KIND_POWER_Q):
                    continue
                if int(r["id"]) >= mid_start:
                    continue
                ends = br_ends.get(int(r["object_id"]))
                if ends:
                    incident_real_flow.update(ends)

    # Степень узла (активные ветви) — для terminal_inj_tight_degree.
    node_degree: dict[int, int] = {}
    if terminal_inj_tight_degree > 0:
        for r in branches_arr:
            if not r["status"]:
                continue
            for end in (int(r["from_node"]), int(r["to_node"])):
                node_degree[end] = node_degree.get(end, 0) + 1

    mid = mid_start
    n_v_added = 0
    n_zinj_added = 0
    n_block_bus_v = 0
    n_unobs_v = 0

    new_rows: list[dict] = []
    for row in nodes_arr:
        if not row["status"]:
            continue
        nid = int(row["id"])
        vn = float(row["voltage_nominal"])
        vm = float(row["voltage_magnitude"])
        is_slack = int(row["node_type"]) == 2  # NodeType.SLACK = 2

        if add_voltage_priors and nid not in have_v and vn > 0:
            v_prior_value = vm if vm > 0 else vn
            is_pv_with_vzd = False
            if node_load_props is not None and nid in node_load_props:
                prop = node_load_props[nid]
                vzd = float(prop.get("vzd", 0.0))
                # Slack и PV-узлы с заданной уставкой U_ZAD используют её как
                # V-prior (семантика входного формата: balancing-узел держит V=vzd,
                # PV-генератор управляет Q для V=vzd). Раньше slack
                # исключался — pseudo брал V=Vnom (flat), что отличалось от
                # vzd на 2-5 кВ и тянуло SE к номиналу, а не к уставке.
                if vzd > 0 and (prop.get("exist_gen") or is_slack):
                    v_prior_value = vzd
                    if not is_slack:
                        is_pv_with_vzd = True
            sigma = voltage_prior_variance
            is_block_bus = nid in block_buses
            if sigma is None:
                if is_slack:
                    sigma = (slack_voltage_sigma_frac * vn) ** 2
                elif is_pv_with_vzd:
                    sigma = (0.05 * vn) ** 2  # PV-узел жёстче, vzd≈vras
                elif (
                    unobservable_v_sigma_frac is not None
                    and nid not in real_v_neighbor
                    and nid not in incident_real_flow
                    and abs(v_prior_value - vn) >= unobservable_v_min_vm_deviation * vn
                ):
                    # Ненаблюдаемый узел без real-V (и без real-V-соседа /
                    # инцидентного flow по гейтам), у которого прайор —
                    # нетривиальная рабочая точка before_OC (не номинал-
                    # плейсхолдер): эталонная SE держит V на vm. Жёсткий якорь
                    # вместо 5% — иначе loose σ даёт дрейф.
                    sigma = (unobservable_v_sigma_frac * vn) ** 2
                    n_unobs_v += 1
                elif is_block_bus and block_bus_v_sigma_frac is not None:
                    # Блочная шина (LV-узел трансформатора без real-V-meas):
                    # 5% Vn слишком tight — V_LV должна следовать за V_HV/tap,
                    # а pseudo-V=Vn тянет к номиналу. Ослабляем σ.
                    sigma = (block_bus_v_sigma_frac * vn) ** 2
                    n_block_bus_v += 1
                else:
                    # 5% Vnom: жёстче чем 10% (был раньше). На weakly-observable
                    # узлах (degree=1, empty-model P_inj σ=100) 10% позволял SE
                    # утаскивать V в 0.3-0.5 p.u. — pseudo-V не держал якорь.
                    sigma = (0.05 * vn) ** 2
            new_rows.append(
                {
                    "id": mid,
                    "object_type": OBJ_NODE,
                    "object_id": nid,
                    "measurement_type": KIND_VOLTAGE,
                    "value": v_prior_value,
                    "variance": float(sigma),
                    "status": True,
                    "quality": 0,
                    "is_pseudo": True,
                }
            )
            mid += 1
            n_v_added += 1

        if add_zero_injections and nid not in have_pinj:
            pg = float(row["generation_p"])
            pn = float(row["load_p"])
            qg = float(row["generation_q"])
            qn = float(row["load_q"])
            p_inj_prior = pg - pn
            q_inj_prior = qg - qn

            if node_load_props is not None and nid in node_load_props:
                prop = node_load_props[nid]
                transit = not bool(prop.get("exist_load")) and not bool(prop.get("exist_gen"))
            else:
                transit = abs(pg) < 1e-3 and abs(pn) < 1e-3 and abs(qg) < 1e-3 and abs(qn) < 1e-3

            if is_empty_model:
                if node_load_props is not None and transit:
                    var_p = zero_inj_variance * 10.0
                else:
                    var_p = zero_inj_variance * empty_model_loose_factor
            else:
                # Нагрузочные/ген узлы — рыхлее транзитных в load_inj_loose_factor раз
                # (default 10.0 = историческое поведение). На радиальных терминалах
                # с материализованной нагрузкой≈0 и без real-ТМ это рыхление пускает
                # V в overshoot (ветвь-Q гонит фантомный Q).
                var_p = zero_inj_variance if transit else zero_inj_variance * load_inj_loose_factor
            # Радиальный терминал (degree ≤ N): net-инжекция = материализ. нагрузка,
            # рыхлый pseudo пускает overshoot V через цепочку высоко-X ветвей.
            # Зажимаем до базовой σ (как транзит). Таргет: deg=1 тупики
            # региональных моделей, НЕ трогает meshed-узлы (worst deg=3).
            # default off (degree=0).
            if (
                terminal_inj_tight_degree > 0
                and node_degree.get(nid, 99) <= terminal_inj_tight_degree
            ):
                var_p = min(var_p, zero_inj_variance)
            var_q = var_p
            # Boundary-узлы — большая σ² (см. docstring и
            # docs/audit/audit_se_boundary_nodes.md).
            if nid in boundary:
                var_p *= boundary_inj_loose_factor
                var_q *= boundary_inj_loose_factor
            new_rows.append(
                {
                    "id": mid,
                    "object_type": OBJ_NODE,
                    "object_id": nid,
                    "measurement_type": KIND_POWER_INJECTION_P,
                    "value": float(p_inj_prior),
                    "variance": float(var_p),
                    "status": True,
                    "quality": 0,
                    "is_pseudo": True,
                }
            )
            mid += 1
            if nid not in have_qinj:
                new_rows.append(
                    {
                        "id": mid,
                        "object_type": OBJ_NODE,
                        "object_id": nid,
                        "measurement_type": KIND_POWER_INJECTION_Q,
                        "value": float(q_inj_prior),
                        "variance": float(var_q),
                        "status": True,
                        "quality": 0,
                        "is_pseudo": True,
                    }
                )
                mid += 1
                n_zinj_added += 1

    stats = {
        "v_priors_added": n_v_added,
        "zero_inj_added": n_zinj_added,
        "boundary_nodes": len(boundary),
        "block_buses": len(block_buses),
        "block_bus_v_loose": n_block_bus_v,
        "unobservable_v_tight": n_unobs_v,
    }
    return new_rows, stats


def add_pseudo_measurements(
    model: Working,
    *,
    add_voltage_priors: bool = True,
    add_zero_injections: bool = True,
    voltage_prior_variance: float | None = None,
    slack_voltage_sigma_frac: float = 0.001,
    zero_inj_variance: float = 100.0,
    empty_model_loose_factor: float = 100.0,
    load_inj_loose_factor: float = 10.0,
    terminal_inj_tight_degree: int = 0,
    node_load_props: dict[int, dict] | None = None,
    mid_start: int = 300_000_000,
    boundary_node_ids: set[int] | None = None,
    boundary_branch_p_threshold: float | None = None,
    boundary_inj_loose_factor: float = 10000.0,
    block_bus_v_sigma_frac: float | None = None,
    unobservable_v_sigma_frac: float | None = None,
    unobservable_v_exclude_real_v_neighbor: bool = True,
    unobservable_v_exclude_incident_flow: bool = False,
    unobservable_v_min_vm_deviation: float = 0.0,
) -> dict:
    """Дополнить модель псевдо-измерениями для устранения недонаблюдаемости.

    1. **V-приоры**: для каждого активного узла без VOLTAGE-измерения
       добавить ``V = voltage_magnitude/nominal`` с дисперсией:
       slack — ``(slack_voltage_sigma_frac·Vnom)²``; остальные —
       ``(0.1·Vnom)²`` (слабый прайор).
    2. **Zero-injection**: для активных узлов без INJECTION-измерения
       добавить ``P_inj = Q_inj = (pg-pn)`` (из node-таблицы) с дисперсией:

       * **транзитный узел** (``exist_load=0 AND exist_gen=0`` если задан
         ``node_load_props``, иначе fallback ``pg=pn=qg=qn=0`` в node):
         жёсткий prior ``variance=zero_inj_variance``.
       * **нагрузочный/генераторный**: слабый prior
         ``variance = 10·zero_inj_variance``.
       * **empty-model** (заполнено меньше 1% узлов в node-таблице):
         все узлы получают ``variance = zero_inj_variance·empty_model_loose_factor``,
         кроме true-транзитов (``EXIST_PN=0 AND EXIST_PG=0`` из
         ``node_load_props``), у которых дисперсия
         ``10·zero_inj_variance``.

    Args:
        model: Working для модификации (in-place).
        add_voltage_priors: добавлять ли V-прайоры.
        add_zero_injections: добавлять ли P/Q-инжекции.
        voltage_prior_variance: явное σ²; ``None`` — авто.
        slack_voltage_sigma_frac: фракция σ_V для slack-узла.
        zero_inj_variance: базовое σ² для transit узлов (МВт²).
        empty_model_loose_factor: множитель для empty-model.
        node_load_props: ``ny → {exist_load, exist_gen, vzd, ...}`` —
            семантика входного формата EXIST_PN/PG. Без него — fallback на
            нули в node-таблице.
        mid_start: начальный ID для новых measurements.
        boundary_node_ids: явный набор ID-узлов, к которым нужно
            применить boundary-режим (см. ниже). Объединяется с
            результатом авто-детекции через ``boundary_branch_p_threshold``.
        boundary_branch_p_threshold: порог |P| (МВт) для авто-детекции
            boundary-узлов. ``None`` — авто-детект отключён. Если задан,
            узел помечается как boundary, если к нему подходит хотя бы
            одна active branch с **real** branch-P-измерением (не
            pseudo) с ``|P| ≥ threshold``, и при этом сам узел не имеет
            real-P_inj-измерения. Это распознаёт «эквивалентные границы»
            региональной модели — узлы за пределами активной области,
            через которые большая мощность приходит/уходит, но локально
            P_inj = 0 (в режиме регионального SE режим соседей не виден).
        boundary_inj_loose_factor: множитель к σ² для
            P_inj/Q_inj-pseudo на boundary-узлах. Default 10000.0 — даёт
            σ²×10⁴ (σ × 100), что фактически выводит boundary
            P_inj/Q_inj-prior из давления на оценку, оставляя его в
            системе только для observability.
        unobservable_v_sigma_frac: если задан — для **ненаблюдаемых** узлов
            без real-V якорим pseudo-V жёстко: ``σ = (frac·Vnom)²`` вместо
            обычных 5%. Мотивация: эталонная SE держит ненаблюдаемый
            узел на рабочей точке before_OC (``voltage_magnitude`` ≈ его
            after_OC V), а наш loose 5%-прайор позволяет солверу дрейфовать
            (top_dV доминанта на региональных моделях: terminal-хвосты уходят
            на 10-22 % от эталонной SE). Жёсткий якорь к vm-прайору
            воспроизводит якорение эталонной SE.
            ``None`` (default) — поведение не меняется (5%).
        unobservable_v_exclude_real_v_neighbor: при tighten НЕ якорить узел,
            у которого есть сосед по active-ветви с **real**-V-замером —
            такой узел наблюдаем через линию (эталонная SE ведёт его V от
            соседа, не держит на vm). Default True.
        unobservable_v_exclude_incident_flow: при tighten НЕ якорить узел с
            **real** branch-P/Q-замером на инцидентной ветви. Default False
            (эмпирически inc-flow НЕ удерживает V от дрейфа — узел всё равно
            выигрывает от якоря).
        unobservable_v_min_vm_deviation: минимальное относительное отклонение
            прайора от номинала ``|v_prior - Vnom|/Vnom`` для tighten. Default
            0.0 (без ограничения). Мотивация: когда ``vm`` ≈ ``Vnom`` (или
            ``vm=0`` → fallback к Vnom), прайор — плейсхолдер-номинал, а не
            реальная рабочая точка before_OC; эталонная SE на таких узлах
            могла увести V далеко от номинала (на региональных моделях ~½ узлов
            prior>2% от эталонной SE) →
            жёсткий якорь к номиналу регрессирует медиану. Порог >0 якорит
            только узлы с нетривиальным before_OC V.

    Returns:
        ``{"v_priors_added": N, "zero_inj_added": N, "boundary_nodes": N}``
        — счётчики.
    """
    nodes_arr = model.nodes.to_numpy()
    meas_arr = model.measurements.to_numpy()
    branches_arr = model.branches.to_numpy()

    # exist_load/exist_gen заполняет XmlFormat (XML-pipeline);
    # если они есть — выводим node_load_props из модели автоматически.
    if node_load_props is None and (
        np.any(nodes_arr["exist_load"]) or np.any(nodes_arr["exist_gen"])
    ):
        from gridstate.preprocessing.node_props import extract_node_load_props_from_model

        node_load_props = extract_node_load_props_from_model(model)

    new_rows, stats = _add_pseudo_measurements_on_arrays(
        nodes_arr,
        branches_arr,
        meas_arr,
        node_load_props,
        add_voltage_priors=add_voltage_priors,
        add_zero_injections=add_zero_injections,
        voltage_prior_variance=voltage_prior_variance,
        slack_voltage_sigma_frac=slack_voltage_sigma_frac,
        zero_inj_variance=zero_inj_variance,
        empty_model_loose_factor=empty_model_loose_factor,
        load_inj_loose_factor=load_inj_loose_factor,
        terminal_inj_tight_degree=terminal_inj_tight_degree,
        mid_start=mid_start,
        boundary_node_ids=boundary_node_ids,
        boundary_branch_p_threshold=boundary_branch_p_threshold,
        boundary_inj_loose_factor=boundary_inj_loose_factor,
        block_bus_v_sigma_frac=block_bus_v_sigma_frac,
        unobservable_v_sigma_frac=unobservable_v_sigma_frac,
        unobservable_v_exclude_real_v_neighbor=unobservable_v_exclude_real_v_neighbor,
        unobservable_v_exclude_incident_flow=unobservable_v_exclude_incident_flow,
        unobservable_v_min_vm_deviation=unobservable_v_min_vm_deviation,
    )
    # Пакетная вставка за одну конкатенацию: per-row .add() даёт O(n²) на
    # тысячах псевдо-измерений (узкое место крупных моделей).
    model.measurements.add_many(new_rows)
    return stats
