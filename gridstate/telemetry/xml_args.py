"""Применение телеметрии/материализации к узловому режиму (prep-ядра + адаптеры).

Контрактные ядра z-вектора (``_apply_telemetry_on_arrays``) и материализации режима
(``_materialize_area_on_arrays``) + тонкие адаптеры применения готовых числовых планов
(``apply_telemetry_resolved`` / ``apply_materialize_resolved``). Спецификации привязок
(``FormulaSpec`` / ``ArgEntry`` + kind-карты) — в ``gridstate.telemetry._specs``.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median

from gridstate.telemetry._specs import (
    _INJ_MT,
    _KIND_MAP,
    _NODE_INJ_MAP,
    ArgEntry,  # noqa: F401  (реэкспорт для обратной совместимости)
    FormulaSpec,
)
from gridstate.telemetry.loss_filter import is_branch_q_consistent_with_physics
from gridstate.telemetry.quality import QUALITY_BAD, QUALITY_QUESTIONABLE, aggregate_qualities
from gridstate.telemetry.units import variance_branch_q, variance_power, variance_voltage


def apply_telemetry_resolved(
    model,
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
    *,
    total_args: int,
    questionable_sigma2_multiplier: float = 100.0,
    branch_p_sigma_frac: float = 0.02,
    branch_q_sigma_frac: float = 0.07,
    branch_q_sigma_charging_alpha: float = 0.10,
    sign_inconsistency_threshold_mw: float | None = 100.0,
    q_inconsistency_threshold_mvar: float | None = None,
    q_inconsistency_high_voltage_kv: float = 500.0,
    q_inconsistency_threshold_mvar_hv: float | None = None,
    q_inconsistency_action_hv: str = "drop",
    q_inconsistency_downweight_factor: float = 100.0,
    q_loss_filter_enabled: bool = True,
    q_loss_filter_floor_mvar: float = 50.0,
    q_loss_filter_rel_pct: float = 30.0,
    q_loss_filter_action: str = "downweight",
    q_loss_filter_downweight_factor: float = 100.0,
) -> dict[str, int]:
    """Фаза-A (релокация): применить готовый ``resolved`` z-вектор к ``measurements``.

    Чистое применение (без XML/snapshot/формул): снимок ``measurements``/``nodes``/
    ``branches`` → ядро :func:`_apply_telemetry_on_arrays` → write-back + ``.add()``
    NODE-инжекций. Зовётся шагом ``run()`` на своей позиции (после ``apply_topology``/
    ``apply_rpn``: ядро читает branch ``status``/``susceptance``).
    """
    meas_arr = model.measurements.to_numpy().copy()
    nodes_arr = model.nodes.to_numpy()
    branches_arr = model.branches.to_numpy()

    stats, new_rows = _apply_telemetry_on_arrays(
        meas_arr,
        nodes_arr,
        branches_arr,
        arg_keys,
        resolved,
        total_args=total_args,
        questionable_sigma2_multiplier=questionable_sigma2_multiplier,
        branch_p_sigma_frac=branch_p_sigma_frac,
        branch_q_sigma_frac=branch_q_sigma_frac,
        branch_q_sigma_charging_alpha=branch_q_sigma_charging_alpha,
        sign_inconsistency_threshold_mw=sign_inconsistency_threshold_mw,
        q_inconsistency_threshold_mvar=q_inconsistency_threshold_mvar,
        q_inconsistency_high_voltage_kv=q_inconsistency_high_voltage_kv,
        q_inconsistency_threshold_mvar_hv=q_inconsistency_threshold_mvar_hv,
        q_inconsistency_action_hv=q_inconsistency_action_hv,
        q_inconsistency_downweight_factor=q_inconsistency_downweight_factor,
        q_loss_filter_enabled=q_loss_filter_enabled,
        q_loss_filter_floor_mvar=q_loss_filter_floor_mvar,
        q_loss_filter_rel_pct=q_loss_filter_rel_pct,
        q_loss_filter_action=q_loss_filter_action,
        q_loss_filter_downweight_factor=q_loss_filter_downweight_factor,
    )

    model.measurements.update_from_array(meas_arr)
    for row in new_rows:
        model.measurements.add(row)

    return stats


def _apply_telemetry_on_arrays(
    meas_arr,
    nodes_arr,
    branches_arr,
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    *,
    total_args: int,
    questionable_sigma2_multiplier: float = 100.0,
    branch_p_sigma_frac: float = 0.02,
    branch_q_sigma_frac: float = 0.07,
    branch_q_sigma_charging_alpha: float = 0.10,
    sign_inconsistency_threshold_mw: float | None = 100.0,
    q_inconsistency_threshold_mvar: float | None = None,
    q_inconsistency_high_voltage_kv: float = 500.0,
    q_inconsistency_threshold_mvar_hv: float | None = None,
    q_inconsistency_action_hv: str = "drop",
    q_inconsistency_downweight_factor: float = 100.0,
    q_loss_filter_enabled: bool = True,
    q_loss_filter_floor_mvar: float = 50.0,
    q_loss_filter_rel_pct: float = 30.0,
    q_loss_filter_action: str = "downweight",
    q_loss_filter_downweight_factor: float = 100.0,
) -> tuple[dict[str, int], list[dict]]:
    """ЯДРО: применение телеметрии над контрактными numpy-массивами.

    Чистая работа над ``SE_INPUT.measurements`` (``meas_arr``, мутируется
    in-place) / ``SE_INPUT.nodes`` (``nodes_arr``) / ``SE_INPUT.branches``
    (``branches_arr``). БЕЗ внешних зависимостей, БЕЗ snapshot, БЕЗ FORMULE: все XML/snapshot
    значения уже посчитаны адаптером и переданы в ``resolved`` —
    ``{(obj_id, kind): (value|None, n_resolved, guid_first, quality)}``.
    Порядок итерации задаётся ``arg_keys`` (= ``list(args.keys())``).

    Возвращает ``(stats, new_rows)``: ``new_rows`` — список dict-ов для
    последующего ``model.measurements.add()`` (NODE-инжекции PG/PN/QG/QN);
    их ``id`` назначаются здесь, чтобы быть бит-идентичными прежнему
    in-loop ``.add()``-блоку.
    """

    _variance_voltage = variance_voltage

    def _variance_p(value_mva: float) -> float:
        return variance_power(value_mva, sigma_frac=branch_p_sigma_frac)

    # Index measurements: (ot, mt, side, oid) → row idx in numpy array
    arr = meas_arr
    meas_idx: dict[tuple[int, int, int, int], int] = {}
    for i, r in enumerate(arr):
        key = (
            int(r["object_type"]),
            int(r["measurement_type"]),
            int(r["branch_side"]),
            int(r["object_id"]),
        )
        meas_idx.setdefault(key, i)

    # node_id → voltage_nominal для расчёта sigma_V.
    vn_by_node: dict[int, float] = {int(r["id"]): float(r["voltage_nominal"]) for r in nodes_arr}
    # branch_id → status. Меру нельзя ставить на отключённую ветвь:
    # WLS получит residual P/Q≠0 на ветви где Y-bus её зануляет —
    # iter будет компенсировать через V/δ соседних узлов и портит
    # решение. Проверяется при активации branch-meas (ot=1).
    branch_status_by_id: dict[int, bool] = {int(r["id"]): bool(r["status"]) for r in branches_arr}

    # Сначала отключим ВСЕ measurements (на входе они приходят со status=True
    # по умолчанию). После apply сделаем status=True только для тех, что
    # покрыты snapshot-ом.
    arr["status"] = False

    stats = {
        "applied": 0,
        "applied_questionable": 0,
        "skipped_no_value": 0,
        "skipped_no_meas": 0,
        "skipped_bad_quality": 0,
        "total_args": total_args,
        "skipped_kind_unsupported": 0,
        "skipped_formula_error": 0,
        "skipped_v_below_half_nominal": 0,
        "node_inj_added": 0,
    }

    # Аккумулятор для NODE injections:
    #   (obj_id, "P"|"Q") → list of (signed_val, guid, quality).
    # Net P_inj = sum(values), quality = max(qualities) (worst-case).
    inj_acc: dict[tuple[int, str], list[tuple[float, str, int]]] = {}

    # Pre-pass: распознать ветви с sign-inconsistent P-измерениями
    # (PBEG и PEND оба входят в линию, т.е. одного знака после
    # учёта invert) — это указывает на битые привязки FORMULE-ARG
    # к не той ветви или на устаревшую телеметрию. Сами branch-meas
    # таких ветвей не активируем.
    inconsistent_branches: set[int] = set()
    if sign_inconsistency_threshold_mw is not None:
        for obj_id, kind in arg_keys:
            if kind != "PBEG":
                continue
            if (obj_id, "PEND") not in resolved:
                continue
            v_beg = resolved[(obj_id, "PBEG")][0]
            v_end = resolved[(obj_id, "PEND")][0]
            if v_beg is None or v_end is None:
                continue
            # PBEG/PEND-значения уже учитывают `INVERT_CK2011` для
            # своих ARG. На физически согласованной ВЛ выполняется
            # ``v_beg ≈ -v_end + потери``, т.е. они **противоположных
            # знаков**. Если |v_beg + v_end| превосходит порог —
            # это либо знак-аномалия, либо неправильная привязка.
            if abs(v_beg + v_end) >= sign_inconsistency_threshold_mw:
                inconsistent_branches.add(int(obj_id))
        stats["sign_inconsistent_branches"] = len(inconsistent_branches)

    # Pre-pass для Q: проверка физической согласованности
    # |Q_BEG + Q_END| с π-схемой ветви (B·V², X-loss).
    #
    # **Старый подход** (deprecated, default отключён): flat threshold
    # `q_inconsistency_threshold_mvar` ложно дропает 750 кВ Q-меры с
    # зарядной B (Q_charging ≈ 1066 МВар легитимны).
    #
    # **Новый подход** (`q_loss_filter_enabled=True`, default):
    # сравнивает |Q_beg+Q_end| с расчётным ожиданием через V_flat=V_nom
    # и branch.{susceptance, reactance}. Outlier если расхождение
    # `|обс - exp| > max(floor, rel_pct/100 · exp)`. См.
    # `gridstate.telemetry.loss_filter.is_branch_q_consistent_with_physics`.
    branch_vn: dict[int, float] = {}
    branch_b: dict[int, float] = {}
    branch_x: dict[int, float] = {}
    for i in range(len(branches_arr)):
        bid = int(branches_arr[i]["id"])
        fn = int(branches_arr[i]["from_node"])
        vn_kv = vn_by_node.get(fn, 0.0)
        branch_vn[bid] = vn_kv
        branch_b[bid] = float(branches_arr[i]["susceptance"])
        branch_x[bid] = float(branches_arr[i]["reactance"])

    effective_hv = (
        q_inconsistency_threshold_mvar_hv
        if q_inconsistency_threshold_mvar_hv is not None
        else q_inconsistency_threshold_mvar
    )
    q_inconsistent_branches: set[int] = set()  # ветви для drop
    q_inconsistent_hv_branches: set[int] = set()  # ветви для downweight

    flat_filter_active = (
        q_inconsistency_threshold_mvar is not None or q_inconsistency_threshold_mvar_hv is not None
    )
    if flat_filter_active or q_loss_filter_enabled:
        for obj_id, kind in arg_keys:
            if kind != "QBEG":
                continue
            if (obj_id, "QEND") not in resolved:
                continue
            v_beg = resolved[(obj_id, "QBEG")][0]
            v_end = resolved[(obj_id, "QEND")][0]
            if v_beg is None or v_end is None:
                continue
            vn = branch_vn.get(int(obj_id), 0.0)

            # Новый physical detector (приоритет над flat-thresh).
            if q_loss_filter_enabled and vn > 0:
                # Достанем P_typical из PBEG/PEND для оценки X-loss.
                p_typical = 0.0
                for p_kind in ("PBEG", "PEND"):
                    if (obj_id, p_kind) not in resolved:
                        continue
                    pv = resolved[(obj_id, p_kind)][0]
                    if pv is not None:
                        p_typical = max(p_typical, abs(pv))
                is_ok, _, _, _ = is_branch_q_consistent_with_physics(
                    q_observed_mvar=v_beg + v_end,
                    vn_kv=vn,
                    susceptance_si=branch_b.get(int(obj_id), 0.0),
                    reactance_si=branch_x.get(int(obj_id), 0.0),
                    p_typical_mw=p_typical,
                    floor_mvar=q_loss_filter_floor_mvar,
                    rel_pct=q_loss_filter_rel_pct,
                )
                if not is_ok:
                    if q_loss_filter_action == "downweight":
                        q_inconsistent_hv_branches.add(int(obj_id))
                    else:
                        q_inconsistent_branches.add(int(obj_id))
                continue

            # Fallback: старый flat-detector.
            is_hv = vn >= q_inconsistency_high_voltage_kv
            threshold = effective_hv if is_hv else q_inconsistency_threshold_mvar
            if threshold is None:
                continue
            if abs(v_beg + v_end) >= threshold:
                if is_hv and q_inconsistency_action_hv == "downweight":
                    q_inconsistent_hv_branches.add(int(obj_id))
                else:
                    q_inconsistent_branches.add(int(obj_id))
        stats["q_inconsistent_branches"] = len(q_inconsistent_branches)
        stats["q_inconsistent_hv_branches"] = len(q_inconsistent_hv_branches)

    for obj_id, kind in arg_keys:
        # NODE injection (PG/PN/QG/QN) или GENERATOR (PG_G<n>, QG_G<n>) →
        # накапливать для последующего .add(). PG_G* и QG_G* считаются как
        # положительный вклад P_gen/Q_gen к узлу (как и обычное PG/QG).
        kind_base = kind.split("_G", 1)[0] if kind.startswith(("PG_G", "QG_G")) else kind
        if kind_base in _NODE_INJ_MAP:
            value, n_res, guid_first, q = resolved[(obj_id, kind)]
            if value is None:
                if n_res == 0:
                    stats["skipped_no_value"] += 1
                else:
                    stats["skipped_formula_error"] += 1
                continue
            if q == QUALITY_BAD:
                stats["skipped_bad_quality"] += 1
                continue
            pq, mult = _NODE_INJ_MAP[kind_base]
            val = value * mult
            inj_acc.setdefault((obj_id, pq), []).append((val, guid_first, q))
            continue

        if kind not in _KIND_MAP:
            stats["skipped_kind_unsupported"] += 1
            continue
        # Branch с sign-inconsistent P-meas — все 4 измерения этого
        # LINE-OBJ_ID не активируем (данные битые).
        if kind in ("PBEG", "PEND", "QBEG", "QEND") and obj_id in inconsistent_branches:
            stats.setdefault("skipped_sign_inconsistent", 0)
            stats["skipped_sign_inconsistent"] += 1
            ot_f, mt_f, side_f, _ = _KIND_MAP[kind]
            idx_f = meas_idx.get((ot_f, mt_f, side_f, obj_id))
            if idx_f is not None:
                arr[idx_f]["filter_flag"] = 5  # p_sign_inconsistency
            continue
        # Q-inconsistent ветви — только Q-меры пропускаем, P оставляем.
        if kind in ("QBEG", "QEND") and obj_id in q_inconsistent_branches:
            stats.setdefault("skipped_q_inconsistent", 0)
            stats["skipped_q_inconsistent"] += 1
            ot_f, mt_f, side_f, _ = _KIND_MAP[kind]
            idx_f = meas_idx.get((ot_f, mt_f, side_f, obj_id))
            if idx_f is not None:
                arr[idx_f]["filter_flag"] = 2  # q_inconsistency
            continue
        # Q-inconsistent на HV-ВЛ с action=downweight: НЕ дропаем; маркер
        # для увеличения σ² ×factor применим после активации значения.
        # (логика — сразу под расчётом val/var)
        ot, mt, side, sign = _KIND_MAP[kind]
        # Не активируем меру на отключённой ветви — иначе WLS получит
        # residual P/Q≠0 на ветви, чью Y-bus занулил.
        if ot == 1 and not branch_status_by_id.get(int(obj_id), False):
            stats.setdefault("skipped_branch_off", 0)
            stats["skipped_branch_off"] += 1
            continue
        key = (ot, mt, side, obj_id)
        idx = meas_idx.get(key)
        if idx is None:
            stats["skipped_no_meas"] += 1
            continue
        value, n_res, guid_first, q = resolved[(obj_id, kind)]
        if value is None:
            if n_res == 0:
                stats["skipped_no_value"] += 1
            else:
                stats["skipped_formula_error"] += 1
            continue
        if q == QUALITY_BAD:
            stats["skipped_bad_quality"] += 1
            arr[idx]["filter_flag"] = 1  # bad_quality
            continue
        val = value * sign
        # V-measurement < 50% Vnom — sentinel «датчик не работает» / отключённое
        # оборудование. Принимать его как жёсткий якорь нельзя: SE будет тянуть
        # V→0 на узле и в окрестности (см. audit_se_nonphysical_minimum.md). Не
        # активируем — добавит pseudo-V=Vnom через add_pseudo_measurements.
        if mt == 2 and ot == 0:
            obj_node = int(arr[idx]["object_id"])
            vn = vn_by_node.get(obj_node, 220.0)
            if vn > 0 and abs(val) < 0.5 * vn:
                stats["skipped_v_below_half_nominal"] += 1
                arr[idx]["filter_flag"] = 4  # v_below_half_nominal
                continue
        if arr[idx]["status"]:
            arr[idx]["value"] = float(arr[idx]["value"]) + val
            # Worst-case аккумуляция quality при многократном попадании.
            arr[idx]["quality"] = max(int(arr[idx]["quality"]), q)
        else:
            arr[idx]["value"] = val
            arr[idx]["status"] = True
            arr[idx]["source_guid"] = guid_first
            arr[idx]["quality"] = q
        # variance: V — по vn узла, P/Q — по магнитуде
        if mt == 2:  # NODE V
            obj_node = int(arr[idx]["object_id"])
            vn = vn_by_node.get(obj_node, 220.0)
            var = _variance_voltage(float(arr[idx]["value"]), vn)
        elif mt == 1:  # BRANCH Q
            # Charging-aware σ_Q (единый источник — variance_branch_q): на
            # длинных HV/EHV-ВЛ с большой B legitimate Q ≈ V²·B; static σ_frac
            # недооценивает шум. σ_min = α·|B|·Vn² (default α=0.10) → Q-замеры
            # 750 кВ ВЛ получают σ ≈ 50–100 МВар. См. memory
            # `odu_q_sigma_calibration`; Цех-3 sweep подтвердил α=0.10 как
            # универсальный оптимум на 4 региональных моделях.
            vn_q = branch_vn.get(int(obj_id), 0.0) if ot == 1 else 0.0
            b_si_q = branch_b.get(int(obj_id), 0.0) if ot == 1 else 0.0
            var = variance_branch_q(
                float(arr[idx]["value"]),
                charging_mvar=abs(b_si_q) * vn_q * vn_q,
                charging_alpha=branch_q_sigma_charging_alpha,
                sigma_frac=branch_q_sigma_frac,
            )
        else:  # BRANCH P (mt=0)
            var = _variance_p(float(arr[idx]["value"]))
        if int(arr[idx]["quality"]) == QUALITY_QUESTIONABLE:
            var *= questionable_sigma2_multiplier
        # Q-inconsistent на HV-ВЛ — downweight σ² × factor (loose якорь,
        # как в эталонном OC с low weight). Применяется после расчёта
        # default σ.
        # Источник: либо новый physical filter (action="downweight"),
        # либо старый flat-detector (action_hv="downweight").
        if kind in ("QBEG", "QEND") and obj_id in q_inconsistent_hv_branches:
            if q_loss_filter_enabled and q_loss_filter_action == "downweight":
                var *= q_loss_filter_downweight_factor
            elif q_inconsistency_action_hv == "downweight":
                var *= q_inconsistency_downweight_factor
            arr[idx]["filter_flag"] = 2  # q_inconsistency mark
        arr[idx]["variance"] = float(var)
        arr[idx]["weight"] = 1.0 / float(var)
        if int(arr[idx]["quality"]) == QUALITY_QUESTIONABLE:
            stats["applied_questionable"] += 1
        else:
            stats["applied"] += 1

    # Соберём NODE injections для последующего .add() адаптером. Локальная
    # sigma 5% (не 2% как JSON-TI) — net inj это агрегированная величина
    # (PG+PN с разными ARG), уверенность ниже чем у направленного branch
    # flow. Tighter sigma (как в `_variance_power` 2%) эмпирически ломала
    # convergence. id назначаются здесь (бит-идентично прежнему .add()).
    new_rows: list[dict] = []
    next_id = int(arr["id"].max()) + 1 if len(arr) else 1
    for (obj_id, pq), entries in inj_acc.items():
        net = sum(v for v, _, _ in entries)
        guid_first = entries[0][1]
        net_q = aggregate_qualities([q for _, _, q in entries])
        mt = _INJ_MT[pq]
        sigma = max(0.05 * abs(net), 0.5)
        var = sigma * sigma + (1.0 if pq == "P" else 0.5)
        if net_q == QUALITY_QUESTIONABLE:
            var *= questionable_sigma2_multiplier
        new_rows.append(
            {
                "id": next_id,
                "object_type": 0,
                "object_id": int(obj_id),
                "measurement_type": mt,
                "branch_side": -1,
                "value": float(net),
                "variance": float(var),
                "weight": 1.0 / float(var),
                "status": True,
                "quality": int(net_q),
                "source_guid": guid_first,
            }
        )
        next_id += 1
        stats["node_inj_added"] += 1
        if net_q == QUALITY_QUESTIONABLE:
            stats["applied_questionable"] += 1

    return stats, new_rows


def _val_twin(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def assign_cod_from_xml(
    model,
    args: dict[tuple[int, str], FormulaSpec],
    *,
    dup_val_rtol: float = 0.01,
    dup_val_atol: float = 0.5,
    apply: str = "mark",
    representative: str = "none",
) -> dict[str, int]:
    """Пометить дубли замеров «один SCADA-сигнал на несколько объектов».

    Один физический SCADA-сигнал может зеркалиться на 2+ объекта модели с равным
    значением; группа-ключ — поле ``numer`` первого ARG привязки. Здесь активные
    V/branch-меры группируются по ``numer``: если один ``numer`` стоит на ≥2 РАЗНЫХ
    объектах с value-twin (равным значением) — это дубль-кластер. По умолчанию все
    копии помечаются ``filter_flag=7`` («дубль»); опционально одна копия
    (минимальный object_id) оставляется представителем.

    Узловые инжекции (PN/PG/QN/QG) НЕ обрабатываются: они агрегируются в один
    P_inj/Q_inj на узел (нет одиночного ``numer``), а дубли «генерация дважды» уже
    нейтрализованы ``aggregate_generators_to_node`` + неттингом. Здесь — V и
    branch-flow.

    Args:
        model: модель с активными measurements (после применения телеметрии).
        args: ``dict[(object_id, kind) → FormulaSpec]`` (``ArgEntry`` несёт ``numer``).
        dup_val_rtol, dup_val_atol: допуск равенства value для value-twin.
        apply: ``"mark"`` — только ``filter_flag=7`` (статус не трогаем); ``"drop"`` —
            дополнительно ``status=False`` у дублей.
        representative: ``"none"`` (default) — метить ВСЕ копии value-twin кластера;
            ``"min_object"`` — оставить одну копию (минимальный object_id) как якорь.

    Returns:
        ``{"clusters": N, "marked_cod7": N, "dropped": N, "groups_numer": N}``.
    """
    arr = model.measurements.to_numpy().copy()
    # карта (ot, mt, side, oid) → primary NUMER (только V + branch-flow kinds)
    numer_by_key: dict[tuple[int, int, int, int], str] = {}
    for (oid, kind), spec in args.items():
        if kind not in _KIND_MAP or not spec.args:
            continue
        nmr = (spec.args[0].numer or "").strip()
        if not nmr:
            continue
        ot, mt, side, _sign = _KIND_MAP[kind]
        numer_by_key[(ot, mt, side, int(oid))] = nmr

    # активные меры с известным NUMER → группировка
    groups: dict[str, list[int]] = defaultdict(list)  # numer → row idx
    for i in range(len(arr)):
        if not bool(arr[i]["status"]):
            continue
        if "is_pseudo" in arr.dtype.names and bool(arr[i]["is_pseudo"]):
            continue
        key = (
            int(arr[i]["object_type"]),
            int(arr[i]["measurement_type"]),
            int(arr[i]["branch_side"]),
            int(arr[i]["object_id"]),
        )
        nmr = numer_by_key.get(key)
        if nmr:
            groups[nmr].append(i)

    stats = {"clusters": 0, "marked_cod7": 0, "dropped": 0, "groups_numer": len(groups)}
    for _nmr, idxs in groups.items():
        if len(idxs) < 2:
            continue
        # требуем ≥2 РАЗНЫХ объекта (cross-object) + value-twin
        by_obj: dict[int, int] = {}
        for i in idxs:
            by_obj.setdefault(int(arr[i]["object_id"]), i)
        if len(by_obj) < 2:
            continue
        members = list(by_obj.values())
        v0 = float(arr[members[0]]["value"])
        if not all(
            _val_twin(v0, float(arr[i]["value"]), dup_val_rtol, dup_val_atol) for i in members
        ):
            continue
        # representative="none" → метить ВСЕ копии (как в исходной OC); иначе оставить
        # одну (min object_id) как cod=0-якорь.
        rep = (
            None
            if representative == "none"
            else min(members, key=lambda i: int(arr[i]["object_id"]))
        )
        stats["clusters"] += 1
        for i in members:
            if i == rep:
                continue
            arr[i]["filter_flag"] = 7  # cod=7 «Дубль замера»
            stats["marked_cod7"] += 1
            if apply == "drop":
                arr[i]["status"] = False
                stats["dropped"] += 1
    model.measurements.update_from_array(arr)
    return stats


def _materialize_area_on_arrays(
    nodes_arr,
    obs: dict[int, float],
    *,
    max_col: str,
    min_col: str,
    exist_col: str,
    set_col: str,
    max_cap: float,
    skip_neg_min: float,
    global_k_fallback: float,
    fill: bool = True,
) -> dict[str, float]:
    """ЯДРО (CLASS-2): материализация одного поля узла над ``SE_INPUT``-массивом.

    Контрактная float-математика: читает ``status``/``id``/``area_id``/
    ``<max_col>``/``<min_col>``/``<exist_col>`` из ``nodes_arr`` (структурный
    ``SE_INPUT.nodes.input_dtype()``) и мутирует ``nodes_arr[<set_col>]`` in place.
    Наблюдаемый режим резолвится вне ядра — сюда приходит уже готовый числовой
    ``obs`` (``{node_id → value}``), как ``resolved_taps`` в apply_rpn.

    * **наблюдаемые** (``nid in obs``) → ``nodes_arr[set_col][i] = obs[nid]`` вербатим;
    * **ненаблюдаемые** с ``<exist_col>`` → (если ``fill``) ``clamp(k_area·<max_col>)``,
      где ``k_area = median(obs/<max_col>)`` по наблюдаемым несентинельным узлам
      района (``area_id``); fallback ``global_k_fallback``.

    Sentinel-санация (для k-калибровки и fill): ``0 < max ≤ max_cap`` и
    ``min ≥ skip_neg_min``.

    ``fill=False`` → только вербатим (для генерации: area-разнос ``k·gen_max``
    бимодален и ненадёжен — генератор либо диспетчеризован, либо нет; A/B даёт
    на региональной модели WLS dV_max 0.25→0.43). Богатой PG-телеметрии вербатим достаточно.

    Returns:
        ``{"n_obs", "n_fill", "sum"}`` (``sum`` округлён до 0.1; узловые значения,
        питающие солвер, пишутся per-node и от порядка итерации не зависят).
    """
    obs_ids = set(obs)
    status = nodes_arr["status"]
    ids = nodes_arr["id"]
    areas = nodes_arr["area_id"]
    maxv = nodes_arr[max_col]
    minv = nodes_arr[min_col]
    exist = nodes_arr[exist_col]
    setv = nodes_arr[set_col]
    n = len(nodes_arr)

    def _ok(wmax: float, wmin: float) -> bool:
        return 0.0 < wmax <= max_cap and wmin >= skip_neg_min

    k_area: dict[int, float] = {}
    glob = global_k_fallback
    if fill:
        rat: dict[int, list[float]] = defaultdict(list)
        for i in range(n):
            if not bool(status[i]):
                continue
            nid = int(ids[i])
            if nid not in obs_ids:
                continue
            wmax = float(maxv[i])
            wmin = float(minv[i])
            if _ok(wmax, wmin):
                rat[int(areas[i])].append(obs[nid] / wmax)
        k_area = {a: float(median(v)) for a, v in rat.items() if v}
        all_rat = [r for v in rat.values() for r in v]
        glob = float(median(all_rat)) if all_rat else global_k_fallback

    cnt = {"n_obs": 0, "n_fill": 0, "sum": 0.0}
    for i in range(n):
        if not bool(status[i]):
            continue
        nid = int(ids[i])
        if nid in obs_ids:
            setv[i] = float(obs[nid])
            cnt["n_obs"] += 1
            cnt["sum"] += float(obs[nid])
        elif fill and int(exist[i]):
            wmax = float(maxv[i])
            wmin = float(minv[i])
            if _ok(wmax, wmin):
                a = int(areas[i])
                v = min(max(k_area.get(a, glob) * wmax, wmin), wmax)
                setv[i] = float(v)
                cnt["n_fill"] += 1
                cnt["sum"] += v
    cnt["sum"] = round(cnt["sum"], 1)
    return cnt


def apply_materialize_resolved(
    model,
    obs: dict[str, dict[int, float]],
    *,
    fill_q: bool = True,
    materialize_generation: bool = True,
    materialize_generation_fill: bool = False,
    max_load: float = 5e4,
    max_gen: float = 5e4,
    skip_neg_min: float = -100.0,
    global_k_fallback: float = 0.40,
    gen_global_k_fallback: float = 0.80,
) -> dict[str, dict[str, float]]:
    """Применить готовый наблюдаемый узловой режим ``obs`` к узлам модели.

    Чистое применение (без snapshot/формул): ``obs`` (``{"load_p"/"load_q"/"gen_p"/
    "gen_q" → {node_id → value}}``, готовый числовой план наблюдаемых инжекций) →
    контрактные ядра :func:`_materialize_area_on_arrays` → write-back. Зовётся шагом
    ``run()`` на своей позиции (после ``aggregate_generators``: 4 поля пишутся per-node,
    generation_* перекрывает агрегацию — порядок сохранён).

    Контрактный снимок узлов: 4 поля материализуются на ОДНОМ массиве (``set_col`` у
    каждого вызова свой — load_p/load_q/generation_p/generation_q; k-калибровка читает
    только max/min/exist-колонки, которые не мутируются → вызовы независимы), затем
    единый write-back.
    """
    nodes_arr = model.nodes.to_numpy().copy()
    out: dict[str, dict[str, float]] = {}
    out["load_p"] = _materialize_area_on_arrays(
        nodes_arr,
        obs["load_p"],
        max_col="load_p_max",
        min_col="load_p_min",
        exist_col="exist_load",
        set_col="load_p",
        max_cap=max_load,
        skip_neg_min=skip_neg_min,
        global_k_fallback=global_k_fallback,
    )
    if fill_q:
        out["load_q"] = _materialize_area_on_arrays(
            nodes_arr,
            obs["load_q"],
            max_col="load_q_max",
            min_col="load_q_min",
            exist_col="exist_load",
            set_col="load_q",
            max_cap=max_load,
            skip_neg_min=skip_neg_min,
            global_k_fallback=global_k_fallback,
        )
    if materialize_generation:
        out["gen_p"] = _materialize_area_on_arrays(
            nodes_arr,
            obs["gen_p"],
            max_col="generation_p_max",
            min_col="generation_p_min",
            exist_col="exist_gen",
            set_col="generation_p",
            max_cap=max_gen,
            skip_neg_min=-1e18,  # generation_*_min<0 норма, не boundary-sentinel
            global_k_fallback=gen_global_k_fallback,
            fill=materialize_generation_fill,
        )
        if fill_q:
            out["gen_q"] = _materialize_area_on_arrays(
                nodes_arr,
                obs["gen_q"],
                max_col="generation_q_max",
                min_col="generation_q_min",
                exist_col="exist_gen",
                set_col="generation_q",
                max_cap=max_gen,
                skip_neg_min=-1e18,
                global_k_fallback=gen_global_k_fallback,
                fill=materialize_generation_fill,
            )
    model.nodes.update_from_array(nodes_arr)
    return out
