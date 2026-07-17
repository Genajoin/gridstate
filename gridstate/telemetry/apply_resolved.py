"""Применение предвычисленных числовых планов телеметрии/материализации к режиму.

Контрактные ядра z-вектора (``_apply_telemetry_on_arrays``) и материализации режима
(``_materialize_area_on_arrays``) + тонкие адаптеры применения готовых числовых планов
(``apply_telemetry_resolved`` / ``apply_materialize_resolved``). Планы вычислены вне
ядра производителем данных; здесь — только применение к массивам ``SE_INPUT`` (без
формат-слоя источника). Kind-карты замеров (``_KIND_MAP`` / ``_NODE_INJ_MAP`` /
``_INJ_MT``) — в ``gridstate.telemetry._specs``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from gridstate.constants import FilterFlag
from gridstate.telemetry._filters import build_measurement_index
from gridstate.telemetry._specs import _INJ_MT, _KIND_MAP, _NODE_INJ_MAP
from gridstate.telemetry.loss_filter import is_branch_q_consistent_with_physics
from gridstate.telemetry.quality import QUALITY_BAD, QUALITY_QUESTIONABLE, aggregate_qualities
from gridstate.telemetry.units import variance_branch_q, variance_power, variance_voltage


if TYPE_CHECKING:
    from gridstate.working import Working


@dataclass(frozen=True)
class TelemetryApplyConfig:
    """Tunable thresholds/factors for :func:`_apply_telemetry_on_arrays`.

    Single source of truth for the defaults that were previously duplicated
    between the public wrapper :func:`apply_telemetry_resolved` and the core.
    The wrapper keeps its individual keyword arguments for backward
    compatibility and assembles this config to hand down.
    """

    questionable_sigma2_multiplier: float = 100.0
    v_sigma2_scale_by_node: dict[int, float] | None = None
    flow_sigma2_scale_by_branch: dict[tuple[int, str], float] | None = None
    branch_p_sigma_frac: float = 0.02
    branch_q_sigma_frac: float = 0.07
    branch_q_sigma_charging_alpha: float = 0.10
    sign_inconsistency_threshold_mw: float | None = 100.0
    q_inconsistency_threshold_mvar: float | None = None
    q_inconsistency_high_voltage_kv: float = 500.0
    q_inconsistency_threshold_mvar_hv: float | None = None
    q_inconsistency_action_hv: str = "drop"
    q_inconsistency_downweight_factor: float = 100.0
    q_loss_filter_enabled: bool = True
    q_loss_filter_floor_mvar: float = 50.0
    q_loss_filter_rel_pct: float = 30.0
    q_loss_filter_action: str = "downweight"
    q_loss_filter_downweight_factor: float = 100.0


def apply_telemetry_resolved(
    model: Working,
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
    *,
    total_args: int,
    questionable_sigma2_multiplier: float = 100.0,
    v_sigma2_scale_by_node: dict[int, float] | None = None,
    flow_sigma2_scale_by_branch: dict[tuple[int, str], float] | None = None,
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
    """Применить готовый ``resolved`` z-вектор к ``measurements``.

    Чистое применение (без XML/snapshot/формул): снимок ``measurements``/``nodes``/
    ``branches`` → ядро :func:`_apply_telemetry_on_arrays` → write-back + ``.add()``
    NODE-инжекций. Зовётся шагом ``run()`` на своей позиции (после ``apply_topology``/
    ``apply_rpn``: ядро читает branch ``status``/``susceptance``).
    """
    config = TelemetryApplyConfig(
        questionable_sigma2_multiplier=questionable_sigma2_multiplier,
        v_sigma2_scale_by_node=v_sigma2_scale_by_node,
        flow_sigma2_scale_by_branch=flow_sigma2_scale_by_branch,
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
        config=config,
    )

    model.measurements.update_from_array(meas_arr)
    # Пакетная вставка за одну конкатенацию (per-row .add() = O(n²)).
    model.measurements.add_many(new_rows)

    return stats


def _detect_sign_inconsistent_branches(
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    threshold_mw: float | None,
) -> set[int]:
    """Pre-pass: распознать ветви с sign-inconsistent P-измерениями
    (PBEG и PEND оба входят в линию, т.е. одного знака после
    учёта invert) — это указывает на битую привязку меры к не той
    ветви или на устаревшую телеметрию. Сами branch-meas таких ветвей
    не активируем.
    """
    inconsistent_branches: set[int] = set()
    if threshold_mw is None:
        return inconsistent_branches
    for obj_id, kind in arg_keys:
        if kind != "PBEG":
            continue
        if (obj_id, "PEND") not in resolved:
            continue
        v_beg = resolved[(obj_id, "PBEG")][0]
        v_end = resolved[(obj_id, "PEND")][0]
        if v_beg is None or v_end is None:
            continue
        # PBEG/PEND-значения приходят с уже применённым знаком (инверсия
        # закодирована во входном числовом плане адаптером). На физически
        # согласованной ВЛ выполняется ``v_beg ≈ -v_end + потери``, т.е. они
        # **противоположных знаков**. Если |v_beg + v_end| превосходит порог —
        # это либо знак-аномалия, либо неправильная привязка.
        if abs(v_beg + v_end) >= threshold_mw:
            inconsistent_branches.add(int(obj_id))
    return inconsistent_branches


def _detect_q_inconsistent_branches(
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    branch_vn: dict[int, float],
    branch_b: dict[int, float],
    branch_x: dict[int, float],
    config: TelemetryApplyConfig,
) -> tuple[set[int], set[int]]:
    """Pre-pass для Q: проверка физической согласованности
    |Q_BEG + Q_END| с π-схемой ветви (B·V², X-loss).

    **Старый подход** (deprecated, default отключён): flat threshold
    `q_inconsistency_threshold_mvar` ложно дропает 750 кВ Q-меры с
    зарядной B (Q_charging ≈ 1066 МВар легитимны).

    **Новый подход** (`q_loss_filter_enabled=True`, default):
    сравнивает |Q_beg+Q_end| с расчётным ожиданием через V_flat=V_nom
    и branch.{susceptance, reactance}. Outlier если расхождение
    `|обс - exp| > max(floor, rel_pct/100 · exp)`. См.
    `gridstate.telemetry.loss_filter.is_branch_q_consistent_with_physics`.

    Возвращает ``(q_inconsistent_branches, q_inconsistent_hv_branches)`` —
    ветви для drop и ветви для downweight соответственно.
    """
    effective_hv = (
        config.q_inconsistency_threshold_mvar_hv
        if config.q_inconsistency_threshold_mvar_hv is not None
        else config.q_inconsistency_threshold_mvar
    )
    q_inconsistent_branches: set[int] = set()  # ветви для drop
    q_inconsistent_hv_branches: set[int] = set()  # ветви для downweight

    flat_filter_active = (
        config.q_inconsistency_threshold_mvar is not None
        or config.q_inconsistency_threshold_mvar_hv is not None
    )
    if not (flat_filter_active or config.q_loss_filter_enabled):
        return q_inconsistent_branches, q_inconsistent_hv_branches

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
        if config.q_loss_filter_enabled and vn > 0:
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
                floor_mvar=config.q_loss_filter_floor_mvar,
                rel_pct=config.q_loss_filter_rel_pct,
            )
            if not is_ok:
                if config.q_loss_filter_action == "downweight":
                    q_inconsistent_hv_branches.add(int(obj_id))
                else:
                    q_inconsistent_branches.add(int(obj_id))
            continue

        # Fallback: старый flat-detector.
        is_hv = vn >= config.q_inconsistency_high_voltage_kv
        threshold = effective_hv if is_hv else config.q_inconsistency_threshold_mvar
        if threshold is None:
            continue
        if abs(v_beg + v_end) >= threshold:
            if is_hv and config.q_inconsistency_action_hv == "downweight":
                q_inconsistent_hv_branches.add(int(obj_id))
            else:
                q_inconsistent_branches.add(int(obj_id))
    return q_inconsistent_branches, q_inconsistent_hv_branches


def _measurement_variance(
    meas_arr: np.ndarray,
    idx: int,
    *,
    mt: int,
    ot: int,
    obj_id: int,
    vn_by_node: dict[int, float],
    branch_vn: dict[int, float],
    branch_b: dict[int, float],
    config: TelemetryApplyConfig,
) -> float:
    """Variance for the just-written measurement, by measurement type.

    V uses the node nominal voltage; branch Q is charging-aware; branch P
    scales with magnitude.
    """
    # variance: V — по vn узла, P/Q — по магнитуде
    value = float(meas_arr[idx]["value"])
    if mt == 2:  # NODE V
        obj_node = int(meas_arr[idx]["object_id"])
        vn = vn_by_node.get(obj_node, 220.0)
        return variance_voltage(value, vn)
    if mt == 1:  # BRANCH Q
        # Charging-aware σ_Q (единый источник — variance_branch_q): на
        # длинных HV/EHV-ВЛ с большой B legitimate Q ≈ V²·B; static σ_frac
        # недооценивает шум. σ_min = α·|B|·Vn² (default α=0.10) → Q-замеры
        # 750 кВ ВЛ получают σ ≈ 50–100 МВар. См. memory
        # `odu_q_sigma_calibration`; Цех-3 sweep подтвердил α=0.10 как
        # универсальный оптимум на 4 региональных моделях.
        vn_q = branch_vn.get(int(obj_id), 0.0) if ot == 1 else 0.0
        b_si_q = branch_b.get(int(obj_id), 0.0) if ot == 1 else 0.0
        return variance_branch_q(
            value,
            charging_mvar=abs(b_si_q) * vn_q * vn_q,
            charging_alpha=config.branch_q_sigma_charging_alpha,
            sigma_frac=config.branch_q_sigma_frac,
        )
    # BRANCH P (mt=0)
    return variance_power(value, sigma_frac=config.branch_p_sigma_frac)


def _apply_measurement_loop(
    meas_arr: np.ndarray,
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    meas_idx: dict[tuple[int, int, int, int], int],
    vn_by_node: dict[int, float],
    branch_status_by_id: dict[int, bool],
    branch_vn: dict[int, float],
    branch_b: dict[int, float],
    inconsistent_branches: set[int],
    q_inconsistent_branches: set[int],
    q_inconsistent_hv_branches: set[int],
    config: TelemetryApplyConfig,
    stats: dict[str, int],
    inj_acc: dict[tuple[int, str], list[tuple[float, str, int]]],
) -> None:
    """Main application loop over ``arg_keys``.

    Mutates ``meas_arr`` (write-back of activated measurements), ``stats``
    (counters) and ``inj_acc`` (accumulated NODE injections) in place.
    """
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
                meas_arr[idx_f]["filter_flag"] = int(FilterFlag.P_SIGN_INCONSISTENCY)
            continue
        # Q-inconsistent ветви — только Q-меры пропускаем, P оставляем.
        if kind in ("QBEG", "QEND") and obj_id in q_inconsistent_branches:
            stats.setdefault("skipped_q_inconsistent", 0)
            stats["skipped_q_inconsistent"] += 1
            ot_f, mt_f, side_f, _ = _KIND_MAP[kind]
            idx_f = meas_idx.get((ot_f, mt_f, side_f, obj_id))
            if idx_f is not None:
                meas_arr[idx_f]["filter_flag"] = int(FilterFlag.Q_INCONSISTENCY)
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
            meas_arr[idx]["filter_flag"] = int(FilterFlag.BAD_QUALITY)
            continue
        val = value * sign
        # V-measurement < 50% Vnom — sentinel «датчик не работает» / отключённое
        # оборудование. Принимать его как жёсткий якорь нельзя: SE будет тянуть
        # V→0 на узле и в окрестности (см. audit_se_nonphysical_minimum.md). Не
        # активируем — добавит pseudo-V=Vnom через add_pseudo_measurements.
        if mt == 2 and ot == 0:
            obj_node = int(meas_arr[idx]["object_id"])
            vn = vn_by_node.get(obj_node, 220.0)
            if vn > 0 and abs(val) < 0.5 * vn:
                stats["skipped_v_below_half_nominal"] += 1
                meas_arr[idx]["filter_flag"] = int(FilterFlag.V_BELOW_HALF_NOMINAL)
                continue
        if meas_arr[idx]["status"]:
            meas_arr[idx]["value"] = float(meas_arr[idx]["value"]) + val
            # Worst-case аккумуляция quality при многократном попадании.
            meas_arr[idx]["quality"] = max(int(meas_arr[idx]["quality"]), q)
        else:
            meas_arr[idx]["value"] = val
            meas_arr[idx]["status"] = True
            meas_arr[idx]["source_guid"] = guid_first
            meas_arr[idx]["quality"] = q
        var = _measurement_variance(
            meas_arr,
            idx,
            mt=mt,
            ot=ot,
            obj_id=obj_id,
            vn_by_node=vn_by_node,
            branch_vn=branch_vn,
            branch_b=branch_b,
            config=config,
        )
        if int(meas_arr[idx]["quality"]) == QUALITY_QUESTIONABLE:
            var *= config.questionable_sigma2_multiplier
        # Точечный масштаб σ² V-меры узла из плана производителя данных
        # ({node_id: factor}; factor<1 усиливает доверие к мере — «якорный»
        # датчик, factor>1 ослабляет). Только узловые V (mt=2, ot=0).
        if mt == 2 and ot == 0 and config.v_sigma2_scale_by_node:
            scale = config.v_sigma2_scale_by_node.get(int(meas_arr[idx]["object_id"]))
            if scale is not None:
                var *= float(scale)
                stats.setdefault("v_sigma2_scaled", 0)
                stats["v_sigma2_scaled"] += 1
        # Аналогичный план для потоковых мер ветвей: {(branch_id, kind): factor},
        # kind ∈ PBEG/PEND/QBEG/QEND. factor>1 ослабляет доверие (мера-кандидат
        # в дефектные по данным производителя), factor<1 усиливает.
        if ot == 1 and config.flow_sigma2_scale_by_branch:
            scale = config.flow_sigma2_scale_by_branch.get((int(obj_id), kind))
            if scale is not None:
                var *= float(scale)
                stats.setdefault("flow_sigma2_scaled", 0)
                stats["flow_sigma2_scaled"] += 1
        # Q-inconsistent на HV-ВЛ — downweight σ² × factor (loose якорь,
        # как в эталонном OC с low weight). Применяется после расчёта
        # default σ.
        # Источник: либо новый physical filter (action="downweight"),
        # либо старый flat-detector (action_hv="downweight").
        if kind in ("QBEG", "QEND") and obj_id in q_inconsistent_hv_branches:
            if config.q_loss_filter_enabled and config.q_loss_filter_action == "downweight":
                var *= config.q_loss_filter_downweight_factor
            elif config.q_inconsistency_action_hv == "downweight":
                var *= config.q_inconsistency_downweight_factor
            meas_arr[idx]["filter_flag"] = int(FilterFlag.Q_INCONSISTENCY)  # q_inconsistency mark
        meas_arr[idx]["variance"] = float(var)
        meas_arr[idx]["weight"] = 1.0 / float(var)
        if int(meas_arr[idx]["quality"]) == QUALITY_QUESTIONABLE:
            stats["applied_questionable"] += 1
        else:
            stats["applied"] += 1


def _build_node_injection_rows(
    inj_acc: dict[tuple[int, str], list[tuple[float, str, int]]],
    meas_arr: np.ndarray,
    config: TelemetryApplyConfig,
    stats: dict[str, int],
) -> list[dict]:
    """Собрать NODE injections для последующего .add() адаптером. Локальная
    sigma 5% (не 2% как JSON-TI) — net inj это агрегированная величина
    (PG+PN с разными ARG), уверенность ниже чем у направленного branch
    flow. Tighter sigma (как в `_variance_power` 2%) эмпирически ломала
    convergence. id назначаются здесь (бит-идентично прежнему .add()).
    """
    new_rows: list[dict] = []
    next_id = int(meas_arr["id"].max()) + 1 if len(meas_arr) else 1
    for (obj_id, pq), entries in inj_acc.items():
        net = sum(v for v, _, _ in entries)
        guid_first = entries[0][1]
        net_q = aggregate_qualities([q for _, _, q in entries])
        mt = _INJ_MT[pq]
        sigma = max(0.05 * abs(net), 0.5)
        var = sigma * sigma + (1.0 if pq == "P" else 0.5)
        if net_q == QUALITY_QUESTIONABLE:
            var *= config.questionable_sigma2_multiplier
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

    return new_rows


def _apply_telemetry_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    *,
    total_args: int,
    config: TelemetryApplyConfig,
) -> tuple[dict[str, int], list[dict]]:
    """ЯДРО: применение телеметрии над контрактными numpy-массивами.

    Чистая работа над ``SE_INPUT.measurements`` (``meas_arr``, мутируется
    in-place) / ``SE_INPUT.nodes`` (``nodes_arr``) / ``SE_INPUT.branches``
    (``branches_arr``). БЕЗ внешних зависимостей, БЕЗ формул источника: все числовые
    значения мер уже посчитаны адаптером и переданы в ``resolved`` —
    ``{(obj_id, kind): (value|None, n_resolved, guid_first, quality)}``.
    Порядок итерации задаётся ``arg_keys`` (= ``list(args.keys())``).

    Возвращает ``(stats, new_rows)``: ``new_rows`` — список dict-ов для
    последующего ``model.measurements.add()`` (NODE-инжекции PG/PN/QG/QN);
    их ``id`` назначаются здесь, чтобы быть бит-идентичными прежнему
    in-loop ``.add()``-блоку.

    Tuning приходит через ``config`` (:class:`TelemetryApplyConfig`).
    """
    # Index measurements: (ot, mt, side, oid) → row idx in numpy array
    meas_idx = build_measurement_index(meas_arr)

    # node_id → voltage_nominal для расчёта sigma_V.
    vn_by_node: dict[int, float] = {
        int(i): float(v) for i, v in zip(nodes_arr["id"], nodes_arr["voltage_nominal"], strict=True)
    }
    # branch_id → status. Меру нельзя ставить на отключённую ветвь:
    # WLS получит residual P/Q≠0 на ветви где Y-bus её зануляет —
    # iter будет компенсировать через V/δ соседних узлов и портит
    # решение. Проверяется при активации branch-meas (ot=1).
    branch_status_by_id: dict[int, bool] = {
        int(i): bool(s) for i, s in zip(branches_arr["id"], branches_arr["status"], strict=True)
    }

    # Сначала отключим ВСЕ measurements (на входе они приходят со status=True
    # по умолчанию). После apply сделаем status=True только для тех, что
    # покрыты snapshot-ом.
    meas_arr["status"] = False

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

    # Per-branch nominal voltage / susceptance / reactance: питают Q-physics
    # pre-pass и charging-aware σ_Q. Vn берётся из узла начала ветви.
    branch_vn: dict[int, float] = {
        int(bid): vn_by_node.get(int(fn), 0.0)
        for bid, fn in zip(branches_arr["id"], branches_arr["from_node"], strict=True)
    }
    branch_b: dict[int, float] = {
        int(bid): float(b)
        for bid, b in zip(branches_arr["id"], branches_arr["susceptance"], strict=True)
    }
    branch_x: dict[int, float] = {
        int(bid): float(x)
        for bid, x in zip(branches_arr["id"], branches_arr["reactance"], strict=True)
    }

    inconsistent_branches = _detect_sign_inconsistent_branches(
        arg_keys, resolved, config.sign_inconsistency_threshold_mw
    )
    if config.sign_inconsistency_threshold_mw is not None:
        stats["sign_inconsistent_branches"] = len(inconsistent_branches)

    q_inconsistent_branches, q_inconsistent_hv_branches = _detect_q_inconsistent_branches(
        arg_keys, resolved, branch_vn, branch_b, branch_x, config
    )
    flat_filter_active = (
        config.q_inconsistency_threshold_mvar is not None
        or config.q_inconsistency_threshold_mvar_hv is not None
    )
    if flat_filter_active or config.q_loss_filter_enabled:
        stats["q_inconsistent_branches"] = len(q_inconsistent_branches)
        stats["q_inconsistent_hv_branches"] = len(q_inconsistent_hv_branches)

    _apply_measurement_loop(
        meas_arr,
        arg_keys,
        resolved,
        meas_idx,
        vn_by_node,
        branch_status_by_id,
        branch_vn,
        branch_b,
        inconsistent_branches,
        q_inconsistent_branches,
        q_inconsistent_hv_branches,
        config,
        stats,
        inj_acc,
    )

    new_rows = _build_node_injection_rows(inj_acc, meas_arr, config, stats)

    return stats, new_rows


def _materialize_area_on_arrays(
    nodes_arr: np.ndarray,
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
    """ЯДРО: материализация одного поля узла над ``SE_INPUT``-массивом.

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
    model: Working,
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
