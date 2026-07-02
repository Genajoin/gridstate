"""Единая библиотечная обёртка production-пайплайна SE.

Один вход для внешних потребителей (UI, CLI) вместо дублирования оркестрации в
каждом из них: на входе рабочий слой ``model`` (+ опционально предвычисленные
числовые планы ``derived``) и ``PipelineConfig``.

Три вещи в одном модуле:

* :class:`PipelineConfig` — dataclass со ВСЕМИ ручками; **дефолты здесь —
  единственный источник истины** (gridstate не зависит от pydantic).
* :data:`STEPS` — упорядоченный реестр шагов (имя, заголовок, группа, описание,
  toggle-поле). :func:`run` исполняет их по порядку.
* :func:`manifest` — JSON-сериализуемое описание (шаги + параметры с дефолтами,
  типами, ограничениями, label/help/group). UI строит форму из него — никакого
  хардкода списка функций/дефолтов на стороне UI/CLI.

Прогресс — через callback ``on_event(event: dict)`` (протокол совместим со
streaming-эвентами UI): ``step_start`` / ``step_done`` / ``step_skipped`` /
``step_error``. gridstate не зависит от веб-фреймворка — потребитель сам оборачивает
события в свой транспорт (NDJSON и т.п.).

Пример::

    from gridstate.contract.serialize import load_se_input_npz
    from gridstate.contract.runtime import run as contract_run
    from gridstate.pipeline import PipelineConfig, manifest

    se_input = load_se_input_npz("model.npz")
    result = contract_run(se_input, config=PipelineConfig())
    # UI:
    schema = manifest()   # {"steps": [...], "params": {...}, "groups": [...]}
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any

from gridstate.api import estimate, populate_quality_summary
from gridstate.bad_data_repass import apply_bad_data_plan, classify_bad_data
from gridstate.contract.derived import DerivedInputs
from gridstate.post_processing import refine_anti_overshoot
from gridstate.preprocessing import (
    add_pseudo_measurements,
    mirror_voltage_through_unit_tap_links,
    synthesize_node_injection_from_branch_flows,
)
from gridstate.result import SEResult
from gridstate.telemetry import (
    aggregate_generators_to_node,
    apply_flow_sigma_floor,
    apply_generator_status_from_node,
    apply_reactors_to_node_shunt,
    apply_voltage_meas_calibration_for_gen_nodes,
    apply_voltage_range_filter,
    deactivate_orphan_measurements,
    normalize_breaker_reactance,
    resolve_merged_measurement_conflicts,
)
from gridstate.telemetry.apply_resolved import apply_materialize_resolved, apply_telemetry_resolved
from gridstate.telemetry.on_line import apply_topology_resolved
from gridstate.telemetry.rpn import apply_rpn_resolved
from gridstate.telemetry.voltage_nominal import apply_voltage_nominal_resolved
from gridstate.topology import (
    disable_disconnected_components,
    disable_isolated_nodes,
    disable_orphan_branches,
    refine_node_types_from_generators,
    refine_slack_to_one,
)
from gridstate.v_mirror import apply_v_mirror_plan, classify_v_mirror
from gridstate.v_refine import apply_v_refine_plan, classify_v_refine
from gridstate.working import Working


# ---------------------------------------------------------------------------
# Конфигурация (dataclass — единственный источник истины по дефолтам)
# ---------------------------------------------------------------------------
#
# Каждое поле несёт ``metadata`` для UI-манифеста:
#   kind:    "toggle" (булев шаг вкл/выкл) | "param" (скаляр)
#   group:   секция формы
#   label:   человекочитаемая подпись
#   help:    подсказка
#   control: подсказка контролу UI ("switch"/"number"/"select")
#   min/max: границы (для number); choices: список (для select)
#   depends: показывать только если другое поле == значению (напр. method=ipm)


def _toggle(default: bool, *, group: str, label: str, help: str = "") -> Any:
    return field(
        default=default,
        metadata={
            "kind": "toggle",
            "control": "switch",
            "group": group,
            "label": label,
            "help": help,
        },
    )


def _param(
    default: Any,
    *,
    group: str,
    label: str,
    control: str,
    help: str = "",
    min: Any = None,
    max: Any = None,
    choices: list | None = None,
    depends: dict | None = None,
) -> Any:
    md: dict[str, Any] = {
        "kind": "param",
        "control": control,
        "group": group,
        "label": label,
        "help": help,
    }
    if min is not None:
        md["min"] = min
    if max is not None:
        md["max"] = max
    if choices is not None:
        md["choices"] = choices
    if depends is not None:
        md["depends"] = depends
    return field(default=default, metadata=md)


_G_XML = "Препроцессинг входа"
_G_MODE = "Режим (нагрузка/генерация)"
_G_CASCADE = "Каскад и статусы"
_G_PSEUDO = "Псевдо-измерения"
_G_EST = "Оценка состояния"
_G_IPM = "IPM"
_G_POST = "Пост-обработка (Линия C)"


@dataclass
class PipelineConfig:
    """Все ручки пайплайна. Дефолты = валидированный production (canon)."""

    # --- XML-препроцессинг ---
    normalize_breakers: bool = _toggle(
        True,
        group=_G_XML,
        label="Нормализация X короткозамыкателей",
        help="volt-aware X_pu для R=X=0 ветвей (убирает β-выбросы dV_max на LV).",
    )
    apply_voltage_nominal: bool = _toggle(
        False,
        group=_G_XML,
        label="Vnom (номинальное напряжение)",
        help="Заполнить voltage_nominal=0 узлов из готового плана Vnom. Обычно vn уже "
        "задан на входе; default OFF.",
    )
    apply_topology: bool = _toggle(
        True,
        group=_G_XML,
        label="Топология (ON_LINE статусы)",
        help="Применить готовый план ON_LINE-статусов → status узлов/ветвей/ген/реакторов.",
    )
    apply_rpn: bool = _toggle(
        True,
        group=_G_XML,
        label="РПН (№ отпаек)",
        help="Применить готовый план № отпаек → динамический tap_ratio/phase_shift.",
    )
    apply_reactors: bool = _toggle(
        True,
        group=_G_XML,
        label="Реакторы → шунт узла",
        help="apply_reactors_to_node_shunt (ШР в shunt_b). Физически необходим "
        "для длинных ВЛ 500/750 кВ.",
    )
    apply_voltage_range_filter: bool = _toggle(
        True,
        group=_G_XML,
        label="Фильтр V вне диапазона",
        help="downweight V вне voltage_critical/max+10%.",
    )
    resolve_merged_conflicts: bool = _toggle(
        True,
        group=_G_XML,
        label="Слить дубли измерений",
        help="resolve_merged_measurement_conflicts: слить дубли замеров на одном объекте.",
    )
    aggregate_generators: bool = _toggle(
        True,
        group=_G_XML,
        label="Агрегировать генераторы к узлу",
        help="aggregate_generators_to_node: multi-gen → узловая генерация.",
    )
    apply_gen_v_calibration: bool = _toggle(
        True,
        group=_G_XML,
        label="Калибровка σ² V ген-узлов",
        help="apply_voltage_meas_calibration_for_gen_nodes.",
    )
    flow_sigma_floor_kv_frac: float | None = _param(
        None,
        group=_G_XML,
        label="σ-floor flow-мер (доля шкалы)",
        control="number",
        min=0.0,
        max=1.0,
        help="σ_min real branch-flow мер P/Q = доля шкалы канала √3·Vn·1кА "
        "(Vn ветви = max(Vn концов)): variance := max(variance, floor²). "
        "Лечит пере-доверие мелким потокам (σ≈α·|z| занижена при малых z). "
        "None = выключено; 0.010 = 1 % шкалы (110 кВ → 1.9 МВт, 500 кВ → 8.7 МВт).",
    )

    # --- Режим ---
    materialize: bool = _toggle(
        True,
        group=_G_MODE,
        label="Материализация режима",
        help="Применить наблюдаемый узловой режим node.pn/qn/pg/qg "
        "(нагрузка по районам + генерация вербатим). Кормит IPM box-init.",
    )

    # --- Каскад / статусы ---
    refine_slack: bool = _toggle(
        True,
        group=_G_CASCADE,
        label="Выбор slack (refine_slack_to_one)",
        help="Родная семантика эталонной SE (НЕ KOCMOC-НБУ, тот deprecated).",
    )
    refine_node_types: bool = _toggle(True, group=_G_CASCADE, label="Типы узлов из генераторов")
    disable_orphan_branches: bool = _toggle(
        True,
        group=_G_CASCADE,
        label="Гасить orphan-ветви (×2, H46)",
        help="disable_orphan_branches до и после disconnected_components (каскад H46).",
    )
    disable_disconnected: bool = _toggle(
        True, group=_G_CASCADE, label="Гасить отсоединённые компоненты"
    )
    disable_isolated: bool = _toggle(True, group=_G_CASCADE, label="Гасить изолированные узлы")
    apply_generator_status: bool = _toggle(
        True,
        group=_G_CASCADE,
        label="Статус генераторов от узла",
        help="apply_generator_status_from_node (node off ⇒ gen off).",
    )
    deactivate_orphan_measurements: bool = _toggle(
        True, group=_G_CASCADE, label="Деактивировать orphan-измерения"
    )

    # --- Псевдо-измерения ---
    synthesize_injections: bool = _toggle(
        True,
        group=_G_PSEUDO,
        label="Синтез P/Q_inj из потоков",
        help="synthesize_node_injection_from_branch_flows на терминалах без real-meas.",
    )
    mirror_voltage_unit_tap: bool = _toggle(
        True,
        group=_G_PSEUDO,
        label="Зеркалить V через ktr=1 связи",
        help="wye-точки 3-обм АТ получают V-якорь HV-шины.",
    )
    add_pseudo: bool = _toggle(
        True,
        group=_G_PSEUDO,
        label="Добавить псевдо-приоры",
        help="add_pseudo_measurements: слабые V/P_inj/Q_inj приоры от недонаблюдаемости.",
    )
    unobservable_v_sigma_frac: float = _param(
        0.02,
        group=_G_PSEUDO,
        label="σ V ненабл. узлов (доля)",
        control="number",
        min=0.001,
        max=1.0,
        help="unobservable_v_sigma_frac.",
    )
    unobservable_v_min_vm_deviation: float = _param(
        0.01,
        group=_G_PSEUDO,
        label="Порог V-якоря (доля Vnom)",
        control="number",
        min=0.0,
        max=0.5,
        help="Гейт жёсткого V-якоря: на flat-XML (vm=vn) → no-op. См. cex3_pseudov.",
    )

    # --- Оценка состояния ---
    algorithm: str = _param(
        "wls",
        group=_G_EST,
        label="Алгоритм",
        control="select",
        choices=["wls", "ipm"],
        help="wls — Gauss-Newton (надёжно сходится); ipm — primal log-barrier с box-vars.",
    )
    init: str = _param(
        "flat",
        group=_G_EST,
        label="Начальное приближение",
        control="select",
        choices=["flat", "results", "slack"],
        help="flat (V=1,δ=0) | results | slack.",
    )
    tolerance: float = _param(
        1e-3,
        group=_G_EST,
        label="Допуск сходимости",
        control="number",
        min=1e-8,
        max=1.0,
        help="default региональной модели 1e-3.",
    )
    max_iterations: int = _param(
        80,
        group=_G_EST,
        label="Макс. итераций",
        control="number",
        min=1,
        max=500,
        help="default региональной модели 80.",
    )
    kkt_solver: str = _param(
        "auto",
        group=_G_EST,
        label="KKT-солвер",
        control="select",
        choices=["auto", "cholmod", "scipy"],
        help="Решатель Newton-систем: cholmod — CHOLMOD через cvxopt с реюзом "
        "символьной факторизации (×8-11 на крупных моделях); scipy — spsolve "
        "(прежнее поведение бит-в-бит); auto — cholmod при установленном cvxopt.",
    )
    huber_c: float | None = _param(
        None,
        group=_G_EST,
        label="Huber c (SHGM-IRLS)",
        control="number",
        min=0.0,
        max=100.0,
        help="None = авто (1.5 для wls, 2.0 для ipm). >0 включает робастный downweight.",
    )
    top_residuals_n: int = _param(
        20,
        group=_G_EST,
        label="Топ-N худших невязок",
        control="number",
        min=0,
        max=200,
        help="Размер worst_residuals/worst_imbalance в quality summary (bad-data панель).",
    )

    # --- IPM ---
    ipm_balance_weight_factor: float = _param(
        0.1,
        group=_G_IPM,
        label="Вес balance-pseudo",
        control="number",
        min=1e-4,
        max=10.0,
        depends={"algorithm": "ipm"},
        help="Меньше → balance мягче. default 0.1.",
    )
    ipm_bound_relax: float = _param(
        0.0,
        group=_G_IPM,
        label="Релакс границ box",
        control="number",
        min=0.0,
        max=1.0,
        depends={"algorithm": "ipm"},
        help="Расширяет [lo,hi] на долю (hi-lo).",
    )
    ipm_prior_sigma2_bus_equiv_pu: float = _param(
        0.01,
        group=_G_IPM,
        label="σ² prior BUS-эквив.",
        control="number",
        min=0.0,
        max=10.0,
        depends={"algorithm": "ipm"},
        help="default 0.01 p.u.².",
    )

    # --- Пост-обработка (Линия C) ---
    bad_data: bool = _toggle(
        False,
        group=_G_POST,
        label="Bad-data re-pass (двухпроходный)",
        help="Классификация real-мер по остаткам решённого SE (flip знак-флипов, "
        "reject битых нулей/монстров, демпф Qinj) + повторный warm-solve. "
        "Парный иммунитет согласованных branch-пар и guard покрытия (node, домен) "
        "защищают честные меры. Конфигурация валидирована на 4 ОДУ. Default OFF "
        "(меняет решение и время; включается потребителем).",
    )
    bad_data_threshold: float = _param(
        10.0,
        group=_G_POST,
        label="Порог T (на σ_det)",
        control="number",
        min=1.0,
        max=100.0,
        depends={"bad_data": True},
        help="Кандидат: |z−h|/σ_det > T. Держатели глубоких ям имеют rn_det 10-15 — "
        "T>10 их пропускает.",
    )
    bad_data_sigma_cap: float = _param(
        30.0,
        group=_G_POST,
        label="Cap детекционной σ (МВт/МВАр)",
        control="number",
        min=0.0,
        max=1000.0,
        depends={"bad_data": True},
        help="σ_det = min(σ, cap): снимает маскировку конфликтов гигантской σ≈α·|z| "
        "больших потоков. Веса солвера не меняются. ≤0 — без капа.",
    )
    bad_data_flip_ratio: float = _param(
        0.33,
        group=_G_POST,
        label="γ flip-критерия",
        control="number",
        min=0.0,
        max=1.0,
        depends={"bad_data": True},
        help="flip вместо reject, если |z+h| < γ·|z−h| (относительный критерий: "
        "у ямы h ≠ −z точно).",
    )
    bad_data_damp_factor: float = _param(
        5.0,
        group=_G_POST,
        label="Демпф Qinj (σ × k)",
        control="number",
        min=1.0,
        max=100.0,
        depends={"bad_data": True},
        help="Qinj-кандидаты не отключаются, а демпфируются: variance × k². "
        "Снятие Qinj со слепого узла сажает его на мусорный pseudo-приор.",
    )
    v_refine: bool = _toggle(
        False,
        group=_G_POST,
        label="V-refine (ужесточение согласованных V)",
        help="Двухпроходный: ужесточить σ (× factor) real V-мер, согласованных "
        "с решением первого прохода (|z−h|/σ < rn), + warm re-solve. Лечит "
        "глобальный bias занижения V (оценка провисает ниже собственных V-мер). "
        "Конфликтные V-меры (битый замер кромки) НЕ трогаются. Значения/статусы "
        "не меняются — только дисперсии, грубую ошибку внести не может. "
        "Валидирован на 4 ОДУ. Default OFF (включается потребителем).",
    )
    v_refine_rn: float = _param(
        3.0,
        group=_G_POST,
        label="Порог согласованности rn",
        control="number",
        min=1.0,
        max=100.0,
        depends={"v_refine": True},
        help="Ужесточаем V-меру, если |z−h|/σ < rn (согласована с решением). "
        "Выше порога — кандидат в грубую ошибку, оставляем рыхлой.",
    )
    v_refine_factor: float = _param(
        0.7,
        group=_G_POST,
        label="Множитель σ (factor)",
        control="number",
        min=0.05,
        max=1.0,
        depends={"v_refine": True},
        help="variance := variance · factor² (σ × factor). 0.7 — оптимум 4 ОДУ; "
        "0.5 точнее для bias, но ломает class-max региона с битой V-кромкой.",
    )
    v_mirror: bool = _toggle(
        False,
        group=_G_POST,
        label="V-mirror (значение pseudo-V слепых кластеров)",
        help="Двухпроходный: pseudo-V-плейсхолдеры (value=Vnom) слепых кластеров "
        "без real-TM переставить в median pu границы ТОГО ЖЕ класса × Vnom (лишь "
        "узлы систематически ниже границы — lift-гейт), + warm re-solve. Лечит "
        "провисание слепых хвостов к номиналу (OC держит их на уровне границы). "
        "Меняется только значение приора, σ не трогается. Валидирован на 4 ОДУ "
        "(Восток p50 −13%/p95 −16%). Default OFF (включается потребителем).",
    )
    v_mirror_max_pu_dev: float = _param(
        0.25,
        group=_G_POST,
        label="Гейт границы |pu−1|",
        control="number",
        min=0.01,
        max=1.0,
        depends={"v_mirror": True},
        help="Кластер перетираем, только если median(V/Vnom) границы в пределах "
        "[1−d, 1+d]. Отбрасывает мусорную границу с диким уровнем.",
    )
    v_mirror_min_lift: float = _param(
        0.01,
        group=_G_POST,
        label="Lift-гейт (pu граница − pu узел)",
        control="number",
        min=0.0,
        max=0.5,
        depends={"v_mirror": True},
        help="Трогаем узел, только если его решение ниже границы своего класса "
        "более чем на min_lift (в pu). Узлы уже-на-уровне не двигаем — иначе их "
        "подъём пушит наблюдаемую сеть вверх (реальный-vm регион регрессирует).",
    )
    v_mirror_cross_at: bool = _toggle(
        False,
        group=_G_POST,
        label="V-mirror через АТ (cross-class)",
        help="Когда у слепого кластера НЕТ границы того же класса, но есть граница "
        "через active trafo (branch_type=1, tap>0) — взять median(pu решённой "
        "trafo-границы)·Vnom (pu-инвариант к идеальному tap). lift/max_pu_dev-гейты "
        "сохраняются. Default OFF: pu-инвариант через tap≈2 даёт 2-4% остаточную "
        "ошибку, Юг-регрессия не исключена. Включать после A/B 4 ОДУ.",
    )
    anti_overshoot: bool = _toggle(
        True,
        group=_G_POST,
        label="Anti-overshoot уточнение",
        help="refine_anti_overshoot со само-валидацией: гасит нефизичный overshoot V "
        "на слабонаблюдаемых radial-узлах; принимает ТОЛЬКО если max(V/Vnom)↓, "
        "иначе откат. Безопасен по построению (не может ухудшить).",
    )
    anti_overshoot_ceiling: float = _param(
        1.15,
        group=_G_POST,
        label="Потолок V/Vnom",
        control="number",
        min=1.0,
        max=1.5,
        depends={"anti_overshoot": True},
        help="Узлы выше этого порога без real-V-меры зажимаются. default 1.15.",
    )
    reconcile_balance: bool = _toggle(
        True,
        group=_G_POST,
        label="Закрыть узловой небаланс оценок",
        help="reconcile_node_balance: финализировать разнесение gen/load — "
        "слить остаток inj_calc − (gen−load) по разметке узла. Выход SE "
        "становится согласованным режимом (вход PF, промоут). V/δ не задеты.",
    )


def default_config() -> PipelineConfig:
    """Конфиг с production-дефолтами."""
    return PipelineConfig()


# ---------------------------------------------------------------------------
# Контекст исполнения + дескриптор шага
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    model: Any
    cfg: PipelineConfig
    derived: DerivedInputs | None = None
    result: SEResult | None = None


@dataclass(frozen=True)
class Step:
    """Дескриптор шага пайплайна для реестра/манифеста."""

    name: str
    title: str
    group: str
    description: str
    fn: Callable[[_Ctx], dict | None]
    toggle: str | None = None  # имя bool-поля cfg; None = всегда вкл
    needs_derived: bool = False  # шагу нужен числовой план DerivedInputs (иначе скип)
    # Сетевая деривация: шаг мутирует ТОЛЬКО сетевые таблицы (nodes/branches/
    # generators — статусы, tap, R/X/G/B, шунты, типы узлов), не measurements.
    # Подмножество network=True исполняется prepare_network() для
    # материализации решаемой сети без прогона SE («одна сеть» для SE/PF).
    network: bool = False


def _effective_huber_c(cfg: PipelineConfig) -> float:
    if cfg.huber_c is not None:
        return float(cfg.huber_c)
    return 2.0 if cfg.algorithm == "ipm" else 1.5


def _ipm_kwargs(cfg: PipelineConfig) -> dict:
    return {
        "balance_weight_factor": cfg.ipm_balance_weight_factor,
        "bound_relax": cfg.ipm_bound_relax,
        "prior_sigma2_bus_equiv_pu": cfg.ipm_prior_sigma2_bus_equiv_pu,
    }


def _estimate_kwargs(cfg: PipelineConfig, *, init: str, **overrides: Any) -> dict[str, Any]:
    """Solver kwargs shared by every estimate/re-solve call in the pipeline.

    Single assembly point for the solver call: the first pass and every warm
    re-solve (bad-data, v-refine, v-mirror, anti-overshoot) differ only in
    ``init`` and the ``overrides`` they pass (anti-overshoot uses a larger
    iteration budget and its own tolerance).
    """
    kw: dict[str, Any] = {
        "algorithm": cfg.algorithm,
        "init": init,
        "tolerance": cfg.tolerance,
        "max_iterations": cfg.max_iterations,
        "huber_c": _effective_huber_c(cfg),
        "kkt_solver": cfg.kkt_solver,
        # Quality summary is computed once, on the final solution in run()
        # (see populate_quality_summary); intermediate solves skip it.
        "include_quality_summary": False,
        "reconcile_balance": cfg.reconcile_balance,
    }
    kw.update(overrides)
    if cfg.algorithm == "ipm":
        kw.update(_ipm_kwargs(cfg))
    return kw


def _warm_resolve(ctx: _Ctx, **overrides: Any) -> SEResult:
    """Warm re-solve on the (edited) working model and store it in ``ctx``.

    V/δ of the previous pass are already in the working layer, hence
    ``init="results"``.
    """
    ctx.result = estimate(ctx.model, **_estimate_kwargs(ctx.cfg, init="results", **overrides))
    return ctx.result


def _refine_two_pass(
    ctx: _Ctx,
    *,
    classify: Callable[[_Ctx], Any],
    apply: Callable[[_Ctx, Any], dict],
    empty_stats: Callable[[Any], dict],
    unusable_reason: str,
) -> dict:
    """Shared skeleton of the two-pass post-solve steps (Line C).

    Guard on a usable first-pass solution, classify residuals into a plan,
    apply the plan to the working model and warm re-solve. The steps differ
    only in their classifier/applier and skip wording; the plan object must
    expose an ``empty`` property.
    """
    assert ctx.result is not None  # steps run after estimate → result exists
    if not ctx.result.success:
        # An unusable solution (completed=False) has unreliable residuals —
        # classifying on them is worse than skipping the re-pass.
        return {"skipped": unusable_reason}
    plan = classify(ctx)
    if plan.empty:
        return empty_stats(plan)
    stats = apply(ctx, plan)
    result = _warm_resolve(ctx)
    stats.update(
        {
            "success": bool(result.success),
            "iterations": int(result.iterations),
        }
    )
    return stats


# --- реализации шагов (каждая принимает ctx, возвращает stats-dict) ---


def _s_normalize_breakers(ctx: _Ctx) -> dict:
    return dict(normalize_breaker_reactance(ctx.model) or {})


def _s_voltage_nominal(ctx: _Ctx) -> dict:
    # Применяем готовый план Vnom (вычислен вне ядра).
    assert ctx.derived is not None and ctx.derived.voltage_nominal is not None
    apply_voltage_nominal_resolved(ctx.model, ctx.derived.voltage_nominal)
    return {}


def _s_topology(ctx: _Ctx) -> dict:
    # Применяем готовый ON_LINE-план статусов (вычислен вне ядра).
    assert ctx.derived is not None and ctx.derived.topology_resolved is not None
    return dict(apply_topology_resolved(ctx.model, ctx.derived.topology_resolved) or {})


def _s_rpn(ctx: _Ctx) -> dict:
    return dict(apply_rpn_resolved(ctx.model) or {})


def _s_reactors(ctx: _Ctx) -> dict:
    return dict(apply_reactors_to_node_shunt(ctx.model) or {})


def _s_telemetry(ctx: _Ctx) -> dict:
    # Применяем готовый z-вектор (вычислен вне ядра). Глушение прежних статусов
    # (на входе measurements приходят со status=True) выполняет само ядро
    # _apply_telemetry_on_arrays (arr["status"]=False перед активацией).
    assert ctx.derived is not None and ctx.derived.telemetry_resolved is not None
    assert ctx.derived.telemetry_arg_keys is not None
    return dict(
        apply_telemetry_resolved(
            ctx.model,
            ctx.derived.telemetry_resolved,
            ctx.derived.telemetry_arg_keys,
            total_args=ctx.derived.telemetry_total_args,
        )
        or {}
    )


def _s_voltage_range_filter(ctx: _Ctx) -> dict:
    return dict(apply_voltage_range_filter(ctx.model) or {})


def _s_resolve_merged(ctx: _Ctx) -> dict:
    return dict(resolve_merged_measurement_conflicts(ctx.model) or {})


def _s_refine_slack(ctx: _Ctx) -> dict:
    return dict(refine_slack_to_one(ctx.model) or {})


def _s_refine_node_types(ctx: _Ctx) -> dict:
    return dict(refine_node_types_from_generators(ctx.model) or {})


def _s_disable_orphan_branches(ctx: _Ctx) -> dict:
    # H46: повтор после disconnected_components ниже выполняется тем же шагом-парой.
    return dict(disable_orphan_branches(ctx.model) or {})


def _s_disable_disconnected(ctx: _Ctx) -> dict:
    stats = dict(disable_disconnected_components(ctx.model) or {})
    # H46: пере-отключить ветви, инцидентные только что выключенным узлам.
    if ctx.cfg.disable_orphan_branches:
        recheck = dict(disable_orphan_branches(ctx.model) or {})
        stats["orphan_recheck_disabled"] = recheck.get("disabled", 0)
    return stats


def _s_disable_isolated(ctx: _Ctx) -> dict:
    return dict(disable_isolated_nodes(ctx.model) or {})


def _s_generator_status(ctx: _Ctx) -> dict:
    return dict(apply_generator_status_from_node(ctx.model) or {})


def _s_aggregate_generators(ctx: _Ctx) -> dict:
    return dict(aggregate_generators_to_node(ctx.model) or {})


def _s_gen_v_calibration(ctx: _Ctx) -> dict:
    return dict(apply_voltage_meas_calibration_for_gen_nodes(ctx.model) or {})


def _s_deactivate_orphan_measurements(ctx: _Ctx) -> dict:
    return dict(deactivate_orphan_measurements(ctx.model) or {})


def _s_synthesize_injections(ctx: _Ctx) -> dict:
    return dict(synthesize_node_injection_from_branch_flows(ctx.model) or {})


def _s_mirror_voltage(ctx: _Ctx) -> dict:
    return dict(mirror_voltage_through_unit_tap_links(ctx.model) or {})


def _s_materialize(ctx: _Ctx) -> dict:
    # Применяем готовый наблюдаемый узловой режим (вычислен вне ядра;
    # см. cex3_materialize_invert_sign_fix).
    if ctx.derived is None or ctx.derived.materialize_obs is None:
        return {"skipped": "нет плана материализации (нужны наблюдаемые инжекции)"}
    apply_materialize_resolved(ctx.model, ctx.derived.materialize_obs)
    return {}


def _s_add_pseudo(ctx: _Ctx) -> dict:
    return dict(
        add_pseudo_measurements(
            ctx.model,
            unobservable_v_sigma_frac=ctx.cfg.unobservable_v_sigma_frac,
            unobservable_v_min_vm_deviation=ctx.cfg.unobservable_v_min_vm_deviation,
        )
        or {}
    )


def _s_flow_sigma_floor(ctx: _Ctx) -> dict:
    # σ-floor real-flow мер от шкалы канала. Идёт ПОСЛЕ add_pseudo: селектор
    # is_pseudo==0 гарантирует, что псевдо-приоры не затрагиваются, а итоговые
    # variance real-flow мер соответствуют валидированному A/B-прототипу.
    frac = ctx.cfg.flow_sigma_floor_kv_frac
    if frac is None or frac <= 0:
        return {"skipped": "выключено (flow_sigma_floor_kv_frac=None)"}
    return dict(apply_flow_sigma_floor(ctx.model, kv_frac=float(frac)) or {})


def _s_estimate(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    ctx.result = estimate(ctx.model, **_estimate_kwargs(cfg, init=cfg.init))
    return {
        "algorithm": cfg.algorithm,
        "success": bool(ctx.result.success),
        "iterations": int(ctx.result.iterations),
        "objective_value": float(ctx.result.objective_value),
    }


def _s_bad_data_repass(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_bad_data(
            c.model.measurements.to_numpy(),
            c.model.branches.to_numpy(),
            threshold=cfg.bad_data_threshold,
            sigma_cap=cfg.bad_data_sigma_cap,
            flip_ratio=cfg.bad_data_flip_ratio,
        ),
        apply=lambda c, plan: apply_bad_data_plan(
            c.model, plan, damp_factor=cfg.bad_data_damp_factor
        ),
        empty_stats=lambda plan: {"candidates": plan.n_candidates, "skipped": "no-op (план пуст)"},
        unusable_reason="решение непригодно — нет надёжных остатков",
    )


def _s_v_refine(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_v_refine(
            c.model.measurements.to_numpy(),
            rn_threshold=cfg.v_refine_rn,
        ),
        apply=lambda c, plan: apply_v_refine_plan(c.model, plan, factor=cfg.v_refine_factor),
        empty_stats=lambda plan: {
            "consistent": plan.n_consistent,
            "skipped": "no-op (нет real V-мер)",
        },
        unusable_reason="решение непригодно — нет надёжных остатков",
    )


def _s_v_mirror(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_v_mirror(
            c.model.measurements.to_numpy(),
            c.model.branches.to_numpy(),
            c.model.nodes.to_numpy(),
            max_pu_dev=cfg.v_mirror_max_pu_dev,
            min_lift=cfg.v_mirror_min_lift,
            cross_at=cfg.v_mirror_cross_at,
        ),
        apply=lambda c, plan: apply_v_mirror_plan(c.model, plan),
        empty_stats=lambda _plan: {
            "clusters": 0,
            "skipped": "no-op (нет слепых кластеров с границей)",
        },
        unusable_reason="решение непригодно — нет надёжного уровня границы",
    )


def _s_anti_overshoot(ctx: _Ctx) -> dict:
    assert ctx.result is not None  # шаг идёт после estimate → результат уже есть
    cfg = ctx.cfg

    def _resolve() -> SEResult:
        # Warm re-solve with extra iteration headroom (80 truncates → phantom
        # non-convergence). WLS tol=1e-4, IPM tol=1e-3 — as in stage_c_after_oc.
        return _warm_resolve(
            ctx,
            max_iterations=150,
            tolerance=1e-4 if cfg.algorithm == "wls" else 1e-3,
        )

    ctx.result, stats = refine_anti_overshoot(
        ctx.model,
        ctx.result,
        _resolve,
        ceiling=cfg.anti_overshoot_ceiling,
    )
    return dict(stats or {})


# ---------------------------------------------------------------------------
# Реестр шагов (упорядоченный = валидированная последовательность прогона)
# ---------------------------------------------------------------------------

STEPS: list[Step] = [
    Step(
        "normalize_breakers",
        "Нормализация короткозамыкателей",
        _G_XML,
        "volt-aware X_pu для R=X=0 ветвей.",
        _s_normalize_breakers,
        toggle="normalize_breakers",
        network=True,
    ),
    Step(
        "voltage_nominal",
        "Vnom из XML",
        _G_XML,
        "Заполнить voltage_nominal=0 узлов из плана Vnom.",
        _s_voltage_nominal,
        toggle="apply_voltage_nominal",
        needs_derived=True,
        network=True,
    ),
    Step(
        "topology",
        "Топология из ON_LINE",
        _G_XML,
        "Применить план ON_LINE-статусов → status.",
        _s_topology,
        toggle="apply_topology",
        needs_derived=True,
        network=True,
    ),
    Step(
        "rpn",
        "РПН из TM",
        _G_XML,
        "Применить план № отпаек → динамический tap_ratio.",
        _s_rpn,
        toggle="apply_rpn",
        # needs_derived — наследие: выбор отпайки давно едет через входную
        # таблицу tap_steps (производитель данных), derived шаг не читает.
        # Флаг оставлен сознательно (бит-в-бит): derived=None означает
        # «вход без деривации», и применять РПН на таком входе не нужно.
        needs_derived=True,
        network=True,
    ),
    Step(
        "reactors",
        "Реакторы → шунт",
        _G_XML,
        "apply_reactors_to_node_shunt.",
        _s_reactors,
        toggle="apply_reactors",
        network=True,
    ),
    # ИНВАРИАНТ ПОРЯДКА: telemetry строго ПОСЛЕ rpn — применение z-вектора
    # читает branch.susceptance, который H30-шунт-факторизация РПН меняет.
    Step(
        "telemetry",
        "Телеметрия (z-вектор)",
        _G_XML,
        "Применить z-вектор → measurements.",
        _s_telemetry,
        needs_derived=True,
    ),
    # --- хвост stage_a: slack + типы + каскад статусов ---
    Step(
        "refine_slack",
        "Выбор slack",
        _G_CASCADE,
        "refine_slack_to_one.",
        _s_refine_slack,
        toggle="refine_slack",
        network=True,
    ),
    Step(
        "refine_node_types",
        "Типы узлов из генераторов",
        _G_CASCADE,
        "refine_node_types_from_generators.",
        _s_refine_node_types,
        toggle="refine_node_types",
        network=True,
    ),
    Step(
        "disable_orphan_branches",
        "Гасить orphan-ветви",
        _G_CASCADE,
        "disable_orphan_branches.",
        _s_disable_orphan_branches,
        toggle="disable_orphan_branches",
        network=True,
    ),
    Step(
        "disable_disconnected",
        "Гасить отсоединённые компоненты (+H46)",
        _G_CASCADE,
        "disable_disconnected_components + повтор orphan-branches (H46).",
        _s_disable_disconnected,
        toggle="disable_disconnected",
        network=True,
    ),
    Step(
        "disable_isolated",
        "Гасить изолированные узлы",
        _G_CASCADE,
        "disable_isolated_nodes.",
        _s_disable_isolated,
        toggle="disable_isolated",
        network=True,
    ),
    Step(
        "generator_status",
        "Статус генераторов от узла",
        _G_CASCADE,
        "apply_generator_status_from_node.",
        _s_generator_status,
        toggle="apply_generator_status",
        network=True,
    ),
    # --- этап B: отбраковка измерений + режим + псевдо ---
    Step(
        "voltage_range_filter",
        "Фильтр V вне диапазона",
        _G_XML,
        "apply_voltage_range_filter: downweight V вне диапазона.",
        _s_voltage_range_filter,
        toggle="apply_voltage_range_filter",
    ),
    Step(
        "resolve_merged",
        "Слить дубли измерений",
        _G_XML,
        "resolve_merged_measurement_conflicts: слить дубли на объекте.",
        _s_resolve_merged,
        toggle="resolve_merged_conflicts",
    ),
    Step(
        "refine_node_types_2",
        "Типы узлов (повтор, идемпотентно)",
        _G_CASCADE,
        "refine_node_types_from_generators — повтор в stage_b (безопасно).",
        _s_refine_node_types,
        toggle="refine_node_types",
    ),
    # ИНВАРИАНТ ПОРЯДКА (цепочка из 4 шагов, перестановка молча меняет числа):
    # aggregate_generators (перезаписывает node.generation_* суммой активных
    # генераторов) → materialize (наблюдаемый режим поверх) → add_pseudo
    # (P/Q-приоры читают итоговые pg−pn) → flow_sigma_floor (селектор
    # is_pseudo==0 требует, чтобы все псевдо-меры уже были добавлены).
    Step(
        "aggregate_generators",
        "Агрегировать генераторы к узлу",
        _G_XML,
        "aggregate_generators_to_node.",
        _s_aggregate_generators,
        toggle="aggregate_generators",
    ),
    Step(
        "gen_v_calibration",
        "Калибровка σ² V ген-узлов",
        _G_XML,
        "apply_voltage_meas_calibration_for_gen_nodes.",
        _s_gen_v_calibration,
        toggle="apply_gen_v_calibration",
    ),
    Step(
        "deactivate_orphan_measurements",
        "Деактивировать orphan-измерения",
        _G_CASCADE,
        "deactivate_orphan_measurements.",
        _s_deactivate_orphan_measurements,
        toggle="deactivate_orphan_measurements",
    ),
    Step(
        "synthesize_injections",
        "Синтез P/Q_inj из потоков",
        _G_PSEUDO,
        "synthesize_node_injection_from_branch_flows.",
        _s_synthesize_injections,
        toggle="synthesize_injections",
    ),
    Step(
        "mirror_voltage",
        "Зеркалить V через ktr=1",
        _G_PSEUDO,
        "mirror_voltage_through_unit_tap_links.",
        _s_mirror_voltage,
        toggle="mirror_voltage_unit_tap",
    ),
    Step(
        "materialize",
        "Материализация режима из XML",
        _G_MODE,
        "Применить наблюдаемый режим node.pn/qn/pg/qg.",
        _s_materialize,
        toggle="materialize",
        needs_derived=True,
    ),
    Step(
        "add_pseudo",
        "Псевдо-приоры",
        _G_PSEUDO,
        "add_pseudo_measurements.",
        _s_add_pseudo,
        toggle="add_pseudo",
    ),
    Step(
        "flow_sigma_floor",
        "σ-floor flow-мер от шкалы канала",
        _G_XML,
        "apply_flow_sigma_floor: variance real branch-flow мер ≥ (frac·√3·Vn·1кА)².",
        _s_flow_sigma_floor,
    ),
    Step(
        "estimate",
        "Оценка состояния (WLS/IPM)",
        _G_EST,
        "gridstate.estimate: solve_wls / solve_ipm.",
        _s_estimate,
    ),
    # ИНВАРИАНТ ПОРЯДКА: bad_data_repass → v_refine → v_mirror → anti_overshoot
    # строго МЕЖДУ estimate и финалом. bad_data правит грубые ошибки (значения/
    # статусы) по остаткам первого solve; v_refine ужесточает σ согласованных V
    # по уже-правленым остаткам; v_mirror переставляет значение pseudo-V слепых
    # кластеров по уровню (теплейшей) границы; anti-overshoot полирует итог.
    Step(
        "bad_data_repass",
        "Bad-data re-pass (Линия C)",
        _G_POST,
        "classify_bad_data по остаткам solve + warm re-solve на правленых мерах.",
        _s_bad_data_repass,
        toggle="bad_data",
    ),
    Step(
        "v_refine",
        "V-refine ужесточение (Линия C)",
        _G_POST,
        "classify_v_refine согласованных V по остаткам + warm re-solve на ужесточённых σ.",
        _s_v_refine,
        toggle="v_refine",
    ),
    Step(
        "v_mirror",
        "V-mirror значение слепых pseudo-V (Линия C)",
        _G_POST,
        "classify_v_mirror: pseudo-V слепых кластеров → median pu границы × Vnom + warm re-solve.",
        _s_v_mirror,
        toggle="v_mirror",
    ),
    Step(
        "anti_overshoot",
        "Anti-overshoot уточнение (Линия C)",
        _G_POST,
        "refine_anti_overshoot со само-валидацией (revert).",
        _s_anti_overshoot,
        toggle="anti_overshoot",
    ),
]


# ---------------------------------------------------------------------------
# Исполнение
# ---------------------------------------------------------------------------

# Страж согласованности config ↔ derived («cfg дважды»): производитель планов
# и исполнитель шагов обязаны гейтиться ОДНИМ конфигом. Если включённый шаг
# не получил свой план — это рассинхрон конфигов (derive с одним cfg, run с
# другим), а не легальный вход; падаем рано и понятно, не assert'ом внутри
# шага. materialize здесь сознательно НЕ перечислен: его план опционален
# (шаг мягко скипается с reason — легальный вход без наблюдаемого режима).
_REQUIRED_DERIVED_PLANS: dict[str, tuple[str, ...]] = {
    "voltage_nominal": ("voltage_nominal",),
    "topology": ("topology_resolved",),
    "telemetry": ("telemetry_resolved", "telemetry_arg_keys"),
}


def _check_derived_consistency(
    cfg: PipelineConfig, derived: DerivedInputs, *, network_only: bool = False
) -> None:
    """Проверить, что каждый включённый needs_derived-шаг получил свой план.

    Вызывается только при ``derived is not None`` (вход без деривации легален —
    needs_derived-шаги тогда пропускаются целиком).
    """
    missing: list[str] = []
    for step in STEPS:
        plans = _REQUIRED_DERIVED_PLANS.get(step.name)
        if not plans:
            continue
        if network_only and not step.network:
            continue
        if step.toggle is not None and not getattr(cfg, step.toggle):
            continue
        missing.extend(
            f"шаг '{step.name}' включён, derived.{attr} отсутствует"
            for attr in plans
            if getattr(derived, attr, None) is None
        )
    if missing:
        raise ValueError(
            "Рассинхрон config ↔ derived: "
            + "; ".join(missing)
            + ". Деривация планов и прогон обязаны гейтиться одним и тем же config."
        )


def _emit(on_event: Callable[[dict], None] | None, event: dict) -> None:
    if on_event is not None:
        on_event(event)


def _execute_step(step: Step, ctx: _Ctx, on_event: Callable[[dict], None] | None) -> None:
    """Run one step with the shared skip/timing/error event protocol.

    Both ``run()`` and ``prepare_network()`` go through here, so toggle/derived
    skips, duration accounting and ``step_error`` emission behave identically
    in the full pipeline and in the network-only subset.
    """
    if step.toggle is not None and not getattr(ctx.cfg, step.toggle):
        _emit(
            on_event,
            {
                "type": "step_skipped",
                "name": step.name,
                "reason": f"отключено ({step.toggle}=False)",
            },
        )
        return
    if step.needs_derived and ctx.derived is None:
        _emit(
            on_event,
            {"type": "step_skipped", "name": step.name, "reason": "нет XML-деривации"},
        )
        return
    _emit(on_event, {"type": "step_start", "name": step.name})
    t0 = time.monotonic()
    try:
        stats = step.fn(ctx) or {}
    except Exception as exc:
        _emit(
            on_event,
            {
                "type": "step_error",
                "name": step.name,
                "error": repr(exc),
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        raise
    _emit(
        on_event,
        {
            "type": "step_done",
            "name": step.name,
            "stats": stats,
            "duration_ms": int((time.monotonic() - t0) * 1000),
        },
    )


def _build_working(model: Any) -> Any:
    """Рабочий слой = gridstate-native ``Working``-контейнер.

    Принимает либо объект-модель с коллекциями (строит ``Working`` из его
    ``nodes``/``branches``/``measurements``/``generators``), либо уже готовый
    :class:`~gridstate.working.Working` (тогда возвращает его ``copy()``).
    Последнее — прямой вход: вызывающий собирает ``Working.from_arrays(...)``
    из numpy-массивов и передаёт прямо в ``run``. В обоих случаях результат —
    независимая копия: переданный Input остаётся read-only.
    """
    if isinstance(model, Working):
        # Клон, а НЕ pass-through: иначе run() мутировал бы переданный Working
        # (псевдо-измерения, V/δ) — Input должен оставаться read-only и на
        # npz-входе (повторный прогон на том же объекте иначе падал).
        return model.copy()
    return Working.from_model(model)


def _seed_warm_start(working: Any, init_state: Any) -> int:
    """Засеять V/δ рабочей копии из прошлого ``SEResult`` — явный тёплый старт.

    Input read-only ⇒ ``run`` каждый раз клонирует Input (V/δ = плоские из XML),
    поэтому межпрогонный warm-start нельзя получить чтением перетёртого
    ``voltage_magnitude`` (как раньше). Вместо этого вызывающий передаёт прошлый
    результат как ``init_state``; его V/δ (kV/рад, keyed по id) пишутся в узлы
    рабочей копии ДО препроцессинга. Шаг ``estimate`` с ``init="results"`` их
    подхватит (препроцессинг V/δ не трогает). Возвращает число засеянных узлов.

    Источник V/δ — ``init_state.outputs.nodes`` (Output-таблица), с фолбэком на
    ``init_state.model.nodes``.
    """
    vd: dict[int, tuple[float, float]] = {}
    out_nodes = getattr(getattr(init_state, "outputs", None), "nodes", None)
    if (
        out_nodes is not None
        and getattr(out_nodes, "size", 0) > 0
        and "voltage_magnitude" in (out_nodes.dtype.names or ())
    ):
        for i in range(len(out_nodes)):
            vd[int(out_nodes["id"][i])] = (
                float(out_nodes["voltage_magnitude"][i]),
                float(out_nodes["voltage_angle"][i]),
            )
    else:
        src = init_state.model.nodes.to_numpy()
        for r in src:
            vd[int(r["id"])] = (float(r["voltage_magnitude"]), float(r["voltage_angle"]))

    arr = working.nodes.to_numpy().copy()
    seeded = 0
    for i in range(len(arr)):
        got = vd.get(int(arr[i]["id"]))
        if got is not None:
            arr[i]["voltage_magnitude"] = got[0]
            arr[i]["voltage_angle"] = got[1]
            seeded += 1
    working.nodes.update_from_array(arr)
    return seeded


def run(
    model: Any,
    *,
    config: PipelineConfig | None = None,
    derived: DerivedInputs | None = None,
    on_event: Callable[[dict], None] | None = None,
    init_state: Any = None,
) -> SEResult:
    """Прогнать полный SE-пайплайн и вернуть ``SEResult``. **Input read-only.**

    ``run`` — чистая функция ``(Input, config) → Output``: входная ``model`` НЕ
    мутируется. Внутри строится рабочая копия (working-слой), на ней идёт весь
    препроцессинг + солвер, результат — в ``SEResult`` (``result.model`` — это
    рабочая копия с V/δ/потоками, ``result.outputs`` — Output-таблицы keyed по
    id). Следствия: повторный ``run`` на той же ``model`` детерминирован *by
    construction* (без сброса); edit→rerun работает (правьте Input — движок его
    только читает); тёплый старт — явным ``init`` + прежним результатом.

    **Контракт сменился:** раньше ``model`` обновлялась in-place; теперь читайте
    результат из ``result.model`` / ``result.outputs`` / ``result.v_pu``, а не из
    переданного объекта (он остаётся в исходном состоянии).

    Args:
        model: рабочий слой (:class:`~gridstate.working.Working`) или совместимый
            носитель контрактных таблиц. НЕ мутируется.
        config: :class:`PipelineConfig`; ``None`` → production-дефолты.
        derived: предвычисленные числовые планы (:class:`~gridstate.contract.derived.
            DerivedInputs`) — топология/РПН/телеметрия/материализация/Vnom. Если
            задан — шаги применяют готовые планы контрактными ядрами. Если ``None`` —
            шаги, требующие числового плана (помеченные ``needs_derived``), пропускаются
            (модель должна уже нести измерения).
        on_event: callback прогресса. Получает dict-события ``step_start`` /
            ``step_done`` / ``step_skipped`` / ``step_error`` / ``final``.
        init_state: прошлый ``SEResult`` для **тёплого старта** (цепочка
            ``run(wls)`` → ``run(ipm, init_state=res_wls)``). Его V/δ засеваются
            в рабочую копию до препроцессинга, и оценка идёт от них (init
            форсируется в ``"results"``). ``None`` — холодный старт по ``cfg.init``.

    Returns:
        ``SEResult`` (он же ``ctx.result``); результат — в ``result.model``
        (рабочая копия) и ``result.outputs``. Входная ``model`` не изменена.
    """
    cfg = config or PipelineConfig()

    # Working-слой: gridstate-native рабочий слой поверх контрактных массивов. Весь
    # препроцессинг/солвер мутируют ТОЛЬКО его; входная model остаётся read-only →
    # повтор детерминирован by construction (см. _build_working/Working), сброс не нужен.
    working = _build_working(model)

    # Тёплый старт: засеять V/δ из прошлого результата + форсировать init="results".
    if init_state is not None:
        seeded = _seed_warm_start(working, init_state)
        cfg = replace(cfg, init="results")
        _emit(on_event, {"type": "warm_start", "seeded_nodes": seeded})

    ctx = _Ctx(model=working, cfg=cfg)

    # Числовые планы (топология/РПН/телеметрия/материализация/Vnom) приходят готовыми
    # (``derived``) — производитель данных вычислил их вне ядра. Шаги ниже применяют их
    # контрактными ядрами на своих позициях. Если планов нет — needs_derived-шаги
    # пропускаются (модель должна уже нести измерения).
    if derived is not None:
        _check_derived_consistency(cfg, derived)
        ctx.derived = derived

    _emit(
        on_event, {"type": "clone", "nodes": len(working.nodes), "branches": len(working.branches)}
    )

    for step in STEPS:
        _execute_step(step, ctx, on_event)

    assert ctx.result is not None  # _s_estimate (без toggle) всегда заполняет result

    # Quality summary — один раз, на финальном решении: промежуточные solve
    # (estimate, anti-overshoot re-solve) идут с include_quality_summary=False,
    # т.к. её расчёт на крупных моделях сопоставим по цене с самим solve.
    t0 = time.monotonic()
    populate_quality_summary(ctx.result, top_n=cfg.top_residuals_n)
    _emit(
        on_event,
        {
            "type": "step_done",
            "name": "quality_summary",
            "stats": {},
            "duration_ms": int((time.monotonic() - t0) * 1000),
        },
    )

    _emit(
        on_event,
        {
            "type": "final",
            "success": bool(getattr(ctx.result, "success", False)),
            "iterations": int(getattr(ctx.result, "iterations", 0)),
        },
    )
    return ctx.result


def prepare_network(
    model: Any,
    *,
    config: PipelineConfig | None = None,
    derived: DerivedInputs | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Working:
    """Выполнить ТОЛЬКО сетевые деривации пайплайна и вернуть ``Working``.

    Исполняет подмножество ``STEPS`` с ``network=True`` (нормализация
    выключателей, Vnom, ON_LINE-топология, РПН, реакторы, slack/типы узлов,
    каскады статусов, статусы генераторов) — без телеметрии, псевдо-мер и
    солвера. Результат — сеть в том состоянии, в котором её решает SE.

    Назначение — материализация «одной сети»: перенос полученного сетевого
    состояния в модель-носитель (это делает внешний адаптер) делает вход
    консистентным для SE/PF/последующих расчётов; сами деривации
    идемпотентны на материализованной сети (повторный прогон — no-op),
    поэтому последующий полный ``run`` даёт бит-в-бит тот же результат.

    Args:
        model: носитель контрактных таблиц (как у :func:`run`). НЕ мутируется.
        config: :class:`PipelineConfig` — уважаются те же toggle'ы.
        derived: числовые планы; без них needs_derived-шаги пропускаются.
        on_event: callback прогресса (события как у :func:`run`).

    Returns:
        ``Working`` — рабочая копия с применёнными сетевыми деривациями.
    """
    cfg = config or PipelineConfig()
    working: Working = _build_working(model)
    ctx = _Ctx(model=working, cfg=cfg)
    if derived is not None:
        _check_derived_consistency(cfg, derived, network_only=True)
        ctx.derived = derived

    for step in STEPS:
        if not step.network:
            continue
        _execute_step(step, ctx, on_event)
    return working


# ---------------------------------------------------------------------------
# Манифест для UI (JSON-сериализуемый)
# ---------------------------------------------------------------------------


def manifest() -> dict:
    """Декларативное описание пайплайна для UI.

    Возвращает dict (JSON-сериализуемый)::

        {
          "steps":  [{name, title, group, description, optional, toggle,
                      default_enabled, needs_derived}, ...],   # порядок исполнения
          "params": [{name, type, default, control, group, label, help,
                      min?, max?, choices?, depends?}, ...],
          "groups": ["XML-препроцессинг", "Режим...", ...],   # порядок секций
        }

    UI строит форму, итерируя ``params`` (контролы) и ``steps`` (toggle-список),
    группируя по ``group``. Дефолты берёт отсюда — не хардкодит.
    """
    cfg_defaults = PipelineConfig()

    params: list[dict] = []
    groups_order: list[str] = []
    for f in fields(PipelineConfig):
        md = dict(f.metadata)
        if not md:
            continue
        grp = md.get("group", "")
        if grp and grp not in groups_order:
            groups_order.append(grp)
        default = getattr(cfg_defaults, f.name)
        entry = {
            "name": f.name,
            "kind": md.get("kind", "param"),
            "type": _type_name(f.type),
            "default": default,
            "control": md.get("control"),
            "group": grp,
            "label": md.get("label", f.name),
            "help": md.get("help", ""),
        }
        for k in ("min", "max", "choices", "depends"):
            if k in md:
                entry[k] = md[k]
        params.append(entry)

    steps_out: list[dict] = []
    for step in STEPS:
        if step.group and step.group not in groups_order:
            groups_order.append(step.group)
        steps_out.append(
            {
                "name": step.name,
                "title": step.title,
                "group": step.group,
                "description": step.description,
                "optional": step.toggle is not None,
                "toggle": step.toggle,
                "default_enabled": (
                    getattr(cfg_defaults, step.toggle) if step.toggle is not None else True
                ),
                "needs_derived": step.needs_derived,
            }
        )

    return {"steps": steps_out, "params": params, "groups": groups_order}


def _type_name(tp: Any) -> str:
    """Имя типа поля для UI (str/int/float/bool; float|None → 'float')."""
    s = str(tp)
    for name in ("bool", "int", "float", "str"):
        if name in s:
            return name
    return "str"
