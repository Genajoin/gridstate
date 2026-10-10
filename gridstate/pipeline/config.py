"""Pipeline configuration: :class:`PipelineConfig` and its UI metadata.

The dataclass is the single source of truth for the pipeline defaults
(gridstate does not depend on pydantic). Every field carries ``metadata`` for
the UI manifest (:func:`gridstate.pipeline.manifest`):

* ``kind``: ``"toggle"`` (boolean step on/off) | ``"param"`` (scalar);
* ``group``: form section;
* ``label``: human-readable caption;
* ``help``: hint text;
* ``control``: UI control hint (``"switch"`` / ``"number"`` / ``"select"``);
* ``min`` / ``max``: bounds (numbers); ``choices``: allowed values (select);
* ``depends``: show only when another field equals a value (e.g. ``algorithm=ipm``).

``choices`` / ``min`` / ``max`` are enforced: :meth:`PipelineConfig.validate`
runs on construction and at the start of a run.

Downstream configurations subclass :class:`PipelineConfig` to change defaults
(fields stay flat; the class is neither frozen nor slotted). A subclass field
redeclared with a bare default keeps the UI metadata of the base declaration
(:func:`_effective_metadata`), for the manifest and the validation alike.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import Field, dataclass, field, fields
from typing import Any


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


# Form sections (``group`` of fields and steps). The values are user-facing
# captions published by the manifest, so they are kept verbatim.
_G_INPUT = "Препроцессинг входа"
_G_MODE = "Режим (нагрузка/генерация)"
_G_CASCADE = "Каскад и статусы"
_G_PSEUDO = "Псевдо-измерения"
_G_EST = "Оценка состояния"
_G_IPM = "IPM"
_G_POST = "Пост-обработка"


@dataclass
class PipelineConfig:
    """All pipeline knobs. The defaults are the validated production configuration."""

    # --- Input preprocessing ---
    normalize_breakers: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Нормализация X короткозамыкателей",
        help="volt-aware X_pu для R=X=0 ветвей (убирает β-выбросы dV_max на LV).",
    )
    apply_voltage_nominal: bool = _toggle(
        False,
        group=_G_INPUT,
        label="Vnom (номинальное напряжение)",
        help="Заполнить voltage_nominal=0 узлов из готового плана Vnom. Обычно vn уже "
        "задан на входе; default OFF.",
    )
    apply_topology: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Топология (ON_LINE статусы)",
        help="Применить готовый план ON_LINE-статусов → status узлов/ветвей/ген/реакторов.",
    )
    apply_rpn: bool = _toggle(
        True,
        group=_G_INPUT,
        label="РПН (№ отпаек)",
        help="Применить готовый план № отпаек → динамический tap_ratio/phase_shift.",
    )
    apply_reactors: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Реакторы → шунт узла",
        help="apply_reactors_to_node_shunt (ШР в shunt_b). Физически необходим "
        "для длинных ВЛ 500/750 кВ.",
    )
    apply_voltage_range_filter: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Фильтр V вне диапазона",
        help="downweight V вне voltage_critical/max+10%.",
    )
    voltage_filter_min_nominal_kv: float = _param(
        110.0,
        group=_G_INPUT,
        label="Мин. класс V-фильтра, кВ",
        control="number",
        min=0.0,
        max=1200.0,
        help="apply_voltage_range_filter: узлы с Vnom ниже порога не проверяются. "
        "0 = проверять все классы; ловит битые LV U-меры (датчик чужого класса, "
        "value ≫ Vnom), которые проходят нижний фильтр apply_telemetry.",
        depends={"apply_voltage_range_filter": True},
    )
    voltage_filter_deactivate_nonphysical: bool = _toggle(
        False,
        group=_G_INPUT,
        label="Деактивировать нефизичные V",
        help="apply_voltage_range_filter: V<=0, V<Vnom·lower и V>Vnom·upper (физически "
        "невозможные как напряжение) деактивируются (status=False), а не downweight. "
        "σ×10 не гасит pu=127 / отрицательные / коллапс — они рушат solve.",
    )
    voltage_filter_nonphysical_lower_factor: float = _param(
        0.5,
        group=_G_INPUT,
        label="Нижний физ-предел V, ×Vnom",
        control="number",
        min=0.0,
        max=0.9,
        help="voltage_filter_deactivate_nonphysical: V<Vnom·factor деактивируется как "
        "нефизичное (default 0.5 = 50 % Vnom, = soft-floor). V<=0 деактивируется всегда.",
        depends={"voltage_filter_deactivate_nonphysical": True},
    )
    voltage_filter_nonphysical_upper_factor: float = _param(
        1.5,
        group=_G_INPUT,
        label="Верхний физ-предел V, ×Vnom",
        control="number",
        min=1.1,
        max=5.0,
        help="voltage_filter_deactivate_nonphysical: V>Vnom·factor деактивируется как "
        "нефизичное (default 1.5 = 150 % Vnom).",
        depends={"voltage_filter_deactivate_nonphysical": True},
    )
    aggregate_generators: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Агрегировать генераторы к узлу",
        help="aggregate_generators_to_node: multi-gen → узловая генерация.",
    )
    release_load_only_injections: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Снять неполные инжекции ген-узлов",
        help=(
            "release_load_only_injections: a node injection built from the load "
            "component only (PN without PG, QN without QG) is dropped on nodes "
            "with generation; the generation stays free within its range."
        ),
    )
    relax_generation_min_to_injection: bool = _toggle(
        False,
        group=_G_INPUT,
        label="Снять Pmin, противоречащий инжекции",
        help=(
            "relax_generation_min_to_injection: when the P injection reading of a "
            "generating node, plus its load box and 3 sigma, stays below the minimum "
            "generation, the minimum is lowered to the reading."
        ),
    )
    widen_generation_only_injections: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Учесть ящик нагрузки в инжекции из генерации",
        help=(
            "widen_generation_only_injections: a node injection built from the "
            "generation component only (PG without PN, QG without QN) on a node "
            "with a load box is shifted by the box centre and its variance grows "
            "by (hi - lo)^2 / 12, so the load can settle within its box; with "
            "load but no box the load is unknown and the measurement is dropped."
        ),
    )
    apply_gen_v_calibration: bool = _toggle(
        True,
        group=_G_INPUT,
        label="Калибровка σ² V ген-узлов",
        help="apply_voltage_meas_calibration_for_gen_nodes.",
    )
    gen_v_calibration_mode: str = _param(
        "absolute",
        group=_G_INPUT,
        label="Режим калибровки σ² V ген-узлов",
        control="select",
        choices=["absolute", "tighten"],
        help="absolute: variance GOOD V-мер ген-узлов := sigma2 (0.1 кВ²); у шин 6-16 кВ "
        "телеметрийная variance меньше (~0.01-0.025), так что режим их ОСЛАБЛЯЕТ до ×10 "
        "и затирает прежнее масштабирование. tighten: variance := min(текущая, sigma2), "
        "никогда не ослабляет.",
        depends={"apply_gen_v_calibration": True},
    )
    flow_sigma_floor_kv_frac: float | None = _param(
        None,
        group=_G_INPUT,
        label="σ-floor flow-мер (доля шкалы)",
        control="number",
        min=0.0,
        max=1.0,
        help="σ_min real branch-flow мер P/Q = доля шкалы канала √3·Vn·1кА "
        "(Vn ветви = max(Vn концов)): variance := max(variance, floor²). "
        "Лечит пере-доверие мелким потокам (σ≈α·|z| занижена при малых z). "
        "None = выключено; 0.010 = 1 % шкалы (110 кВ → 1.9 МВт, 500 кВ → 8.7 МВт).",
    )

    # --- Operating point ---
    materialize: bool = _toggle(
        True,
        group=_G_MODE,
        label="Материализация режима",
        help="Применить наблюдаемый узловой режим node.pn/qn/pg/qg "
        "(нагрузка по районам + генерация вербатим). Кормит IPM box-init.",
    )

    # --- Cascade / statuses ---
    refine_slack: bool = _toggle(
        True,
        group=_G_CASCADE,
        label="Выбор slack (refine_slack_to_one)",
        help="Keep exactly one slack node, the balancing node chosen by the input "
        "(the legacy multi-slack mode is deprecated).",
    )
    refine_node_types: bool = _toggle(True, group=_G_CASCADE, label="Типы узлов из генераторов")
    disable_orphan_branches: bool = _toggle(
        True,
        group=_G_CASCADE,
        label="Гасить orphan-ветви (×2)",
        help="disable_orphan_branches before and after disconnected_components (a branch "
        "left dangling by a node disabled in the second pass is switched off too).",
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

    # --- Pseudo measurements ---
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
    loosen_unknown_load_pseudo: bool = _toggle(
        True,
        group=_G_PSEUDO,
        label="Рыхлый приор нагрузки без данных",
        help=(
            "add_pseudo_measurements(loosen_unknown_load): a node with load but no "
            "load box and nothing materialized gets its zero P/Q injection prior "
            "loosened where the injection is fixed by flow (or, for Q, voltage) "
            "measurements."
        ),
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
        help="Gate of the hard V anchor: a node is anchored only when its input "
        "operating point deviates from Vnom by more than this share; on a flat "
        "input (vm = vn) the anchor is a no-op.",
    )

    # --- State estimation ---
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
    ipm_boxed_balance_weight_factor: float = _param(
        0.0,
        group=_G_IPM,
        label="Balance weight of boxed nodes",
        control="number",
        min=0.0,
        max=1e6,
        depends={"algorithm": "ipm"},
        help="Nodes whose injection range is fully set by their generation/load boxes get a "
        "balance row with variance median(σ²_data) / factor. The box carries the node's "
        "freedom; a loose balance only hides data conflicts as an imbalance. 0 — off.",
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

    # --- Post-processing ---
    bad_data: bool = _toggle(
        False,
        group=_G_POST,
        label="Bad-data re-pass (двухпроходный)",
        help="Classify real measurements by the residuals of the solved estimate "
        "(flip sign errors, reject broken zeros and outliers, damp Qinj), then "
        "warm re-solve. Pair immunity of consistent branch pairs and the (node, "
        "domain) coverage guard protect honest measurements. Validated on four "
        "large production models. Default OFF (changes the solution and the run "
        "time; enabled by the caller).",
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
        help="Two-pass: tighten σ (× factor) of real V measurements consistent "
        "with the first-pass solution (|z−h|/σ < rn), then warm re-solve. Fixes a "
        "global downward V bias (the estimate sags below its own V measurements). "
        "Conflicting V measurements (a bad boundary reading) are left alone. Values "
        "and statuses are unchanged, only variances, so it cannot inject a gross "
        "error. Validated on four large production models. Default OFF (enabled "
        "by the caller).",
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
        help="variance := variance · factor² (σ × factor). 0.7 is the best value "
        "across the validation models; 0.5 removes more bias but worsens the "
        "per-class maximum error on a model with a bad boundary V measurement.",
    )
    v_mirror: bool = _toggle(
        False,
        group=_G_POST,
        label="V-mirror (значение pseudo-V слепых кластеров)",
        help="Two-pass: move the pseudo-V placeholders (value=Vnom) of blind "
        "clusters without real telemetry to the median pu of their boundary of the "
        "SAME voltage class × Vnom (only nodes systematically below the boundary, "
        "see the lift gate), then warm re-solve. Fixes blind tails sagging to "
        "nominal while physically they stay at the boundary level. Only the prior "
        "value changes, σ is kept. Validated on four large production models "
        "(best case p50 −13 %, p95 −16 %). Default OFF (enabled by the caller).",
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
        help="A node is moved only when its solution is below the boundary of its "
        "class by more than min_lift (pu). Nodes already at the level stay put: "
        "lifting them pushes the observed network up (a model whose input carries "
        "a real operating point regresses).",
    )
    v_mirror_cross_at: bool = _toggle(
        False,
        group=_G_POST,
        label="V-mirror через АТ (cross-class)",
        help="When a blind cluster has NO boundary of its own class but has one "
        "across an active transformer (branch_type=1, tap>0), use median(pu of the "
        "solved transformer boundary)·Vnom (pu is invariant to an ideal tap). The "
        "lift/max_pu_dev gates still apply. Default OFF: the pu invariant across "
        "tap≈2 leaves a 2-4 % residual error and some models may regress; enable "
        "only after an A/B comparison on the target models.",
    )
    refine_merge: bool = _toggle(
        False,
        group=_G_POST,
        label="Слить v_refine/v_mirror/chain в один re-solve (research)",
        help="Performance mode: the v_refine, v_mirror and v_mirror_chain "
        "classifications run in turn on ONE solution (after bad_data), their plans "
        "are applied to the measurements, then ONE warm re-solve replaces three. "
        "Saves two full solves (up to −25 % wall time). The semantics change: "
        "v_mirror/chain classify the solution BEFORE the v_refine tightening (not "
        "bit-identical), so enable it only after an A/B quality check. The "
        "individual toggles are respected (a disabled stage does not take part). "
        "Default OFF.",
    )
    v_mirror_chain: bool = _toggle(
        False,
        group=_G_POST,
        label="Chain-mirror pseudo-V глубоких цепочек (research)",
        help="Two-pass: move the pseudo-V placeholders (value=Vnom) of nodes at "
        "depth d≥1 from the nearest live real V measurement (multi-source BFS over "
        "active branches of the same class, plus autotransformers ≥110 kV with "
        "v_mirror_cross_at) to (1+(pu_src−1)·decay^(d−1))·Vnom, then warm "
        "re-solve. Complements v_mirror, which covers fully blind clusters only, "
        "while nodes observed through flows alone sag to nominal (36-48 % of such "
        "nodes sit at d≥2). The v_mirror gates (max_pu_dev, min_lift) still apply. "
        "Default OFF.",
    )
    v_mirror_chain_decay: float = _param(
        0.85,
        group=_G_POST,
        label="Затухание pu к номиналу за шаг",
        control="number",
        min=0.0,
        max=1.0,
        depends={"v_mirror_chain": True},
        help="pu_target = 1 + (pu_src−1)·decay^(d−1): d=1 takes the full source pu "
        "(like a v_mirror boundary), deeper nodes decay towards Vnom. 0.85 is the "
        "best A/B value (2026-07-06) and degrades none of the validation models; "
        "1.0 (no decay) overshoots: deep nodes whose voltage physically stays near "
        "nominal are lifted by +3…7 kV (p95 +2.2 %). The decay approximates the "
        "physical voltage drop along unmeasured chains; it is not an extrapolation.",
    )
    v_mirror_chain_max_depth: int = _param(
        12,
        group=_G_POST,
        label="Максимальная глубина BFS",
        control="number",
        min=1,
        max=30,
        depends={"v_mirror_chain": True},
        help="The front stops here: with decay=0.85 the contribution at d=12 is "
        "already ~17 % of the lift; chains up to d=18 occur, but that far the "
        "nominal value is the more honest prior.",
    )
    shunt_sanity: bool = _toggle(
        False,
        group=_G_POST,
        label="Shunt-sanity try-off/flip (research)",
        help="Orchestrated BY contract.run (not a pipeline step): after the base "
        "run, active shunts on nodes with a V mismatch |z−h| > v_frac·Vnom are "
        "tried off and flipped with a FULL pipeline re-run on a copy of the input; "
        "a variant is accepted ONLY when Σrn² drops by more than gate_drop. Catches "
        "reactor records that are defective in behaviour (statically "
        "indistinguishable from honest ones). Expensive: (1 + 2·candidates) full "
        "runs. Research: it edits the network from measurement statistics, so on "
        "live telemetry it may flicker between snapshots; for a production fix an "
        "offline data audit is preferable. Validation (2026-07-06, four large "
        "models): dVmax −40.7 % on one model, the others bit-identical.",
    )
    shunt_sanity_v_frac: float = _param(
        0.012,
        group=_G_POST,
        label="Порог V-невязки кандидата (доля Vnom)",
        control="number",
        min=0.001,
        max=0.2,
        depends={"shunt_sanity": True},
        help="Кандидат — шунт-узел с node-V-мерой |z−h| > v_frac·Vnom.",
    )
    shunt_sanity_gate_drop: float = _param(
        5.0,
        group=_G_POST,
        label="Гейт принятия ΔΣrn²",
        control="number",
        min=0.0,
        max=1e6,
        depends={"shunt_sanity": True},
        help="Правка принимается, только если Σrn² падает больше чем на gate_drop.",
    )
    shunt_sanity_max_candidates: int = _param(
        6,
        group=_G_POST,
        label="Максимум кандидатов",
        control="number",
        min=1,
        max=64,
        depends={"shunt_sanity": True},
        help="Кап числа кандидатов (каждый стоит до двух warm re-solve).",
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
    reconcile_respect_bounds: bool = _toggle(
        False,
        group=_G_POST,
        label="Reconcile уважает границы",
        help="The residual split is clipped to load/gen_*_min/max (transit → 0), so "
        "the soft balance residual does not fabricate non-physical negative loads "
        "(a consistent operating point never violates the load minimum pn_min). "
        "The unclosed part goes to stats sum_unclosed_p_mw. Default OFF (previous "
        "behaviour).",
    )

    # --- Validation ---

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check every value against the ``choices`` / ``min`` / ``max`` of its metadata.

        Runs on construction (also for ``dataclasses.replace``) and again at the
        start of :func:`gridstate.pipeline.run`, so an out-of-range value fails
        before any preprocessing instead of deep inside the solver. A subclass
        field redeclared with a bare default is checked against the metadata of
        the base declaration; fields without metadata are not checked. ``None``
        is accepted where the field type allows it (an optional knob that is
        switched off or automatic).

        Raises:
            ValueError: listing every invalid field.
        """
        problems: list[str] = []
        for f in fields(self):
            md = _effective_metadata(f)
            value = getattr(self, f.name)
            choices = md.get("choices")
            lo, hi = md.get("min"), md.get("max")
            if choices is None and lo is None and hi is None:
                continue
            if value is None:
                if "None" not in str(f.type):
                    problems.append(f"{f.name}=None is not allowed")
                continue
            if choices is not None:
                if value not in choices:
                    problems.append(f"{f.name}={value!r} is not one of {choices}")
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                problems.append(f"{f.name}={value!r} is not a number")
            elif math.isnan(value):
                problems.append(f"{f.name}=nan is not a number")
            elif lo is not None and value < lo:
                problems.append(f"{f.name}={value!r} is below the minimum {lo!r}")
            elif hi is not None and value > hi:
                problems.append(f"{f.name}={value!r} is above the maximum {hi!r}")
        if problems:
            raise ValueError(f"invalid {type(self).__name__}: " + "; ".join(problems))


#: UI metadata of the base declarations, by field name.
_BASE_METADATA: dict[str, dict[str, Any]] = {
    f.name: dict(f.metadata) for f in fields(PipelineConfig)
}


def _effective_metadata(f: Field[Any]) -> dict[str, Any]:
    """UI metadata of a config field.

    A subclass that redeclares a field with a bare default (to change the
    default) drops its metadata; the base declaration's metadata then applies,
    for the manifest and for the validation alike.
    """
    return dict(f.metadata) or dict(_BASE_METADATA.get(f.name, {}))
