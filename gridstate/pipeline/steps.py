"""Ordered registry of pipeline steps.

:data:`STEPS` is the validated run sequence: :func:`gridstate.pipeline.run`
executes it in order, :func:`gridstate.pipeline.prepare_network` executes its
``network=True`` subset, and :func:`gridstate.pipeline.manifest` publishes it
to the UI. Titles and descriptions are user-facing manifest text and are kept
verbatim.
"""

from __future__ import annotations

from gridstate.pipeline.config import (
    _G_CASCADE,
    _G_EST,
    _G_INPUT,
    _G_MODE,
    _G_POST,
    _G_PSEUDO,
)
from gridstate.pipeline.context import Step
from gridstate.pipeline.passes import (
    _s_anti_overshoot,
    _s_bad_data_repass,
    _s_estimate,
    _s_refine_merged,
    _s_v_mirror,
    _s_v_mirror_chain,
    _s_v_refine,
)
from gridstate.pipeline.preparation import (
    _s_add_pseudo,
    _s_aggregate_generators,
    _s_deactivate_orphan_measurements,
    _s_disable_disconnected,
    _s_disable_isolated,
    _s_disable_orphan_branches,
    _s_flow_sigma_floor,
    _s_gen_v_calibration,
    _s_generator_status,
    _s_materialize,
    _s_mirror_voltage,
    _s_normalize_breakers,
    _s_reactors,
    _s_refine_node_types,
    _s_refine_slack,
    _s_relax_generation_min_to_injection,
    _s_release_load_only_injections,
    _s_rpn,
    _s_telemetry,
    _s_topology,
    _s_voltage_nominal,
    _s_voltage_range_filter,
    _s_widen_generation_only_injections,
)


STEPS: list[Step] = [
    Step(
        "normalize_breakers",
        "Нормализация короткозамыкателей",
        _G_INPUT,
        "volt-aware X_pu для R=X=0 ветвей.",
        _s_normalize_breakers,
        toggle="normalize_breakers",
        network=True,
    ),
    Step(
        "voltage_nominal",
        "Vnom from the input plan",
        _G_INPUT,
        "Заполнить voltage_nominal=0 узлов из плана Vnom.",
        _s_voltage_nominal,
        toggle="apply_voltage_nominal",
        needs_derived=True,
        network=True,
    ),
    Step(
        "topology",
        "Топология из ON_LINE",
        _G_INPUT,
        "Применить план ON_LINE-статусов → status.",
        _s_topology,
        toggle="apply_topology",
        needs_derived=True,
        network=True,
    ),
    Step(
        "rpn",
        "РПН из TM",
        _G_INPUT,
        "Применить план № отпаек → динамический tap_ratio.",
        _s_rpn,
        toggle="apply_rpn",
        # needs_derived is a legacy flag: the tap position has long come
        # through the input tap_steps table (filled by the data producer), the
        # step does not read derived. The flag is kept on purpose (bit-identical
        # behaviour): derived=None means an input without derivation, and the
        # tap changer must not be applied on such an input.
        needs_derived=True,
        network=True,
    ),
    Step(
        "reactors",
        "Реакторы → шунт",
        _G_INPUT,
        "apply_reactors_to_node_shunt.",
        _s_reactors,
        toggle="apply_reactors",
        network=True,
    ),
    # ORDER INVARIANT: telemetry strictly AFTER rpn. Applying the z-vector
    # reads branch.susceptance, which the tap-dependent shunt scaling of the
    # tap-changer step modifies.
    Step(
        "telemetry",
        "Телеметрия (z-вектор)",
        _G_INPUT,
        "Применить z-вектор → measurements.",
        _s_telemetry,
        needs_derived=True,
    ),
    # --- network preparation tail: slack + node types + status cascade ---
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
        "Гасить отсоединённые компоненты",
        _G_CASCADE,
        "disable_disconnected_components + a repeated orphan-branch pass.",
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
    # --- measurement screening + operating point + pseudo measurements ---
    Step(
        "voltage_range_filter",
        "Фильтр V вне диапазона",
        _G_INPUT,
        "apply_voltage_range_filter: downweight V вне диапазона.",
        _s_voltage_range_filter,
        toggle="apply_voltage_range_filter",
    ),
    Step(
        "refine_node_types_2",
        "Типы узлов (повтор, идемпотентно)",
        _G_CASCADE,
        "refine_node_types_from_generators, repeated after the measurement steps (safe).",
        _s_refine_node_types,
        toggle="refine_node_types",
    ),
    # ORDER INVARIANT (a chain of 4 steps, a permutation silently changes the
    # numbers): aggregate_generators (overwrites node.generation_* with the sum
    # of the active generators) → materialize (the observed operating point on
    # top) → add_pseudo (the P/Q priors read the final pg−pn) → flow_sigma_floor
    # (the is_pseudo==0 selector needs every pseudo measurement already added).
    Step(
        "aggregate_generators",
        "Агрегировать генераторы к узлу",
        _G_INPUT,
        "aggregate_generators_to_node.",
        _s_aggregate_generators,
        toggle="aggregate_generators",
    ),
    Step(
        "release_load_only_injections",
        "Снять неполные инжекции ген-узлов",
        _G_INPUT,
        "release_load_only_injections: PN without PG (QN without QG) on a node "
        "with generation is not a net injection.",
        _s_release_load_only_injections,
        toggle="release_load_only_injections",
        needs_derived=True,
    ),
    Step(
        "relax_generation_min_to_injection",
        "Снять Pmin, противоречащий инжекции",
        _G_INPUT,
        "relax_generation_min_to_injection: a minimum generation above what the "
        "node P injection reading allows is lowered to the reading.",
        _s_relax_generation_min_to_injection,
        toggle="relax_generation_min_to_injection",
        needs_derived=True,
    ),
    Step(
        "widen_generation_only_injections",
        "Учесть ящик нагрузки в инжекции из генерации",
        _G_INPUT,
        "widen_generation_only_injections: PG without PN (QG without QN) on a node "
        "with a load box constrains PG - load, load within the box.",
        _s_widen_generation_only_injections,
        toggle="widen_generation_only_injections",
        needs_derived=True,
    ),
    Step(
        "gen_v_calibration",
        "Калибровка σ² V ген-узлов",
        _G_INPUT,
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
        "mirror_voltage",
        "Зеркалить V через ktr=1",
        _G_PSEUDO,
        "mirror_voltage_through_unit_tap_links.",
        _s_mirror_voltage,
        toggle="mirror_voltage_unit_tap",
    ),
    Step(
        "materialize",
        "Observed operating point materialization",
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
        _G_INPUT,
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
    # ORDER INVARIANT: bad_data_repass → v_refine → v_mirror → anti_overshoot
    # strictly BETWEEN estimate and the end. bad_data fixes gross errors
    # (values/statuses) from the residuals of the first solve; v_refine tightens
    # σ of consistent V measurements on the already corrected residuals;
    # v_mirror moves the pseudo-V value of blind clusters to the level of their
    # boundary; anti-overshoot polishes the result.
    Step(
        "bad_data_repass",
        "Bad-data re-pass",
        _G_POST,
        "classify_bad_data по остаткам solve + warm re-solve на правленых мерах.",
        _s_bad_data_repass,
        toggle="bad_data",
    ),
    Step(
        "v_refine",
        "V-refine ужесточение",
        _G_POST,
        "classify_v_refine согласованных V по остаткам + warm re-solve на ужесточённых σ.",
        _s_v_refine,
        toggle="v_refine",
    ),
    Step(
        "v_mirror",
        "V-mirror значение слепых pseudo-V",
        _G_POST,
        "classify_v_mirror: pseudo-V слепых кластеров → median pu границы × Vnom + warm re-solve.",
        _s_v_mirror,
        toggle="v_mirror",
    ),
    Step(
        "v_mirror_chain",
        "Chain-mirror pseudo-V глубоких цепочек",
        _G_POST,
        "classify_chain_mirror: pseudo-V флэт-плейсхолдеров на d≥1 от real-V → "
        "pu источника с затуханием decay^(d−1) + warm re-solve.",
        _s_v_mirror_chain,
        toggle="v_mirror_chain",
    ),
    Step(
        "refine_merged",
        "Слитый refine-re-solve (перф)",
        _G_POST,
        "refine_merge: планы v_refine+v_mirror+chain по одному решению, один warm re-solve.",
        _s_refine_merged,
        toggle="refine_merge",
    ),
    Step(
        "anti_overshoot",
        "Anti-overshoot уточнение",
        _G_POST,
        "refine_anti_overshoot со само-валидацией (revert).",
        _s_anti_overshoot,
        toggle="anti_overshoot",
    ),
]
