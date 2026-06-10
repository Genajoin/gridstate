"""Prep-слой телеметрии: применение готовых числовых планов к контрактным таблицам.

Адаптеры применения (``apply_*_resolved``) поверх контрактных ядер (``_apply_*_on_arrays``):
телеметрия/z-вектор, ON_LINE-топология, РПН/ПБВ, материализация узлового режима, Vnom.
Плюс независимые prep-шаги: агрегация генераторов, реакторы→шунт, нормализация
короткозамыкателей, слияние дублей, фильтры качества/V/Q-потерь. Все они работают над
числовыми массивами ``SE_INPUT`` (без формат-слоя источника).
"""

from gridstate.telemetry.apply_resolved import (
    apply_materialize_resolved,
    apply_telemetry_resolved,
)
from gridstate.telemetry.loss_filter import (
    BranchLossReport,
    analyze_branch_loss_consistency,
    compute_branch_loss_formulas,
)
from gridstate.telemetry.on_line import apply_topology_resolved
from gridstate.telemetry.quality import (
    QUALITY_BAD,
    QUALITY_GOOD,
    QUALITY_QUESTIONABLE,
    aggregate_qualities,
    inverse_classifier,
    passthrough_classifier,
    strict_classifier,
    tm_code_histogram,
)
from gridstate.telemetry.rpn import apply_rpn_resolved
from gridstate.telemetry.topology import (
    aggregate_generators_to_node,
    apply_generator_status_from_node,
    apply_reactors_to_node_shunt,
    deactivate_orphan_measurements,
    normalize_breaker_reactance,
    resolve_merged_measurement_conflicts,
)
from gridstate.telemetry.units import (
    normalize_guid,
    variance_branch_q,
    variance_power,
    variance_voltage,
)
from gridstate.telemetry.voltage_filter import (
    apply_voltage_meas_calibration_for_gen_nodes,
    apply_voltage_range_filter,
)
from gridstate.telemetry.voltage_nominal import apply_voltage_nominal_resolved


__all__ = [
    "QUALITY_BAD",
    "QUALITY_GOOD",
    "QUALITY_QUESTIONABLE",
    "BranchLossReport",
    "aggregate_generators_to_node",
    "aggregate_qualities",
    "analyze_branch_loss_consistency",
    "apply_generator_status_from_node",
    "apply_materialize_resolved",
    "apply_reactors_to_node_shunt",
    "apply_rpn_resolved",
    "apply_telemetry_resolved",
    "apply_topology_resolved",
    "apply_voltage_meas_calibration_for_gen_nodes",
    "apply_voltage_nominal_resolved",
    "apply_voltage_range_filter",
    "compute_branch_loss_formulas",
    "deactivate_orphan_measurements",
    "inverse_classifier",
    "normalize_breaker_reactance",
    "normalize_guid",
    "passthrough_classifier",
    "resolve_merged_measurement_conflicts",
    "strict_classifier",
    "tm_code_histogram",
    "variance_branch_q",
    "variance_power",
    "variance_voltage",
]
