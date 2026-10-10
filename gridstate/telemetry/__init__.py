"""Telemetry prep layer: apply ready numeric plans to the contract tables.

Application adapters (``apply_*_resolved``) over the contract cores (``_apply_*_on_arrays``):
telemetry/z-vector, ON_LINE topology, tap changers, nodal-regime materialization, Vnom.
Plus independent prep steps: generator aggregation, reactors→shunt, normalization of
short-circuit links, quality / V / Q-loss filters. All of them operate on the
numeric ``SE_INPUT`` arrays (without the source format layer).
"""

from gridstate.telemetry.apply_resolved import (
    apply_materialize_resolved,
    apply_telemetry_resolved,
)
from gridstate.telemetry.generators import (
    aggregate_generators_to_node,
    apply_generator_status_from_node,
)
from gridstate.telemetry.measurements import deactivate_orphan_measurements
from gridstate.telemetry.noise import variance_branch_q, variance_power, variance_voltage
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
from gridstate.telemetry.shunts import (
    apply_reactors_to_node_shunt,
    normalize_breaker_reactance,
)
from gridstate.telemetry.sigma_floor import apply_flow_sigma_floor
from gridstate.telemetry.units import normalize_guid
from gridstate.telemetry.voltage_filter import (
    apply_voltage_meas_calibration_for_gen_nodes,
    apply_voltage_range_filter,
)
from gridstate.telemetry.voltage_nominal import apply_voltage_nominal_resolved


__all__ = [
    "QUALITY_BAD",
    "QUALITY_GOOD",
    "QUALITY_QUESTIONABLE",
    "aggregate_generators_to_node",
    "aggregate_qualities",
    "apply_flow_sigma_floor",
    "apply_generator_status_from_node",
    "apply_materialize_resolved",
    "apply_reactors_to_node_shunt",
    "apply_rpn_resolved",
    "apply_telemetry_resolved",
    "apply_topology_resolved",
    "apply_voltage_meas_calibration_for_gen_nodes",
    "apply_voltage_nominal_resolved",
    "apply_voltage_range_filter",
    "deactivate_orphan_measurements",
    "inverse_classifier",
    "normalize_breaker_reactance",
    "normalize_guid",
    "passthrough_classifier",
    "strict_classifier",
    "tm_code_histogram",
    "variance_branch_q",
    "variance_power",
    "variance_voltage",
]
