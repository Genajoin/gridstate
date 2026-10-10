"""Domain enumerations of gridstate (owned by the contract).

gridstate is a self-contained SE module; the domain enumerations (node/branch/
measurement types) are fixed **here** and are the canonical source: the
telemetry/preprocessing modules refer to these constants.

The numeric values of IntEnum are part of the data contract: they take part in
integer comparisons of the z-vector/filters/tables, so they are stable and are not
changed arbitrarily.
"""

from __future__ import annotations

from enum import IntEnum


# Numerical guard against overflow of R⁻¹ = 1/σ² on the per-measurement diagonal of R.
# The value 1e-10 (the historical WLS floor, algorithms/wls.py) removes a silent
# wls(1e-10)/ipm(1e-12) divergence. On the production models min σ² ≈ 1.8e-7, far
# above the floor, so the choice is bit-neutral (no measurement falls into the
# gap [1e-12, 1e-10)).
SIGMA2_FLOOR: float = 1e-10


class NodeType(IntEnum):
    """Node type (the ``node_type`` column of the contract)."""

    PQ = 0
    PV = 1
    SLACK = 2


class BranchType(IntEnum):
    """Branch type (the ``branch_type`` column of the contract)."""

    LINE = 0
    TRANSFORMER = 1
    REACTOR = 2


class MeasurementType(IntEnum):
    """Measurement type (the ``measurement_type`` column of the contract)."""

    POWER_P = 0
    POWER_Q = 1
    VOLTAGE = 2
    CURRENT = 3
    POWER_INJECTION_P = 4
    POWER_INJECTION_Q = 5


class MeasurementQuality(IntEnum):
    """Quality class of a measurement (``quality`` contract column).

    Values mirror the classifier in :mod:`gridstate.telemetry.quality`
    (``QUALITY_GOOD/QUESTIONABLE/BAD`` are kept there as aliases).
    """

    GOOD = 0
    QUESTIONABLE = 1
    BAD = 2


class FilterFlag(IntEnum):
    """Reason a measurement was deactivated/downweighted (``filter_flag`` column).

    Previously these codes lived as bare literals with reminder comments
    scattered across the telemetry filters; the values are part of the data
    contract and must stay stable.
    """

    OK = 0
    BAD_QUALITY = 1
    Q_INCONSISTENCY = 2
    V_LOSS_INCONSISTENT = 3
    V_BELOW_HALF_NOMINAL = 4
    P_SIGN_INCONSISTENCY = 5
    LOAD_ONLY_INJECTION = 6
    GENERATION_ONLY_INJECTION = 7


class MeasurementObjectType(IntEnum):
    """Measured object type (the ``object_type`` column of the contract)."""

    NODE = 0
    BRANCH = 1
    GENERATOR = 2
    REACTOR = 3
    SECTION = 4
    LPA_ELEMENT = 5
    LPA_DEVICE = 6
    AVPROC = 7
    VIR = 8
    TRANZIT = 9
    REMONT = 10
    LPA_SUBELEMENT = 11
    NONE = -1
