"""Доменные перечисления gridstate (контракт владеет ими).

gridstate — самостоятельный модуль SE; домен-перечисления (типы узлов/ветвей/
измерений) фиксируются **здесь** и являются каноническим источником: модули
телеметрии/препроцессинга ссылаются на эти константы.

Числовые значения IntEnum — часть контракта данных: они участвуют в
целочисленных сравнениях z-вектора/фильтров/таблиц, поэтому стабильны и не
меняются произвольно.
"""

from __future__ import annotations

from enum import IntEnum


# Численный guard от переполнения R⁻¹ = 1/σ² на per-measurement диагонали R.
# Значение 1e-10 (исторический WLS-floor, algorithms/wls.py): устраняет молчаливую
# wls(1e-10)/ipm(1e-12) дивергенцию. На проде (4 ОДУ) min σ²≈1.8e-7 ≫ floor —
# выбор bit-neutral (ни одна мера не попадает в зазор [1e-12, 1e-10)).
SIGMA2_FLOOR: float = 1e-10


class NodeType(IntEnum):
    """Тип узла (``node_type``-колонка контракта)."""

    PQ = 0
    PV = 1
    SLACK = 2


class BranchType(IntEnum):
    """Тип ветви (``branch_type``-колонка контракта)."""

    LINE = 0
    TRANSFORMER = 1
    REACTOR = 2


class MeasurementType(IntEnum):
    """Тип измерения (``measurement_type``-колонка контракта)."""

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


class MeasurementObjectType(IntEnum):
    """Тип объекта измерения (``object_type``-колонка контракта)."""

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
