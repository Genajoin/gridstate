"""Kind-карты телеметрии: kind-код замера → семантика (object_type / mt / side / sign).

Plain int tables (codes from the ``gridstate.constants`` enums) used by the
telemetry core (``gridstate.telemetry.apply_resolved._apply_telemetry_on_arrays``):
they map the string kind of a reading to its numeric coordinates in
``SE_INPUT.measurements``.

Дата-классы привязок (``ArgEntry`` / ``FormulaSpec`` / ``RpnSpec``) — формат-специфика
приёмного слоя источника и живут во внешнем адаптере данных, а НЕ в публичном ядре:
ядро получает уже вычисленные числовые планы (``DerivedInputs``), не зная их формата.
"""

from __future__ import annotations

from gridstate.constants import MeasurementObjectType, MeasurementType


_NODE = int(MeasurementObjectType.NODE)
_BRANCH = int(MeasurementObjectType.BRANCH)
_P = int(MeasurementType.POWER_P)
_Q = int(MeasurementType.POWER_Q)
_V = int(MeasurementType.VOLTAGE)


# kind → (object_type, measurement_type, branch_side, sign)
# Семантика:
#  - NODE-замер (U) → object_type=0 (NODE), mt=2 V, side=-1
#  - BRANCH-замер (PBEG/PEND/QBEG/QEND) → object_type=1 (BRANCH), mt=0/1, side=0/+1
#  - sign: множитель к value. Для V/branches +1.
_KIND_MAP: dict[str, tuple[int, int, int, int]] = {
    "U": (_NODE, _V, -1, +1),  # NODE V
    "PBEG": (_BRANCH, _P, 0, +1),  # BRANCH P_from
    "PEND": (_BRANCH, _P, +1, +1),  # BRANCH P_to
    "QBEG": (_BRANCH, _Q, 0, +1),  # BRANCH Q_from
    "QEND": (_BRANCH, _Q, +1, +1),  # BRANCH Q_to
}

# kind → (P_or_Q, sign-multiplier к net inj). PG/QG +1; PN/QN -1.
# Применяется через .add() с object_type=0, measurement_type=4 (P_inj) / 5 (Q_inj).
_NODE_INJ_MAP: dict[str, tuple[str, int]] = {
    "PG": ("P", +1),
    "PN": ("P", -1),
    "QG": ("Q", +1),
    "QN": ("Q", -1),
}

_INJ_MT = {
    "P": int(MeasurementType.POWER_INJECTION_P),
    "Q": int(MeasurementType.POWER_INJECTION_Q),
}
