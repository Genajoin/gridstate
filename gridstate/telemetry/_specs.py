"""Kind-карты телеметрии: kind-код замера → семантика (object_type / mt / side / sign).

Чистые stdlib-таблицы, потребляемые контрактным ядром z-вектора
(``gridstate.telemetry.apply_resolved._apply_telemetry_on_arrays``): связывают
строковый kind замера с числовыми координатами в ``SE_INPUT.measurements``.

Дата-классы привязок (``ArgEntry`` / ``FormulaSpec`` / ``RpnSpec``) — формат-специфика
приёмного слоя источника и живут во внешнем адаптере данных, а НЕ в публичном ядре:
ядро получает уже вычисленные числовые планы (``DerivedInputs``), не зная их формата.
"""

from __future__ import annotations


# kind → (object_type, measurement_type, branch_side, sign)
# Семантика:
#  - NODE-замер (U) → object_type=0 (NODE), mt=2 V, side=-1
#  - BRANCH-замер (PBEG/PEND/QBEG/QEND) → object_type=1 (BRANCH), mt=0/1, side=0/+1
#  - sign: множитель к value. Для V/branches +1.
_KIND_MAP: dict[str, tuple[int, int, int, int]] = {
    "U": (0, 2, -1, +1),  # NODE V
    "PBEG": (1, 0, 0, +1),  # BRANCH P_from
    "PEND": (1, 0, +1, +1),  # BRANCH P_to
    "QBEG": (1, 1, 0, +1),  # BRANCH Q_from
    "QEND": (1, 1, +1, +1),  # BRANCH Q_to
}

# kind → (P_or_Q, sign-multiplier к net inj). PG/QG +1; PN/QN -1.
# Применяется через .add() с object_type=0, measurement_type=4 (P_inj) / 5 (Q_inj).
_NODE_INJ_MAP: dict[str, tuple[str, int]] = {
    "PG": ("P", +1),
    "PN": ("P", -1),
    "QG": ("Q", +1),
    "QN": ("Q", -1),
}

_INJ_MT = {"P": 4, "Q": 5}
