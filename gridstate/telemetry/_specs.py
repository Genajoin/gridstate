"""Спецификации привязок телеметрии + kind-маппинги (prep-слой).

Дата-классы привязок измерения → переменные (``ArgEntry``/``FormulaSpec``/``RpnSpec``)
+ чистые kind→семантика карты (``_KIND_MAP``/``_NODE_INJ_MAP``/``_INJ_MT``). Чистый
stdlib — потребляются контрактными ядрами prep-слоя (``_apply_telemetry_on_arrays`` /
``assign_cod`` / ``_materialize_area_on_arrays``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArgEntry:
    """Одна именованная ARG-привязка переменной к источнику замера."""

    name: str
    guid: str
    invert: bool
    numer: str = ""  # номер SCADA-сигнала (id кластера дублей)


@dataclass
class FormulaSpec:
    """Выражение + список ARG-привязок одного attachment измерения.

    Один attachment содержит:

    * ``formula`` — выражение;
    * 0..N ``ArgEntry`` — переменные выражения.

    Семантика: подставляем ``ARG.name → value(ARG.guid, snapshot)``,
    инвертируем по ``invert``, вычисляем выражение.
    """

    formula: str
    args: list[ArgEntry] = field(default_factory=list)

    @property
    def first_guid(self) -> str:
        return self.args[0].guid if self.args else ""

    @property
    def first_invert(self) -> bool:
        return self.args[0].invert if self.args else False


# Backward-compat alias: одиночный-ARG спецификатор маппится на FormulaSpec.first_*.
ArgRef = FormulaSpec


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


@dataclass(frozen=True)
class RpnSpec:
    """Спецификация РПН (регулирование под нагрузкой) одной ветви.

    Привязывает ``branch_id`` (= OBJ_ID в ``model.branches``) к ARG-ссылкам
    на TM-значения номеров отпаек продольного (``NUM_A``) и поперечного
    (``NUM_R``) РПН и к таблице ``SHEMA_KTR`` через ``type_rpn``.

    Attributes:
        branch_id: OBJ_ID ветви (= ``model.branches.id``).
        type_rpn: тип РПН (ключ в ``SHEMA_KTR``).
        formula_x: выражение продольного № анцапф.
        args_x: ARG-ы выражения X.
        formula_y: выражение поперечного (ВДТ); ``""`` если нет.
        args_y: ARG-ы выражения Y.
    """

    branch_id: int
    type_rpn: int
    formula_x: str
    args_x: tuple[ArgEntry, ...]
    formula_y: str
    args_y: tuple[ArgEntry, ...]
