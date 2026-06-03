"""Схема контракта данных SE: ``SEInput`` / ``SEOutput``.

Это **декларация** контракта — формализованный, минимальный набор именованных
таблиц с зафиксированными колонками и типами, которые оценка состояния (SE)
реально читает и пишет. Схема фиксирует границу данных на входе и выходе SE.

**Минимальность.** В контракт входят ТОЛЬКО колонки, которые gridstate реально
трогает. Возможный «хвост» полноразмерной модели сети (динамика генераторов,
противоаварийная автоматика, сечения, виртуальные группы,
``kct``/``ves``/``type_ekv`` и т.п.) намеренно НЕ часть контракта.

**Роли колонок** (:class:`Role`) кодируют трёхслойную модель плана:

* :attr:`Role.KEY` — идентичность (``id``, ``from_node`` …), не меняется.
* :attr:`Role.INPUT` — read-only вход: SE читает, но НЕ мутирует.
* :attr:`Role.WORKING` — вход, который препроцессинг **мутирует** по ходу
  прогона (статусы, режим, шунты, tap, начальные V/δ). Эти колонки копируются в
  явный рабочий слой (``Working``), поэтому роль выделена отдельно.
* :attr:`Role.OUTPUT` — результат солвера; живёт в `SEOutput`, отдельно от входа.

``SEInput`` = таблицы с колонками ролей {KEY, INPUT, WORKING} + «сырые» таблицы;
``SEOutput`` = таблицы с колонками ролей {KEY, OUTPUT}. Колонка ``voltage_magnitude``
присутствует и там, и там (во входе — начальное приближение, роль WORKING; на
выходе — решённое значение, роль OUTPUT): это разные слои, конфликта нет.

**Типы — у нас.** dtype-строки колонок зафиксированы здесь: контракт
самодостаточен и является каноническим источником типов. Тест-страж
``tests/test_contract.py`` следит, чтобы контракт оставался верен тому, что код
фактически читает/пишет.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Role(str, Enum):
    """Роль колонки в трёхслойной модели контракта (см. модульную доку)."""

    KEY = "key"
    INPUT = "input"
    WORKING = "working"
    OUTPUT = "output"


# Роли, формирующие входной слой (``SEInput``) и выходной слой (``SEOutput``).
_INPUT_ROLES: tuple[Role, ...] = (Role.KEY, Role.INPUT, Role.WORKING)
_OUTPUT_ROLES: tuple[Role, ...] = (Role.KEY, Role.OUTPUT)


@dataclass(frozen=True)
class ColumnSpec:
    """Одна колонка таблицы контракта.

    Attributes:
        name: имя колонки (== имя атрибута объекта в текущей модели).
        dtype: numpy-dtype-строка (``"f8"``, ``"i4"``, ``"bool"``, ``"U64"`` …).
        role: :class:`Role` — слой/мутабельность.
        required: обязана ли присутствовать во входных данных (для INPUT/WORKING)
            либо гарантированно заполняться (для OUTPUT). ``False`` — допустимо
            отсутствие/дефолт.
        doc: краткое назначение и кто из шагов SE её читает/пишет.
    """

    name: str
    dtype: str
    role: Role
    required: bool = True
    doc: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Схема одной таблицы контракта (узлы/ветви/меры/генераторы/сырые)."""

    name: str
    key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    doc: str = ""

    def column(self, name: str) -> ColumnSpec | None:
        """Спека колонки по имени (или ``None``)."""
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def columns_by_role(self, *roles: Role) -> tuple[ColumnSpec, ...]:
        """Колонки выбранных ролей (порядок объявления сохраняется)."""
        wanted = set(roles)
        return tuple(c for c in self.columns if c.role in wanted)

    def column_names(self, *roles: Role) -> tuple[str, ...]:
        """Имена колонок выбранных ролей (или всех, если ``roles`` пуст)."""
        if not roles:
            return tuple(c.name for c in self.columns)
        return tuple(c.name for c in self.columns_by_role(*roles))

    def required_names(self, *roles: Role) -> tuple[str, ...]:
        """Имена обязательных колонок выбранных ролей."""
        cols = self.columns_by_role(*roles) if roles else self.columns
        return tuple(c.name for c in cols if c.required)

    def numpy_dtype(self, *roles: Role) -> np.dtype:
        """Собрать ``np.dtype`` из колонок выбранных ролей (или всех)."""
        cols = self.columns_by_role(*roles) if roles else self.columns
        return np.dtype([(c.name, c.dtype) for c in cols])

    def input_dtype(self) -> np.dtype:
        """``np.dtype`` входного слоя таблицы (роли KEY/INPUT/WORKING)."""
        return self.numpy_dtype(*_INPUT_ROLES)

    def output_dtype(self) -> np.dtype:
        """``np.dtype`` выходного слоя таблицы (роли KEY/OUTPUT)."""
        return self.numpy_dtype(*_OUTPUT_ROLES)


# ===========================================================================
# Узлы
# ===========================================================================

NODES = TableSchema(
    name="nodes",
    key=("id",),
    doc="Узлы сети. Идентичность — id.",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор узла."),
        # --- read-only вход ---
        ColumnSpec("name", "U64", Role.INPUT, required=False, doc="Имя узла (диагностика/вывод)."),
        ColumnSpec(
            "area_id",
            "i4",
            Role.INPUT,
            required=False,
            doc="Район — материализация режима по area-distribution.",
        ),
        ColumnSpec(
            "voltage_nominal",
            "f8",
            Role.INPUT,
            doc="Vном, кВ — база pu-перевода (читается повсеместно).",
        ),
        ColumnSpec(
            "voltage_min", "f8", Role.INPUT, required=False, doc="U_MIN — фильтр V вне диапазона."
        ),
        ColumnSpec(
            "voltage_max", "f8", Role.INPUT, required=False, doc="U_MAX — фильтр V вне диапазона."
        ),
        ColumnSpec(
            "voltage_critical",
            "f8",
            Role.INPUT,
            required=False,
            doc="U_KRIT — нижний порог фильтра V.",
        ),
        ColumnSpec(
            "voltage_setpoint",
            "f8",
            Role.INPUT,
            required=False,
            doc="U_ZAD — заданное V для PV-якоря.",
        ),
        ColumnSpec(
            "exist_load",
            "i1",
            Role.INPUT,
            required=False,
            doc="Узел может нести нагрузку — write-split p_inj→load, IPM box.",
        ),
        ColumnSpec(
            "exist_gen",
            "i1",
            Role.INPUT,
            required=False,
            doc="Узел может нести генерацию — write-split, IPM box, PV-promotion.",
        ),
        ColumnSpec(
            "sxn_id",
            "i4",
            Role.INPUT,
            required=False,
            doc="Ссылка на load_models — характеристика P(V)/Q(V).",
        ),
        ColumnSpec(
            "load_p_min", "f8", Role.INPUT, required=False, doc="Нижняя P-нагрузка — IPM box."
        ),
        ColumnSpec(
            "load_p_max",
            "f8",
            Role.INPUT,
            required=False,
            doc="Верхняя P-нагрузка — IPM box + clamp материализации.",
        ),
        ColumnSpec(
            "load_q_min", "f8", Role.INPUT, required=False, doc="Нижняя Q-нагрузка — IPM box."
        ),
        ColumnSpec(
            "load_q_max", "f8", Role.INPUT, required=False, doc="Верхняя Q-нагрузка — IPM box."
        ),
        # --- вход, мутируемый препроцессингом (рабочий слой Фазы 4) ---
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Включён ли узел. Мутируют топология/телеметрия/каскад.",
        ),
        ColumnSpec(
            "node_type",
            "i1",
            Role.WORKING,
            doc="0-PQ/1-PV/2-Slack. Мутируют refine_slack/refine_node_types.",
        ),
        ColumnSpec(
            "balance_priority",
            "i4",
            Role.WORKING,
            doc="Приоритет балансирования; читает refine_slack, пишет телеметрия.",
        ),
        ColumnSpec(
            "shunt_g", "f8", Role.WORKING, doc="G шунта, См. Пишет apply_reactors/one_sided."
        ),
        ColumnSpec(
            "shunt_b", "f8", Role.WORKING, doc="B шунта, См. Пишет apply_reactors (ШР→шунт)."
        ),
        ColumnSpec(
            "generation_p",
            "f8",
            Role.WORKING,
            doc="P-генерация, МВт — режим. Пишут материализация/агрегация генераторов.",
        ),
        ColumnSpec("generation_q", "f8", Role.WORKING, doc="Q-генерация, МВАр — режим."),
        ColumnSpec(
            "generation_p_min",
            "f8",
            Role.WORKING,
            doc="Нижняя P-генерация — IPM box; перезаписывает агрегация генераторов.",
        ),
        ColumnSpec(
            "generation_p_max", "f8", Role.WORKING, doc="Верхняя P-генерация — IPM box/агрегация."
        ),
        ColumnSpec(
            "generation_q_min", "f8", Role.WORKING, doc="Нижняя Q-генерация — IPM box/агрегация."
        ),
        ColumnSpec(
            "generation_q_max", "f8", Role.WORKING, doc="Верхняя Q-генерация — IPM box/агрегация."
        ),
        ColumnSpec(
            "load_p", "f8", Role.WORKING, doc="P-нагрузка, МВт — режим. Пишет материализация."
        ),
        ColumnSpec(
            "load_q", "f8", Role.WORKING, doc="Q-нагрузка, МВАр — режим. Пишет материализация."
        ),
        ColumnSpec(
            "voltage_magnitude",
            "f8",
            Role.WORKING,
            doc="Модуль V, кВ — начальное приближение/тёплый старт; солвер перезаписывает.",
        ),
        ColumnSpec(
            "voltage_angle",
            "f8",
            Role.WORKING,
            doc="Угол V, рад — начальное приближение; солвер перезаписывает.",
        ),
    ),
)


# Выходной слой узлов (``SEOutput.nodes``) — отдельное пространство results.
# V/δ присутствуют и здесь (решённые значения), и во входе NODES (роль WORKING,
# начальное приближение): разные слои, поэтому конфликта ролей нет.
NODES_OUTPUT = TableSchema(
    name="nodes",
    key=("id",),
    doc="Результат SE по узлам (V/δ + инжекции/небалансы/оценки нагрузки-генерации).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор узла."),
        ColumnSpec("voltage_magnitude", "f8", Role.OUTPUT, doc="Решённый модуль V, кВ."),
        ColumnSpec("voltage_angle", "f8", Role.OUTPUT, doc="Решённый угол V, рад."),
        ColumnSpec("p_inj_calc", "f8", Role.OUTPUT, doc="P-инжекция узла из решения SE, МВт."),
        ColumnSpec("q_inj_calc", "f8", Role.OUTPUT, doc="Q-инжекция узла из решения SE, МВАр."),
        ColumnSpec(
            "imbalance_p", "f8", Role.OUTPUT, doc="Небаланс P = p_inj_calc − (gen_p − load_p)."
        ),
        ColumnSpec("imbalance_q", "f8", Role.OUTPUT, doc="Небаланс Q."),
        ColumnSpec(
            "load_p_estimated", "f8", Role.OUTPUT, doc="Фактическая P-нагрузка после SE, МВт."
        ),
        ColumnSpec(
            "load_q_estimated", "f8", Role.OUTPUT, doc="Фактическая Q-нагрузка после SE, МВАр."
        ),
        ColumnSpec(
            "generation_p_estimated", "f8", Role.OUTPUT, doc="Фактическая P-генерация, МВт."
        ),
        ColumnSpec(
            "generation_q_estimated", "f8", Role.OUTPUT, doc="Фактическая Q-генерация, МВАр."
        ),
    ),
)


# ===========================================================================
# Ветви
# ===========================================================================

BRANCHES = TableSchema(
    name="branches",
    key=("id", "from_node", "to_node", "parallel_id"),
    doc="Ветви (ВЛ/трансформаторы/реакторы). Идентичность — id (+from/to/parallel).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор ветви."),
        ColumnSpec("from_node", "i4", Role.KEY, doc="Начальный узел."),
        ColumnSpec("to_node", "i4", Role.KEY, doc="Конечный узел."),
        ColumnSpec("parallel_id", "i2", Role.KEY, doc="Номер параллельной ветви."),
        # --- read-only вход ---
        ColumnSpec(
            "name", "U128", Role.INPUT, required=False, doc="Имя ветви (диагностика/вывод)."
        ),
        ColumnSpec("branch_type", "i1", Role.INPUT, doc="0-линия/1-трансформатор/2-реактор."),
        ColumnSpec(
            "current_limit_normal",
            "f8",
            Role.INPUT,
            required=False,
            doc="Токовое ограничение, А — для loading_pct.",
        ),
        ColumnSpec(
            "ti_p_from", "i4", Role.INPUT, required=False, doc="Ссылка ТИ P-нач (детект стороны)."
        ),
        ColumnSpec("ti_q_from", "i4", Role.INPUT, required=False, doc="Ссылка ТИ Q-нач."),
        ColumnSpec("ti_p_to", "i4", Role.INPUT, required=False, doc="Ссылка ТИ P-кон."),
        ColumnSpec("ti_q_to", "i4", Role.INPUT, required=False, doc="Ссылка ТИ Q-кон."),
        # --- вход, мутируемый препроцессингом ---
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Включена ли ветвь. Мутируют топология/каскад/телеметрия.",
        ),
        ColumnSpec(
            "resistance",
            "f8",
            Role.WORKING,
            doc="R, Ом. Мутирует normalize_breaker_reactance (R=X=0 КЗ).",
        ),
        ColumnSpec(
            "reactance", "f8", Role.WORKING, doc="X, Ом. Мутирует normalize_breaker_reactance."
        ),
        ColumnSpec("conductance", "f8", Role.WORKING, doc="G серии, См. Пересчёт при смене tap."),
        ColumnSpec("susceptance", "f8", Role.WORKING, doc="B серии, См. Пересчёт при смене tap."),
        ColumnSpec("conductance_from", "f8", Role.WORKING, doc="G шунта в начале, См."),
        ColumnSpec("susceptance_from", "f8", Role.WORKING, doc="B шунта в начале, См."),
        ColumnSpec("conductance_to", "f8", Role.WORKING, doc="G шунта в конце, См."),
        ColumnSpec("susceptance_to", "f8", Role.WORKING, doc="B шунта в конце, См."),
        ColumnSpec("tap_ratio", "f8", Role.WORKING, doc="Коэф. трансформации. Мутирует apply_rpn."),
        ColumnSpec("phase_shift", "f8", Role.WORKING, doc="Сдвиг фаз, рад. Мутирует apply_rpn."),
    ),
)


# Выходной слой ветвей (``SEOutput.branches``) — перетоки/токи/потери/загрузка.
BRANCHES_OUTPUT = TableSchema(
    name="branches",
    key=("id",),
    doc="Результат SE по ветвям (перетоки P/Q с обеих сторон, токи, потери, загрузка).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор ветви."),
        ColumnSpec("power_from_p", "f8", Role.OUTPUT, doc="P в начале, МВт."),
        ColumnSpec("power_from_q", "f8", Role.OUTPUT, doc="Q в начале, МВАр."),
        ColumnSpec("power_to_p", "f8", Role.OUTPUT, doc="P в конце, МВт."),
        ColumnSpec("power_to_q", "f8", Role.OUTPUT, doc="Q в конце, МВАр."),
        ColumnSpec("current_from", "f8", Role.OUTPUT, doc="Ток в начале, А."),
        ColumnSpec("current_to", "f8", Role.OUTPUT, doc="Ток в конце, А."),
        ColumnSpec("loss_p", "f8", Role.OUTPUT, doc="Активные потери, МВт."),
        ColumnSpec("loss_q", "f8", Role.OUTPUT, doc="Реактивные потери, МВАр."),
        ColumnSpec("loading_pct", "f8", Role.OUTPUT, doc="Загрузка по току, %."),
    ),
)


# ===========================================================================
# Измерения (z-вектор + оценки)
# ===========================================================================

MEASUREMENTS = TableSchema(
    name="measurements",
    key=("id", "object_type", "object_id"),
    doc=(
        "Телеизмерения = z-вектор. В XML-пути почти целиком СТРОЯТСЯ препроцессингом "
        "(FORMULE→меры + синтетика), в пути эталонной SE/прямом пути — поставляются адаптером. "
        "Большинство колонок поэтому WORKING."
    ),
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор измерения."),
        ColumnSpec("object_type", "i1", Role.KEY, doc="0-узел/1-ветвь/2-генератор."),
        ColumnSpec("object_id", "i4", Role.KEY, doc="ID объекта измерения."),
        # --- read-only вход (определяющие/провенанс) ---
        ColumnSpec("measurement_type", "i1", Role.INPUT, doc="0-P/1-Q/2-U/3-I/4-Pinj/5-Qinj."),
        ColumnSpec(
            "min_value", "f8", Role.INPUT, required=False, doc="Нижняя достоверная граница."
        ),
        ColumnSpec(
            "max_value", "f8", Role.INPUT, required=False, doc="Верхняя достоверная граница."
        ),
        ColumnSpec("name", "U128", Role.INPUT, required=False, doc="Имя измерения."),
        ColumnSpec("formula", "U256", Role.INPUT, required=False, doc="FORMULE из XML."),
        ColumnSpec(
            "source_numer", "i4", Role.INPUT, required=False, doc="NUMER аргумента источника."
        ),
        ColumnSpec(
            "tip_ti", "U16", Role.INPUT, required=False, doc="Категория ТИ из эталонной SE."
        ),
        ColumnSpec("prv_num", "U16", Role.INPUT, required=False, doc="Номер провайдера."),
        ColumnSpec("validity_timeout", "i4", Role.INPUT, required=False, doc="VALIDITYTIMEOUTSEC."),
        ColumnSpec("guid_measurement", "U40", Role.INPUT, required=False, doc="GUID измерения."),
        # --- вход, мутируемый/деривируемый препроцессингом ---
        ColumnSpec(
            "value", "f8", Role.WORKING, doc="Значение меры. Пишут телеметрия/материализация."
        ),
        ColumnSpec(
            "variance",
            "f8",
            Role.WORKING,
            doc="Дисперсия σ². Пишут σ_Q-charging/фильтры. Солвер строит R из неё.",
        ),
        ColumnSpec(
            "weight",
            "f8",
            Role.WORKING,
            required=False,
            doc="Вес = 1/σ² (производный). Пишут телеметрия/фильтры; солвер читает variance, не weight.",
        ),
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Активно ли в z-векторе. Глушится/включается препроцессингом.",
        ),
        ColumnSpec(
            "quality", "i1", Role.WORKING, doc="0-хор/1-сомн/2-плох. Пишут фильтры/bad-data."
        ),
        ColumnSpec(
            "branch_side", "i1", Role.WORKING, doc="0-from/1-to/-1-N/A (Yf vs Yt для P_to/Q_to)."
        ),
        ColumnSpec("is_pseudo", "bool", Role.WORKING, doc="Псевдо-приор (add_pseudo/синтез)."),
        ColumnSpec("filter_flag", "i1", Role.WORKING, doc="Причина деактивации (0=ok, …)."),
        ColumnSpec("source_code", "U16", Role.WORKING, doc="Код источника СКАДА (CK2011)."),
        ColumnSpec("source_guid", "U40", Role.WORKING, doc="GUID источника (CKGUID)."),
    ),
)


# Выходной слой измерений (``SEOutput.measurements``) — оценки и невязки.
MEASUREMENTS_OUTPUT = TableSchema(
    name="measurements",
    key=("id",),
    doc="Результат SE по измерениям (оценённое значение + невязка).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор измерения."),
        ColumnSpec("estimated_si", "f8", Role.OUTPUT, doc="Оценка нашего SE, исходные единицы."),
        ColumnSpec("estimated_value", "f8", Role.OUTPUT, doc="Универсальное оценённое значение."),
        ColumnSpec("residual", "f8", Role.OUTPUT, doc="Невязка value − estimated."),
    ),
)


# ===========================================================================
# Генераторы (вход; результаты сворачиваются в node generation_*_estimated)
# ===========================================================================

GENERATORS = TableSchema(
    name="generators",
    key=("id", "node_id"),
    doc="Генераторы. SE читает мощности/границы, мутирует только status (каскад от узла).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Уникальный идентификатор генератора."),
        ColumnSpec("node_id", "i4", Role.KEY, doc="Узел подключения."),
        ColumnSpec("power_output", "f8", Role.INPUT, doc="Выходная P, МВт (→ node gen-инжекция)."),
        ColumnSpec("reactive_output", "f8", Role.INPUT, doc="Выходная Q, МВАр."),
        ColumnSpec("power_min", "f8", Role.INPUT, required=False, doc="Нижняя P — box/агрегация."),
        ColumnSpec("power_max", "f8", Role.INPUT, required=False, doc="Верхняя P — box/агрегация."),
        ColumnSpec(
            "reactive_min", "f8", Role.INPUT, required=False, doc="Нижняя Q — box/агрегация."
        ),
        ColumnSpec(
            "reactive_max", "f8", Role.INPUT, required=False, doc="Верхняя Q — box/агрегация."
        ),
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Включён ли. Мутирует apply_generator_status_from_node (node off ⇒ gen off).",
        ),
    ),
)


# ===========================================================================
# Сырые таблицы (SEInput.raw) — то, что читает core-пайплайн
# ===========================================================================
#
# ПРИМЕЧАНИЕ: FORMULE/ON_LINE/ARG (топология, телеметрия, РПН-спеки,
# материализация) исторически разбирались напрямую внешним адаптером загрузки, а
# их типизированная проекция в raw_tables (``nested_formulas``/``formula_args``)
# lossy (теряется GENERATOR.gen_num, схлопываются NP/PARALLEL у LINE). Поэтому
# полноценный FORMULE-вход в контракт оставлен на будущее обогащение адаптера.


@dataclass(frozen=True)
class RawTableSpec:
    """Сырая таблица входа — подмножество колонок, которое читает core-SE."""

    name: str
    key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    required: bool = False
    doc: str = ""

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def numpy_dtype(self) -> np.dtype:
        return np.dtype([(c.name, c.dtype) for c in self.columns])


def _raw_cols(*specs: tuple[str, str, str]) -> tuple[ColumnSpec, ...]:
    """Хелпер: список ``(name, dtype, doc)`` → колонки роли INPUT (сырые read-only)."""
    return tuple(ColumnSpec(n, d, Role.INPUT, required=False, doc=doc) for n, d, doc in specs)


RAW_TABLES: tuple[RawTableSpec, ...] = (
    RawTableSpec(
        "reactors",
        key=("node_id",),
        required=False,
        doc="ШР → node.shunt_b (apply_reactors_to_node_shunt).",
        columns=_raw_cols(
            ("node_id", "i4", "Узел подключения реактора."),
            ("status", "bool", "Вкл/выкл реактора."),
            ("conductance", "f8", "G, мкСм."),
            ("susceptance", "f8", "B, мкСм (ШР индуктивный)."),
        ),
    ),
    RawTableSpec(
        "tm_values",
        key=("ckguid",),
        required=False,
        doc="Снимок реальной телеметрии (guid→значение).",
        columns=_raw_cols(
            ("ckguid", "U64", "GUID источника СКАДА."),
            ("value", "f8", "Текущее значение замера."),
            ("quality_code", "u4", "Код качества."),
            ("utc_dt_of_value", "U32", "Временная метка значения."),
        ),
    ),
    RawTableSpec(
        "shema_ktr",
        key=("type_rpn", "num_a", "num_r"),
        required=False,
        doc="Таблица отводов РПН/ПБВ → tap_ratio.",
        columns=_raw_cols(
            ("type_rpn", "i4", "Тип РПН."),
            ("num_a", "i4", "Номер отвода (анцапфа)."),
            ("num_r", "i4", "Номер регулировочной ступени."),
            ("ktr_a", "f8", "Коэф. трансформации по анцапфе."),
            ("ktr_r", "f8", "Коэф. по регулировочной ступени."),
            ("ktr_a_vc", "f8", "Коэф. по анцапфе (вольтодобавка)."),
            ("ktr_r_vc", "f8", "Коэф. по ступени (вольтодобавка)."),
        ),
    ),
    RawTableSpec(
        "load_models",
        key=("id",),
        required=False,
        doc="Статические характеристики P(V)/Q(V) (apply_load_characteristic).",
        columns=_raw_cols(
            ("id", "i4", "ID модели (узел.sxn_id, 1-based)."),
            ("coeff_p_a0", "f8", "P0·a0 — постоянная составляющая."),
            ("coeff_p_a1", "f8", "Линейная по U."),
            ("coeff_p_a2", "f8", "Квадратичная по U."),
            ("coeff_q_b0", "f8", "Q0·b0 — постоянная."),
            ("coeff_q_b1", "f8", "Линейная по U."),
            ("coeff_q_b2", "f8", "Квадратичная по U."),
        ),
    ),
)


# ===========================================================================
# Контракт верхнего уровня
# ===========================================================================


@dataclass(frozen=True)
class SEInputSchema:
    """Схема входного контракта: набор таблиц (роли KEY/INPUT/WORKING) + сырые."""

    nodes: TableSchema
    branches: TableSchema
    measurements: TableSchema
    generators: TableSchema
    raw: tuple[RawTableSpec, ...]

    def tables(self) -> tuple[TableSchema, ...]:
        return (self.nodes, self.branches, self.measurements, self.generators)

    def raw_table(self, name: str) -> RawTableSpec | None:
        for rt in self.raw:
            if rt.name == name:
                return rt
        return None


@dataclass(frozen=True)
class SEOutputSchema:
    """Схема выходного контракта: результаты (роли KEY/OUTPUT), keyed по id."""

    nodes: TableSchema
    branches: TableSchema
    measurements: TableSchema

    def tables(self) -> tuple[TableSchema, ...]:
        return (self.nodes, self.branches, self.measurements)


SE_INPUT = SEInputSchema(
    nodes=NODES,
    branches=BRANCHES,
    measurements=MEASUREMENTS,
    generators=GENERATORS,
    raw=RAW_TABLES,
)

SE_OUTPUT = SEOutputSchema(
    nodes=NODES_OUTPUT,
    branches=BRANCHES_OUTPUT,
    measurements=MEASUREMENTS_OUTPUT,
)
