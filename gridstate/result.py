"""Результат оценки состояния — возвращается ``estimate()``.

Ссылка на рабочую модель (``working``-слой) хранится в ``SEResult.model``.
При вызове через :func:`gridstate.pipeline.run` это **копия** входной модели:
вход остаётся read-only, выход живёт в рабочей копии и в ``SEResult``. При
прямом вызове ``estimate(model)`` модель обновляется in-place (низкоуровневый
движок), и ``SEResult.model`` — та же ссылка.

**Output-слой (``outputs``)** — :class:`OutputTables` со структурированными
массивами (узлы/ветви/меры), keyed по id, параллельными Input-коллекциям. Это
канонический Output-контракт для внешних адаптеров (UI/CLI): джойнят Input +
``outputs`` по id для отображения. Поля ``v_pu`` / ``delta_rad`` дублируют V/δ в
p.u./радианах — для отладки и тестов.

Дополнительные поля качества (``chi2``, ``worst_residuals``,
``worst_imbalance``, ``observability_warnings``) заполняются ``estimate()``
после ``write_results_to_model`` — это структурированная сводка для
backward-compat диагностики (раньше собиралась ad-hoc в
``examples/se_xml_inline_snapshot.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from gridstate.working import Working


# Поля выхода солвера по слоям модели — Output-контейнер ``OutputTables``
# извлекает ровно их из рабочей модели после записи результатов.
NODE_OUTPUT_FIELDS: tuple[str, ...] = (
    "voltage_magnitude",
    "voltage_angle",
    "p_inj_calc",
    "q_inj_calc",
    "imbalance_p",
    "imbalance_q",
    "load_p_estimated",
    "load_q_estimated",
    "generation_p_estimated",
    "generation_q_estimated",
    "solved",
)
BRANCH_OUTPUT_FIELDS: tuple[str, ...] = (
    "power_from_p",
    "power_from_q",
    "power_to_p",
    "power_to_q",
    "current_from",
    "current_to",
    "loss_p",
    "loss_q",
    "loading_pct",
)
MEAS_OUTPUT_FIELDS: tuple[str, ...] = (
    "estimated_si",
    "estimated_value",
    "residual",
)


@dataclass
class Chi2Summary:
    """Сводка χ²-теста на финальном решении SE.

    Attributes:
        value: ``J = Σ(r_i² / σ_i²)`` — целевая функция WLS на сходимости.
        dof: степени свободы ``m − n`` (число измерений минус размер state).
        threshold: критическое значение ``χ²(dof, 1 − α)``, ``α = 0.05``.
            ``NaN`` при ``dof ≤ 0`` (тест несостоятелен).
        passes: ``True``, если ``value ≤ threshold`` (нет признаков bad-data).
    """

    value: float
    dof: int
    threshold: float
    passes: bool


@dataclass
class ResidualRow:
    """Одна запись из топа худших нормированных остатков.

    Attributes:
        measurement_id: ``Measurement.id`` исходного измерения.
        kind: текстовый ярлык типа (``"V"``/``"P"``/``"Q"``/``"I"``/
            ``"P_inj"``/``"Q_inj"``/``"?"``).
        value: исходное значение измерения в исходных единицах
            (кВ/МВт/МВАр/А).
        expected: модельное значение ``h(x)`` в тех же единицах.
        residual: ``value − expected`` в исходных единицах.
        normalized_residual: ``|r| / √diag(Ω)`` — нормированный остаток
            (Abur & Expósito §5.6). ``inf`` для non-redundant измерений.
    """

    measurement_id: int
    kind: str
    value: float
    expected: float
    residual: float
    normalized_residual: float


@dataclass
class ImbalanceRow:
    """Одна запись из топа узловых небалансов.

    Attributes:
        node_id: ``Node.id`` узла.
        imbalance_p_mw: ``p_inj_calc − (generation_p − load_p)`` в МВт.
        imbalance_q_mvar: аналогично для реактивной мощности.
    """

    node_id: int
    imbalance_p_mw: float
    imbalance_q_mvar: float


# Текстовые ярлыки для ``ResidualRow.kind`` — синхронизированы с
# ``gridstate.z_vector.KIND_*`` (повторяем здесь как литералы, чтобы не
# тянуть импорт в result.py).
_RESIDUAL_KIND_LABELS: dict[int, str] = {
    0: "P",  # KIND_POWER_P
    1: "Q",  # KIND_POWER_Q
    2: "V",  # KIND_VOLTAGE
    3: "I",  # KIND_CURRENT
    4: "P_inj",  # KIND_POWER_INJECTION_P
    5: "Q_inj",  # KIND_POWER_INJECTION_Q
    6: "Pbal",  # KIND_NODE_BALANCE_P (IPM)
    7: "Qbal",  # KIND_NODE_BALANCE_Q (IPM)
    8: "Pg_pr",  # KIND_BOX_PRIOR_PGEN (IPM)
    9: "Qg_pr",  # KIND_BOX_PRIOR_QGEN (IPM)
    10: "Pn_pr",  # KIND_BOX_PRIOR_PNAG (IPM)
    11: "Qn_pr",  # KIND_BOX_PRIOR_QNAG (IPM)
}


def _select_output_fields(arr: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    """Собрать структурированный массив ``id`` + ``fields`` из коллекции.

    Берутся только поля, реально присутствующие в ``arr.dtype`` (устойчиво к
    разным версиям DTYPE). Если массив пуст или нет ``id`` — пустой результат.
    """
    names = set(arr.dtype.names or ())
    if "id" not in names:
        return np.empty(0, dtype=[("id", "i8")])
    present = [f for f in fields if f in names]
    out_dtype = [("id", "i8")] + [(f, "f8") for f in present]
    out = np.empty(len(arr), dtype=out_dtype)
    out["id"] = arr["id"].astype(np.int64, copy=False)
    for f in present:
        out[f] = arr[f].astype(np.float64, copy=False)
    return out


@dataclass
class OutputTables:
    """Output-слой SE: результаты, keyed по id, параллельные Input-коллекциям.

    Канонический Output-контракт (Input read-only): внешние адаптеры (UI/CLI)
    джойнят Input + эти таблицы по ``id`` для отображения (узлы/ветви на одной
    странице — входные поля слева, результат справа). Извлекается из рабочей
    модели после записи результатов солвера.

    Attributes:
        nodes: структурированный массив ``id`` + :data:`NODE_OUTPUT_FIELDS`
            (voltage_magnitude/voltage_angle/p_inj_calc/q_inj_calc/imbalance_*/
            load_*_estimated/generation_*_estimated).
        branches: ``id`` + :data:`BRANCH_OUTPUT_FIELDS`
            (power_from/to_p/q, current_from/to, loss_p/q, loading_pct).
        measurements: ``id`` + :data:`MEAS_OUTPUT_FIELDS`
            (estimated_si, estimated_value, residual).
    """

    nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[("id", "i8")]))
    branches: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[("id", "i8")]))
    measurements: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[("id", "i8")]))

    def node(self, node_id: int) -> dict | None:
        """Строка выхода узла как dict (или ``None``, если нет)."""
        return self._row(self.nodes, node_id)

    def branch(self, branch_id: int) -> dict | None:
        """Строка выхода ветви как dict (или ``None``)."""
        return self._row(self.branches, branch_id)

    def measurement(self, meas_id: int) -> dict | None:
        """Строка выхода измерения как dict (или ``None``)."""
        return self._row(self.measurements, meas_id)

    @staticmethod
    def _row(arr: np.ndarray, obj_id: int) -> dict | None:
        if arr.size == 0:
            return None
        hits = np.where(arr["id"] == int(obj_id))[0]
        if hits.size == 0:
            return None
        row = arr[int(hits[0])]
        names = arr.dtype.names or ()
        return {name: row[name].item() for name in names}


def extract_output_tables(model: Working) -> OutputTables:
    """Извлечь :class:`OutputTables` из рабочей модели после записи результатов."""
    return OutputTables(
        nodes=_select_output_fields(model.nodes.to_numpy(), NODE_OUTPUT_FIELDS),
        branches=_select_output_fields(model.branches.to_numpy(), BRANCH_OUTPUT_FIELDS),
        measurements=_select_output_fields(model.measurements.to_numpy(), MEAS_OUTPUT_FIELDS),
    )


@dataclass
class SEResult:
    """Результат одного запуска ``estimate()``.

    Attributes:
        model: ссылка на обновлённый ``Working`` (тот же объект, что
            был передан в ``estimate``).
        success: сошёлся ли итерационный процесс.
        iterations: число выполненных итераций.
        objective_value: значение целевой функции J = rᵀ R⁻¹ r на последней
            итерации.
        algorithm: имя использованного алгоритма ("wls" / "ipm").
        v_pu: (n_nodes,) — модули напряжений в p.u. (дублируют
            ``model.nodes.voltage_magnitude / voltage_nominal``).
        delta_rad: (n_nodes,) — углы напряжений в радианах.
        message: произвольное диагностическое сообщение (причина несходимости
            и т. п.).
        chi2: сводка χ²-теста (``None``, если не считалась).
        worst_residuals: топ-N измерений по ``|r_N|`` (по убыванию).
            Пустой список, если расчёт пропущен / нет измерений.
        worst_imbalance: топ-N узлов по ``|imbalance_p|`` (по убыванию).
        observability_warnings: список ``node_id`` узлов с нулевыми
            столбцами ``H`` (state-переменные не покрыты измерениями).
            Пустой, если анализ не проводился.
    """

    model: Working
    success: bool = False
    iterations: int = 0
    objective_value: float = float("nan")
    algorithm: str = ""
    v_pu: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta_rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    message: str = ""
    # Детализация сходимости. WLS: "converged"/"not_converged". IPM —
    # двухуровневая (см. IPMResult.status): "kkt" (строгая стационарность),
    # "completed" (μ-расписание пройдено, решение пригодно), "stalled",
    # "error". success == True ⇔ status ∈ {kkt, completed, converged}.
    convergence_status: str = ""

    # Output-контейнер — результаты keyed по id (узлы/ветви/меры), параллельно
    # Input-коллекциям. Канонический Output-контракт для адаптеров (Input
    # read-only). Заполняется ``estimate()`` после записи результатов.
    outputs: OutputTables = field(default_factory=OutputTables)

    # Quality summary — новые опциональные поля; default — пустые/None
    # для backward-compatibility.
    chi2: Chi2Summary | None = None
    worst_residuals: list[ResidualRow] = field(default_factory=list)
    worst_imbalance: list[ImbalanceRow] = field(default_factory=list)
    observability_warnings: list[int] = field(default_factory=list)


def label_for_kind(kind_code: int) -> str:
    """Получить текстовый ярлык для числового ``MeasurementIndex.kind``."""
    return _RESIDUAL_KIND_LABELS.get(int(kind_code), "?")
