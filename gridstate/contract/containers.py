"""Runtime containers of the contract: :class:`SEInput` and :class:`SEOutput`.

:class:`SEInput` carries the working layer plus the precomputed numeric plans
(:class:`~gridstate.contract.derived.DerivedInputs`); :class:`SEOutput` wraps
the ``SEResult`` of a run into the output tables keyed by id. The facade that
runs the pipeline on them (:func:`gridstate.runtime.run`) lives outside the
contract package, which stays a leaf: importing it never imports the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.contract.tables import SE_OUTPUT
from gridstate.contract.validate import ValidationReport, validate_input
from gridstate.contract.version import CONTRACT_VERSION


if TYPE_CHECKING:
    from gridstate.contract.derived import DerivedInputs
    from gridstate.result import SEResult


# ---------------------------------------------------------------------------
# Вход
# ---------------------------------------------------------------------------


@dataclass
class SEInput:
    """Входной контракт SE (рантайм-контейнер).

    Attributes:
        model: рабочий слой / носитель входных таблиц (read-only источник).
        derived: предвычисленные числовые планы (:class:`~gridstate.contract.derived.
            DerivedInputs`) — топология/РПН/телеметрия/материализация/Vnom, готовые
            к применению контрактными ядрами. ``None`` — соответствующие шаги
            пропускаются (модель должна уже нести измерения).
        contract_version: версия контракта, под которую собраны данные.
    """

    model: Any
    derived: DerivedInputs | None = None
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        derived: DerivedInputs | None = None,
        contract_version: str = CONTRACT_VERSION,
    ) -> SEInput:
        """Обернуть существующую модель во входной контракт."""
        return cls(
            model=model,
            derived=derived,
            contract_version=contract_version,
        )

    def validate(self, *, strict: bool = False) -> ValidationReport:
        """Проверить входные данные против контракта (см. :func:`validate_input`)."""
        return validate_input(self.model, data_version=self.contract_version, strict=strict)


def load_se_input(
    model: Any,
    *,
    contract_version: str = CONTRACT_VERSION,
) -> SEInput:
    """Обернуть модель (уже несущую измерения) во входной контракт (``derived=None``).

    Граница входа — файл данных ``.npz`` (см. :func:`gridstate.contract.serialize.
    load_se_input_npz`), который восстанавливает рабочий слой + предвычисленные планы
    ``DerivedInputs`` без формат-слоя источника. This helper wraps a ready model;
    the ``run`` steps that need derived inputs are skipped (``derived=None``).
    """
    return SEInput(model=model, derived=None, contract_version=contract_version)


# ---------------------------------------------------------------------------
# Выход
# ---------------------------------------------------------------------------


# name → OUTPUT-схема таблицы (единый источник — SEOutputSchema.tables()).
_SE_OUTPUT_BY_NAME = {t.name: t for t in SE_OUTPUT.tables()}


def _empty_output(table_name: str) -> np.ndarray:
    """Пустой структурированный массив выходного слоя для таблицы ``table_name``."""
    return np.empty(0, dtype=_SE_OUTPUT_BY_NAME[table_name].output_dtype())


@dataclass
class SEOutput:
    """Выходной контракт SE: результаты keyed по id (отдельное пространство results).

    Таблицы соответствуют :data:`gridstate.contract.tables.SE_OUTPUT`:
    ``nodes`` (V/δ + инжекции/небалансы/оценки), ``branches`` (перетоки/токи/
    потери/загрузка), ``measurements`` (оценки/невязки). Плюс скаляры решения.

    Attributes:
        nodes/branches/measurements: структурированные массивы (id + OUTPUT-колонки).
        success/iterations/objective_value/algorithm: метаданные сходимости.
        v_pu/delta_rad: V в p.u. / углы (рад) — дубль для отладки/тестов.
        contract_version: версия контракта результата.
        result: исходный ``SEResult`` — back-compat escape hatch. Несёт ``model``
            (рабочая копия), ``chi2``, ``worst_residuals`` и т.п.
    """

    nodes: np.ndarray = field(default_factory=lambda: _empty_output("nodes"))
    branches: np.ndarray = field(default_factory=lambda: _empty_output("branches"))
    measurements: np.ndarray = field(default_factory=lambda: _empty_output("measurements"))
    success: bool = False
    iterations: int = 0
    objective_value: float = float("nan")
    algorithm: str = ""
    # Детализация success (см. SEResult.convergence_status): для IPM
    # "kkt"/"completed"/"stalled"/"error", для WLS "converged"/"not_converged".
    convergence_status: str = ""
    message: str = ""
    v_pu: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta_rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    contract_version: str = CONTRACT_VERSION
    result: SEResult | None = None

    @classmethod
    def from_result(cls, result: SEResult) -> SEOutput:
        """Собрать :class:`SEOutput` из ``SEResult`` (выходные таблицы + скаляры)."""
        out = result.outputs
        return cls(
            nodes=out.nodes,
            branches=out.branches,
            measurements=out.measurements,
            success=bool(result.success),
            iterations=int(result.iterations),
            objective_value=float(result.objective_value),
            algorithm=str(result.algorithm),
            convergence_status=str(result.convergence_status),
            message=str(result.message),
            v_pu=result.v_pu,
            delta_rad=result.delta_rad,
            result=result,
        )

    def node(self, node_id: int) -> dict | None:
        """Строка выхода узла как dict (или ``None``)."""
        return _row(self.nodes, node_id)

    def branch(self, branch_id: int) -> dict | None:
        """Строка выхода ветви как dict (или ``None``)."""
        return _row(self.branches, branch_id)

    def measurement(self, meas_id: int) -> dict | None:
        """Строка выхода измерения как dict (или ``None``)."""
        return _row(self.measurements, meas_id)


def _row(arr: np.ndarray, obj_id: int) -> dict | None:
    if arr.size == 0:
        return None
    hits = np.where(arr["id"] == int(obj_id))[0]
    if hits.size == 0:
        return None
    row = arr[int(hits[0])]
    return {name: row[name].item() for name in (arr.dtype.names or ())}
