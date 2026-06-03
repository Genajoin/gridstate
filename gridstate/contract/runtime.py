"""Рантайм-контейнеры контракта + публичный фасад ``run(SEInput) → SEOutput``.

Публичная граница SE: :class:`SEInput` несёт рабочий слой + предвычисленные
числовые планы (:class:`~gridstate.contract.derived.DerivedInputs`), :func:`run`
делегирует :func:`gridstate.pipeline.run`, а :class:`SEOutput` оборачивает его
``SEResult``. Численный результат идентичен прямому вызову пайплайна (бит-в-бит).

Импорт тяжёлых модулей (``gridstate.pipeline``) — ленивый, внутри :func:`run`: сам
модуль ``runtime`` (и схема ``gridstate.contract``) при импорте пайплайн не тянет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.contract.tables import SE_OUTPUT
from gridstate.contract.validate import ValidationReport, validate_input
from gridstate.contract.version import CONTRACT_VERSION


if TYPE_CHECKING:
    from collections.abc import Callable

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
    ``DerivedInputs`` без формат-слоя источника. Этот хелпер оборачивает уже готовую
    модель; XML/формат-зависимые шаги ``run`` пропускаются (``derived=None``).
    """
    return SEInput(model=model, derived=None, contract_version=contract_version)


# ---------------------------------------------------------------------------
# Выход
# ---------------------------------------------------------------------------


def _empty_output(table_name: str) -> np.ndarray:
    """Пустой структурированный массив выходного слоя для таблицы ``table_name``."""
    schema = {t.name: t for t in SE_OUTPUT.tables()}[table_name]
    return np.empty(0, dtype=schema.output_dtype())


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
        result: исходный ``SEResult`` — back-compat escape hatch (Фаза 1; Фаза 2+
            может убрать). Несёт ``model`` (рабочая копия), ``chi2``,
            ``worst_residuals`` и т.п.
    """

    nodes: np.ndarray = field(default_factory=lambda: _empty_output("nodes"))
    branches: np.ndarray = field(default_factory=lambda: _empty_output("branches"))
    measurements: np.ndarray = field(default_factory=lambda: _empty_output("measurements"))
    success: bool = False
    iterations: int = 0
    objective_value: float = float("nan")
    algorithm: str = ""
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


# ---------------------------------------------------------------------------
# Фасад
# ---------------------------------------------------------------------------


def run(
    se_input: SEInput,
    *,
    config: Any = None,
    on_event: Callable[[dict], None] | None = None,
    init_state: SEOutput | Any = None,
    validate: bool = True,
) -> SEOutput:
    """Публичный контрактный вход SE: ``SEInput → SEOutput``.

    **Фаза 1:** делегирует :func:`gridstate.pipeline.run` на обёрнутой PSC-модели и
    оборачивает ``SEResult`` в :class:`SEOutput`. Результат бит-в-бит совпадает с
    прямым вызовом пайплайна — граница зафиксирована, внутренности не изменены.

    Args:
        se_input: входной контракт (:class:`SEInput`).
        config: :class:`gridstate.pipeline.PipelineConfig`; ``None`` → production-дефолты.
        on_event: callback прогресса (как у пайплайна).
        init_state: прошлый :class:`SEOutput` (или ``SEResult``) для тёплого старта.
        validate: валидировать вход против контракта и **бросать**
            :class:`gridstate.contract.ContractValidationError` при недостающих
            обязательных колонках или несовместимой версии контракта (граница
            «падать рано и явно», план §4 — это единственные проверяемые сейчас
            условия). ``False`` — пропустить проверку. Реальные модели несут полный
            DTYPE → проверка проходит; падение значит реальный дефект входа.

    Returns:
        :class:`SEOutput`.
    """
    from gridstate.pipeline import run as _pipeline_run

    if validate:
        se_input.validate(strict=True)

    # Тёплый старт: пайплайн ждёт объект с ``.outputs.nodes`` / ``.model.nodes`` —
    # это либо прошлый SEResult, либо несём его из SEOutput.result.
    prev = init_state.result if isinstance(init_state, SEOutput) else init_state

    # Числовые планы (``derived``) предвычислены вне ядра; шаги применяют их
    # контрактными ядрами. ``derived=None`` → XML/формат-зависимые шаги пропускаются.
    result = _pipeline_run(
        se_input.model,
        config=config,
        derived=se_input.derived,
        on_event=on_event,
        init_state=prev,
    )
    return SEOutput.from_result(result)
