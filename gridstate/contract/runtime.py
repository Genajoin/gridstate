"""Рантайм-контейнеры контракта + публичный фасад ``run(SEInput) → SEOutput``.

Публичная граница SE: :class:`SEInput` несёт рабочий слой + предвычисленные
числовые планы (:class:`~gridstate.contract.derived.DerivedInputs`), :func:`run`
делегирует :func:`gridstate.pipeline.run`, а :class:`SEOutput` оборачивает его
``SEResult``. Численный результат идентичен прямому вызову пайплайна.

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

    Делегирует :func:`gridstate.pipeline.run` на рабочей модели и оборачивает
    ``SEResult`` в :class:`SEOutput`. Результат идентичен прямому вызову
    пайплайна — граница входа/выхода зафиксирована контрактом.

    Args:
        se_input: входной контракт (:class:`SEInput`).
        config: :class:`gridstate.pipeline.PipelineConfig`; ``None`` → production-дефолты.
        on_event: callback прогресса (как у пайплайна).
        init_state: прошлый :class:`SEOutput` (или ``SEResult``) для тёплого старта.
        validate: валидировать вход против контракта и **бросать**
            :class:`gridstate.contract.ContractValidationError` при недостающих
            обязательных колонках или несовместимой версии контракта (принцип
            «падать рано и явно»). ``False`` — пропустить проверку. Полноценные
            модели несут полный DTYPE → проверка проходит; падение значит реальный
            дефект входа.

    Returns:
        :class:`SEOutput`.
    """
    from gridstate.pipeline import run as _pipeline_run

    if validate:
        se_input.validate(strict=True)

    # Тёплый старт: пайплайн ждёт объект с ``.outputs.nodes`` / ``.model.nodes`` —
    # это либо прошлый SEResult, либо несём его из SEOutput.result.
    prev = init_state.result if isinstance(init_state, SEOutput) else init_state

    # Research-оркестрация shunt-sanity: trial-прогоны требуют ПРИСТИННУЮ копию
    # входа ДО базового прогона (пайплайн мутирует model in place).
    sanity_on = bool(getattr(config, "shunt_sanity", False)) if config is not None else False
    pristine = se_input.model.copy() if sanity_on else None

    # Числовые планы (``derived``) предвычислены вне ядра; шаги применяют их
    # контрактными ядрами. ``derived=None`` → XML/формат-зависимые шаги пропускаются.
    result = _pipeline_run(
        se_input.model,
        config=config,
        derived=se_input.derived,
        on_event=on_event,
        init_state=prev,
    )
    if sanity_on:
        result = _shunt_sanity_rerun(
            base_result=result,
            pristine=pristine,
            config=config,
            derived=se_input.derived,
            on_event=on_event,
        )
    return SEOutput.from_result(result)


def _shunt_sanity_rerun(
    *,
    base_result: SEResult,
    pristine: Any,
    config: Any,
    derived: DerivedInputs | None,
    on_event: Callable[[dict], None] | None,
) -> SEResult:
    """Try-off/flip шунтов-кандидатов ПОЛНЫМИ re-run'ами пайплайна (research).

    Валидированный механизм (4 ОДУ 2026-07-06): кандидаты — активные шунты на
    узлах, где node-V-мера расходится с БАЗОВЫМ решением; каждый вариант
    (off/flip) правится на копии пристинного входа и прогоняется ПОЛНЫМ
    пайплайном; гейт — падение Σrn² (согласие с собственными real-мерами)
    больше ``shunt_sanity_gate_drop``. Ложные кандидаты гейтом отвергаются.

    Сравнение обязано быть «полный прогон против полного прогона»: правка сети
    меняет решение, и σ-ужесточения v_refine (и планы v_mirror/bad_data)
    должны пересчитаться вокруг НОВОГО решения — одиночный warm re-solve
    внутри пайплайна штрафует честную правку и ложно её отвергает.
    """
    from dataclasses import replace as _dc_replace

    from gridstate.pipeline import run as _pipeline_run
    from gridstate.shunt_sanity import classify_shunt_candidates, edit_shunt, sum_rn2

    if not base_result.success:
        return base_result
    model = base_result.model
    plan = classify_shunt_candidates(
        model.measurements.to_numpy(),
        model.nodes.to_numpy(),
        model.shunts.to_numpy(),
        v_frac=float(config.shunt_sanity_v_frac),
        max_candidates=int(config.shunt_sanity_max_candidates),
    )

    def _emit(stats: dict) -> None:
        if on_event is not None:
            on_event({"type": "step_done", "name": "shunt_sanity", "stats": stats})

    if plan.empty:
        _emit({"candidates": 0, "accepted": 0, "skipped": "no-op (нет кандидатов)"})
        return base_result

    trial_cfg = _dc_replace(config, shunt_sanity=False)  # без рекурсии
    base_rn2 = sum_rn2(model.measurements.to_numpy())
    gate = float(config.shunt_sanity_gate_drop)

    def _trial(edits: dict[int, str]) -> tuple[SEResult | None, float]:
        m = pristine.copy()
        for nid, mode in edits.items():
            if edit_shunt(m, nid, mode) == 0:
                return None, float("inf")
        res = _pipeline_run(m, config=trial_cfg, derived=derived)
        if not res.success:
            return None, float("inf")
        # ВАЖНО: пайплайн работает на внутренней копии — считать по res.model
        # (во входном m остаются value=0/est=0 → Σrn² ложно нулевой).
        return res, sum_rn2(res.model.measurements.to_numpy())

    accepted: dict[int, str] = {}
    for nid in plan.candidates:
        best_mode: str | None = None
        best_rn2 = base_rn2 - gate
        for mode in ("off", "flip"):
            _res, rn2 = _trial({nid: mode})
            if rn2 < best_rn2:
                best_mode, best_rn2 = mode, rn2
        if best_mode is not None:
            accepted[nid] = best_mode

    if not accepted:
        _emit(
            {
                "candidates": len(plan.candidates),
                "accepted": 0,
                "skipped": "no-op (гейт отверг всех кандидатов)",
            }
        )
        return base_result

    final_res, final_rn2 = _trial(accepted)
    if final_res is None or final_rn2 >= base_rn2 - gate:
        _emit(
            {
                "candidates": len(plan.candidates),
                "accepted": 0,
                "skipped": "no-op (совместное применение не прошло гейт)",
            }
        )
        return base_result
    _emit(
        {
            "candidates": len(plan.candidates),
            "accepted": len(accepted),
            "edits": {int(k): v for k, v in accepted.items()},
            "rn2_drop": float(base_rn2 - final_rn2),
        }
    )
    return final_res


def prepare_network(
    se_input: SEInput,
    *,
    config: Any = None,
    on_event: Callable[[dict], None] | None = None,
    validate: bool = True,
) -> Any:
    """Контрактная обёртка :func:`gridstate.pipeline.prepare_network`.

    Выполняет ТОЛЬКО сетевые деривации пайплайна (топология/РПН/реакторы/
    нормализация/каскады статусов) над входным контрактом и возвращает
    ``Working`` — сеть в том состоянии, в котором её решает SE. Сам
    ``se_input`` не мутируется. Применение результата к модели-носителю —
    забота внешнего адаптера.
    """
    from gridstate.pipeline import prepare_network as _pipeline_prepare

    if validate:
        se_input.validate(strict=True)
    return _pipeline_prepare(
        se_input.model,
        config=config,
        derived=se_input.derived,
        on_event=on_event,
    )
