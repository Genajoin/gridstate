"""V-refine: двухпроходное ужесточение согласованных real V-измерений.

После первого solve остатки ``z − h`` (``h`` = ``estimated_si``) на
real V-измерениях показывают, где оценка систематически провисает ниже
СОБСТВЕННЫХ замеров напряжения. Корень — недовзвешенная V-мера: загрузчик
ставит плоскую σ_V (несколько кВ), отчего на нормальных уравнениях потоки
перевешивают V и тянут регион вниз монотонно с классом (на 500 кВ
p50(z−h) до +3.4 кВ, >90 % остатков положительны).

Механизм: ужесточить (variance × factor², factor<1) ТОЛЬКО те V-меры,
что согласованы с решением первого прохода (``|z−h|/σ < rn_threshold``),
и пере-решить (warm, ``init="results"``). Конфликтные V-меры НЕ трогаем:
большой остаток на V-мере — кандидат в грубую ошибку (битый замер кромки
вроде Курской АЭС z=760 кВ при истинных 693), её ужесточение лишь
закрепило бы испорченное решение.

Конфигурация (rn_threshold=3, factor=0.7) валидирована универсально на
4 ОДУ-моделях: Восток p50 −37 % / p95 −23 %, СЗ p95 −15 %, глобальный
bias mean тает на всех; class-max нигде не регрессирует > 5 %.

В отличие от bad-data re-pass (``gridstate.bad_data_repass``) этот шаг НЕ
меняет значения и статусы мер — только дисперсии согласованных V, поэтому
не может внести грубую ошибку, лишь перераспределяет доверие.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate.z_vector import KIND_VOLTAGE


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VRefinePlan:
    """План ужесточения V-мер по итогам классификации residuals."""

    tighten_ids: frozenset[int]  # variance := variance · factor²
    n_consistent: int  # согласованных V-мер (rn < threshold) — они и ужесточаются
    n_conflicting: int  # конфликтных V-мер (rn ≥ threshold) — оставлены как есть

    @property
    def empty(self) -> bool:
        return not self.tighten_ids


def classify_v_refine(measurements: np.ndarray, *, rn_threshold: float) -> VRefinePlan:
    """Отобрать согласованные real V-меры для ужесточения.

    Args:
        measurements: structured-массив measurements рабочей модели ПОСЛЕ
            solve (``estimated_si`` заполнены write_measurement_estimates).
        rn_threshold: порог согласованности ``|z−h|/σ < rn_threshold``.

    Returns:
        :class:`VRefinePlan` (может быть ``empty`` — тогда re-solve не нужен).
    """
    sel: np.ndarray = (
        measurements["status"].astype(bool)
        & ~measurements["is_pseudo"].astype(bool)
        & (measurements["measurement_type"] == KIND_VOLTAGE)
        & np.isfinite(measurements["estimated_si"])
    )
    z = measurements["value"][sel]
    h = measurements["estimated_si"][sel]
    sig = np.sqrt(measurements["variance"][sel])
    ids = measurements["id"][sel]

    rn = np.abs(z - h) / np.maximum(sig, 1e-9)
    consistent = rn < rn_threshold
    tighten_ids = {int(i) for i in ids[consistent]}

    return VRefinePlan(
        tighten_ids=frozenset(tighten_ids),
        n_consistent=int(consistent.sum()),
        n_conflicting=int((~consistent).sum()),
    )


def apply_v_refine_plan(model: Working, plan: VRefinePlan, *, factor: float) -> dict:
    """Применить план к ``model.measurements`` (рабочая копия пайплайна).

    tighten → ``variance := variance · factor²`` (σ × factor). Веса солвер
    строит из variance (см. ``z_vector.build_z_and_r``).
    """
    if not plan.tighten_ids:
        return {"tightened": 0, "conflicting": plan.n_conflicting}
    m = model.measurements.to_numpy()
    sel = np.isin(m["id"], list(plan.tighten_ids))
    m["variance"][sel] *= float(factor) ** 2
    model.measurements.update_from_array(m)
    return {"tightened": len(plan.tighten_ids), "conflicting": plan.n_conflicting}
