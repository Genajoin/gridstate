"""V-refine: two-pass tightening of consistent real V measurements.

After the first solve the residuals ``z − h`` (``h`` = ``estimated_si``) of real V
measurements show where the estimate systematically sags below its OWN voltage
readings. The root is an underweighted V measurement: the loader sets a flat σ_V
(several kV), so in the normal equations the flows outweigh V and pull the network
down monotonically with the voltage class (at 500 kV p50(z−h) reaches +3.4 kV, more
than 90 % of the residuals are positive).

Mechanism: tighten (variance × factor², factor<1) ONLY the V measurements consistent
with the first-pass solution (``|z−h|/σ < rn_threshold``) and re-solve (warm,
``init="results"``). Conflicting V measurements are NOT touched: a large residual on
a V measurement marks a gross-error candidate (a bad boundary reading such as
z=760 kV against a true 693 kV), and tightening it would only lock in a corrupted
solution.

The configuration (rn_threshold=3, factor=0.7) is validated with one set of options
on four large production models: best cases p50 −37 % / p95 −23 % and p95 −15 %; the
global mean bias shrinks on all of them and the per-class maximum nowhere regresses
by more than 5 %.

В отличие от bad-data re-pass (``gridstate.bad_data_repass``) этот шаг НЕ
меняет значения и статусы мер — только дисперсии согласованных V, поэтому
не может внести грубую ошибку, лишь перераспределяет доверие.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import meas_mask
from gridstate.working import table_copy, table_replace
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
    # Any object type: V readings are not restricted to NODE objects here.
    sel = meas_mask(measurements, kinds=KIND_VOLTAGE, pseudo=False, finite_estimate=True)
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
    m = table_copy(model.measurements)
    sel = np.isin(m["id"], list(plan.tighten_ids))
    m["variance"][sel] *= float(factor) ** 2
    table_replace(model.measurements, m)
    return {"tightened": len(plan.tighten_ids), "conflicting": plan.n_conflicting}
