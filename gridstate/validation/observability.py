"""Анализ наблюдаемости сети по имеющемуся набору измерений.

Подход — численный, через ранг матрицы Якоби ``H`` на flat-старте:

1. ``H = ∂h/∂E`` строится для всех активных измерений при ``V=1`` p.u., ``δ=0``.
2. Сеть наблюдаема iff ``rank(H) == 2·n_bus − 1``.
3. Если ранг недостаточен — для диагностики считаем нормы столбцов ``H``
   и помечаем узлы, переменные состояния которых ``δ_i / V_i`` не покрыты
   ни одним измерением (нулевой столбец).

Это не полная схема анализа из Abur & Expósito (§4–5), но покрывает
большинство практических случаев недонаблюдаемости (отсутствие измерений
на отдельных узлах/островах).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gridstate.algebra.base import BaseAlgebra
from gridstate.state import StateLayout, flat_start, unpack
from gridstate.units import model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r


if TYPE_CHECKING:
    from power_system import MeasurementCollection, PowerSystemModel


@dataclass
class ObservabilityReport:
    """Итог анализа наблюдаемости.

    Attributes:
        is_observable: ``True`` если ``rank(H) == 2·n_bus − 1``.
        n_state_vars: размер вектора состояния = ``2·n_bus − 1``.
        n_measurements: число активных измерений (после фильтрации
            ``status``/``quality``).
        rank_H: фактический ранг H на flat-старте.
        unobservable_buses: ID узлов, чьи δ или V не покрыты ни одним
            измерением (нулевой столбец H).
        unobservable_branches: ID ветвей, для которых не удалось собрать
            ни одного измерения (зарезервировано — пока всегда пустой).
        diagnostics: человеко-читаемое сообщение для логов.
    """

    is_observable: bool
    n_state_vars: int
    n_measurements: int
    rank_H: int
    unobservable_buses: list[int] = field(default_factory=list)
    unobservable_branches: list[int] = field(default_factory=list)
    diagnostics: str = ""


def analyze_observability(
    model: PowerSystemModel,
    measurements: MeasurementCollection | None = None,
) -> ObservabilityReport:
    """Проверить наблюдаемость и вернуть отчёт.

    Args:
        model: ``PowerSystemModel``.
        measurements: коллекция измерений. ``None`` → ``model.measurements``.

    Returns:
        ``ObservabilityReport``. Если ``n_measurements < n_state_vars`` —
        обязательное условие нарушено, но всё равно проводим ранг-анализ
        (rank будет ≤ n_meas).
    """
    if measurements is None:
        measurements = model.measurements

    network_pu = model_to_pu(model)
    ybus, yf, yt = build_ybus(network_pu)
    z, _, meas_index = build_z_and_r(model, measurements, network_pu)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)

    n_state = layout.size
    n_meas = int(z.shape[0])

    if n_meas == 0:
        return ObservabilityReport(
            is_observable=False,
            n_state_vars=n_state,
            n_measurements=0,
            rank_H=0,
            unobservable_buses=network_pu.bus_ids.tolist(),
            diagnostics="Нет ни одного активного измерения.",
        )

    algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
    delta, v = unpack(flat_start(layout), layout)
    H = algebra.evaluate_jacobian(v, delta)
    H_dense = H.toarray()

    rank_h = int(np.linalg.matrix_rank(H_dense))
    is_obs = rank_h == n_state

    # Диагностика: какие столбцы H пустые?
    col_norms = np.linalg.norm(H_dense, axis=0)
    zero_cols = np.where(col_norms < 1e-10)[0]

    unobs_buses: set[int] = set()
    n_bus = network_pu.n_bus
    for j in zero_cols:
        # j < n_bus-1 → столбец δ_non_slack[j]; иначе → столбец V[j − (n_bus − 1)]
        bus_pos = int(layout.non_slack_idx[j]) if j < n_bus - 1 else int(j - (n_bus - 1))
        unobs_buses.add(int(network_pu.bus_ids[bus_pos]))

    if is_obs:
        diagnostics = (
            f"Сеть наблюдаема: rank(H)={rank_h} = n_state={n_state}, m={n_meas} измерений."
        )
    else:
        diagnostics = (
            f"Сеть НЕ наблюдаема: rank(H)={rank_h} < n_state={n_state} "
            f"(m={n_meas}). Дефицит ранга = {n_state - rank_h}."
        )
        if unobs_buses:
            diagnostics += f" Узлы без покрытия в H: {sorted(unobs_buses)}."

    return ObservabilityReport(
        is_observable=is_obs,
        n_state_vars=n_state,
        n_measurements=n_meas,
        rank_H=rank_h,
        unobservable_buses=sorted(unobs_buses),
        diagnostics=diagnostics,
    )
