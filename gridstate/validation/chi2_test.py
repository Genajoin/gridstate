"""χ²-тест качества оценки состояния.

Adapted from pandapower:
    pandapower/estimation/state_estimation.py (function ``perform_chi2_test``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Идея::

    J  = rᵀ R⁻¹ r                           (индекс качества SE)
    df = m − n                               (степени свободы)
    threshold = χ²(df, 1 − α)                (α = chi2_prob_false)

Если ``J > threshold`` → с вероятностью ≥ 1 − α в данных есть грубые ошибки
(bad data или топологическое расхождение). Тест предполагает, что
``estimate()`` уже выполнен — функция работает с состоянием, записанным
в ``model``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.stats import chi2

from gridstate.validation._diagnostics import compute_diagnostics


if TYPE_CHECKING:
    from gridstate.working import Working, _ArrayCollection


logger = logging.getLogger(__name__)


@dataclass
class Chi2Result:
    """Результат χ²-теста.

    Attributes:
        bad_data_present: ``True``, если ``J > threshold``.
        objective: ``J = rᵀ R⁻¹ r``.
        threshold: критическое значение ``χ²(df, 1 − α)``.
        degrees_of_freedom: ``m − n``.
        n_measurements: число активных измерений ``m``.
        n_state_vars: ``2·n_bus − 1``.
        alpha: использованная вероятность ложного срабатывания.
    """

    bad_data_present: bool
    objective: float
    threshold: float
    degrees_of_freedom: int
    n_measurements: int
    n_state_vars: int
    alpha: float


def chi2_analysis(
    model: Working,
    measurements: _ArrayCollection | None = None,
    chi2_prob_false: float = 0.05,
) -> Chi2Result:
    """Запустить χ²-тест на текущем состоянии ``model``.

    Args:
        model: модель, для которой уже выполнен ``estimate()`` (узлы содержат
            оценённые ``voltage_magnitude`` / ``voltage_angle``).
        measurements: коллекция — если ``None``, берётся ``model.measurements``.
        chi2_prob_false: α — вероятность ложного срабатывания (по умолчанию
            ``0.05``).

    Returns:
        ``Chi2Result``. Если ``df ≤ 0`` (система переопределена слабо или
        наоборот ровно), тест считается несостоятельным — ``bad_data_present``
        будет ``False``, но в логе появится предупреждение.

    Raises:
        ValueError: ``chi2_prob_false`` вне (0, 1).
    """
    if not (0.0 < chi2_prob_false < 1.0):
        raise ValueError(f"chi2_prob_false должна быть в (0, 1); получено {chi2_prob_false}")

    if measurements is None:
        measurements = model.measurements

    diag = compute_diagnostics(model, measurements)
    m = int(diag.r.shape[0])
    n = int(diag.layout.size)
    df = m - n

    objective = float(diag.r @ (diag.R_inv @ diag.r))

    if df <= 0:
        logger.warning(
            "χ²-тест не имеет смысла: df = m−n = %d − %d = %d ≤ 0 (система не избыточна)",
            m,
            n,
            df,
        )
        return Chi2Result(
            bad_data_present=False,
            objective=objective,
            threshold=float("nan"),
            degrees_of_freedom=df,
            n_measurements=m,
            n_state_vars=n,
            alpha=chi2_prob_false,
        )

    threshold = float(chi2.ppf(1.0 - chi2_prob_false, df))
    bad = bool(objective > threshold)
    logger.debug(
        "χ²-тест: J=%.3f, threshold=%.3f, df=%d, bad_data_present=%s",
        objective,
        threshold,
        df,
        bad,
    )
    return Chi2Result(
        bad_data_present=bad,
        objective=objective,
        threshold=threshold,
        degrees_of_freedom=df,
        n_measurements=m,
        n_state_vars=n,
        alpha=chi2_prob_false,
    )
