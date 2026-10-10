"""χ² test of state estimation quality.

Adapted from pandapower:
    pandapower/estimation/state_estimation.py (function ``perform_chi2_test``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Idea::

    J  = rᵀ R⁻¹ r                           (SE quality index)
    df = m − n                               (degrees of freedom)
    threshold = χ²(df, 1 − α)                (α = chi2_prob_false)

If ``J > threshold`` → with probability ≥ 1 − α the data contain gross errors
(bad data or a topological mismatch). The test assumes that
``estimate()`` has already been executed — the function works with the state written
into ``model``.
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
    """Result of the χ² test.

    Attributes:
        bad_data_present: ``True`` if ``J > threshold``.
        objective: ``J = rᵀ R⁻¹ r``.
        threshold: critical value ``χ²(df, 1 − α)``.
        degrees_of_freedom: ``m − n``.
        n_measurements: number of active measurements ``m``.
        n_state_vars: ``2·n_bus − 1``.
        alpha: false-alarm probability used.
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
    """Run the χ² test on the current state of ``model``.

    Args:
        model: a model for which ``estimate()`` has already been executed (the nodes contain
            the estimated ``voltage_magnitude`` / ``voltage_angle``).
        measurements: collection — if ``None``, ``model.measurements`` is used.
        chi2_prob_false: α — false-alarm probability (default
            ``0.05``).

    Returns:
        ``Chi2Result``. If ``df ≤ 0`` (the system is only weakly overdetermined or,
        conversely, exactly determined), the test is considered inconclusive — ``bad_data_present``
        will be ``False``, but a warning will appear in the log.

    Raises:
        ValueError: ``chi2_prob_false`` outside (0, 1).
    """
    if not (0.0 < chi2_prob_false < 1.0):
        raise ValueError(f"chi2_prob_false must be in (0, 1); got {chi2_prob_false}")

    if measurements is None:
        measurements = model.measurements

    diag = compute_diagnostics(model, measurements)
    m = int(diag.r.shape[0])
    n = int(diag.layout.size)
    df = m - n

    objective = float(diag.r @ (diag.R_inv @ diag.r))

    if df <= 0:
        logger.warning(
            "χ² test is meaningless: df = m−n = %d − %d = %d ≤ 0 (the system is not redundant)",
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
        "χ² test: J=%.3f, threshold=%.3f, df=%d, bad_data_present=%s",
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
