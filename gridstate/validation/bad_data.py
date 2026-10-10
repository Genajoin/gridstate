"""Detection and removal of bad data by the largest normalized
residual test (``rn_max``).

Adapted from pandapower:
    pandapower/estimation/state_estimation.py (function ``perform_rn_max_test``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Algorithm::

    Ω = R − H G⁻¹ Hᵀ                       (residual covariance)
    r^N_i = |r_i| / √(diag(Ω)_i)            (normalized residual)

    while max(finite r^N) > threshold:
        mark the maximum r^N as BAD (status=False, quality=BAD)
        re-run estimate()
        recompute r^N

Critical (non-redundant) measurements have ``diag(Ω)_i ≈ 0`` and
``r^N_i = +inf``: their errors are undetectable and removing one makes the
network unobservable, so they are never removed — only reported in
``BadDataResult.critical_meas_ids``.

Returns a ``BadDataResult`` with the list of removed ``meas_id`` and the final
``SEResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gridstate.algebra.normalized_residuals import normalized_residuals
from gridstate.api import estimate
from gridstate.constants import MeasurementQuality
from gridstate.result import SEResult
from gridstate.validation._diagnostics import compute_diagnostics


if TYPE_CHECKING:
    from gridstate.validation._diagnostics import Diagnostics
    from gridstate.working import Working, _ArrayCollection


logger = logging.getLogger(__name__)


# Measurement quality code "BAD" in the contract (value of the ``quality`` column).
QUALITY_BAD = int(MeasurementQuality.BAD)


class ResidualTopRecord(NamedTuple):
    """One entry of ``NormalizedResidualReport.top_records`` (named fields).

    A ``NamedTuple`` (tuple subclass): positional access is preserved for
    backward compatibility while giving the fields readable names.

    Attributes:
        rn: normalized residual ``r^N``.
        meas_id: ``meas_id`` of the measurement.
        object_kind: ``0`` — NODE, ``1`` — BRANCH.
        object_id: ID of the measured object (``-1`` if not found).
        kind: ``MeasurementType`` (0=POWER_P, 1=POWER_Q, 2=VOLTAGE, ...).
        value: measurement value (``nan`` if not found).
        residual: residual ``r``.
        sigma: ``√σ²``.
    """

    rn: float
    meas_id: int
    object_kind: int
    object_id: int
    kind: int
    value: float
    residual: float
    sigma: float


@dataclass
class NormalizedResidualReport:
    """Audit-only report on the normalized residuals after SE.

    Does not modify measurements (unlike ``remove_bad_data``). It is simply
    a diagnostic snapshot: which measurements are suspicious (r^N > threshold),
    top-N by normalized residual.

    Attributes:
        rn: array of normalized residuals (m,). ``inf`` for non-redundant
            (``diag(Ω)`` ≤ 0).
        meas_ids: ``meas_id`` for each element of ``rn`` (the order matches
            the z-vector in the current model).
        top_records: top-N records by ``|rn|`` (excluding ``inf``) —
            list of :class:`ResidualTopRecord` (named fields).
        n_above_threshold: how many measurements (excluding ``inf``) give
            ``rn > rn_threshold``.
        rn_max_finite: ``max(rn[rn<inf])`` — the maximum over the informative ones.
    """

    rn: np.ndarray
    meas_ids: np.ndarray
    top_records: list[ResidualTopRecord]
    n_above_threshold: int
    rn_max_finite: float


def compute_normalized_residuals_report(
    model: Working,
    measurements: _ArrayCollection | None = None,
    *,
    top_n: int = 5,
    rn_threshold: float = 3.0,
) -> NormalizedResidualReport:
    """Build a report on the normalized residuals after SE.

    Works on the current state of ``model`` (``estimate()`` was run earlier).
    Computes ``r^N = |r| / √(diag(Ω))``, where ``Ω = R − H G⁻¹ Hᵀ``. Returns the
    top-``top_n`` suspicious ones and the count of `rn > rn_threshold`. Does **not**
    modify measurements — audit-only.

    Args:
        model: ``Working`` with a completed estimation.
        measurements: ``_ArrayCollection`` — ``None`` → ``model.measurements``.
        top_n: how many top records to return (descending ``|rn|``).
        rn_threshold: threshold for counting suspicious ones (classically 3.0).

    Returns:
        ``NormalizedResidualReport``. ``inf`` values are excluded from the top.
    """
    if measurements is None:
        measurements = model.measurements
    assert measurements is not None

    diag = compute_diagnostics(model, measurements)
    if diag.r.shape[0] == 0:
        return NormalizedResidualReport(
            rn=np.array([], dtype=np.float64),
            meas_ids=np.array([], dtype=np.int64),
            top_records=[],
            n_above_threshold=0,
            rn_max_finite=0.0,
        )
    rn = _normalized_residuals(diag)
    meas_ids = np.asarray(diag.meas_index.meas_id, dtype=np.int64)
    finite_mask = np.isfinite(rn)
    n_above = int(np.sum((rn > rn_threshold) & finite_mask))
    rn_max_finite = float(np.max(rn[finite_mask])) if np.any(finite_mask) else 0.0

    object_kind = np.asarray(diag.meas_index.object_kind, dtype=np.int64)
    kind = np.asarray(diag.meas_index.kind, dtype=np.int64)

    order = np.argsort(-np.where(finite_mask, rn, -np.inf))
    top_records: list[ResidualTopRecord] = []
    for pos in order[:top_n]:
        if not finite_mask[pos]:
            break
        bad_meas = measurements.get_by_id(int(meas_ids[pos]))
        value = float(bad_meas.value) if bad_meas is not None else float("nan")
        obj_id = int(bad_meas.object_id) if bad_meas is not None else -1
        sigma = float(np.sqrt(diag.sigma2[pos]))
        top_records.append(
            ResidualTopRecord(
                rn=float(rn[pos]),
                meas_id=int(meas_ids[pos]),
                object_kind=int(object_kind[pos]),
                object_id=obj_id,
                kind=int(kind[pos]),
                value=value,
                residual=float(diag.r[pos]),
                sigma=sigma,
            )
        )
    return NormalizedResidualReport(
        rn=rn,
        meas_ids=meas_ids,
        top_records=top_records,
        n_above_threshold=n_above,
        rn_max_finite=rn_max_finite,
    )


@dataclass
class BadDataResult:
    """Result of the iterative bad data clean-up.

    Attributes:
        se_result: the final ``SEResult`` after removing all bad
            measurements.
        removed_meas_ids: list of ``meas_id`` of measurements marked as BAD.
        rn_max_history: values of ``max(r^N)`` over the finite ``r^N`` after
            each iteration (including the last one, which passed the test); ``0.0`` when
            no measurement has a finite ``r^N``.
        converged: ``True`` if on the final iteration ``max(r^N) ≤ threshold``.
        critical_meas_ids: ``meas_id`` of the critical (non-redundant)
            measurements of the last checked iteration (``r^N = inf``). They
            are never removed: their errors cannot be detected by the test.
    """

    se_result: SEResult
    removed_meas_ids: list[int] = field(default_factory=list)
    rn_max_history: list[float] = field(default_factory=list)
    converged: bool = False
    critical_meas_ids: list[int] = field(default_factory=list)


def _normalized_residuals(diag: Diagnostics) -> np.ndarray:
    """``r^N = |r| / √(diag(Ω))``, where ``Ω = R − H G⁻¹ Hᵀ``.

    If ``Ω_ii`` is numerically zero or negative (poor conditioning of
    G), the element is returned as ``+∞`` — such a measurement is effectively
    "non-redundant", and the normalized residual is not defined for it.

    Delegates to the shared sparse implementation
    :func:`gridstate.algebra.normalized_residuals.normalized_residuals`
    (also used by the quality summary): no ``m × m`` matrix is formed.
    """
    return normalized_residuals(diag.r, diag.H, diag.sigma2)


def remove_bad_data(
    model: Working,
    measurements: _ArrayCollection | None = None,
    rn_max_threshold: float = 3.0,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    estimate_max_iterations: int = 50,
) -> BadDataResult:
    """Iteratively remove bad measurements and return the final ``SEResult``.

    On each iteration:
        1. ``estimate()`` is run;
        2. ``r^N`` is computed;
        3. If ``max(r^N) > threshold`` — the bad measurement is marked
           ``status=False, quality=BAD`` and the loop repeats.

    Only finite ``r^N`` take part in the test: critical measurements
    (``r^N = inf``) are kept and listed in ``critical_meas_ids``.

    Args:
        model: ``Working`` — updated in place.
        measurements: collection — if ``None``, ``model.measurements``.
        rn_max_threshold: threshold for the normalized residual (classically 3.0).
        max_iterations: maximum number of rejection iterations (protection against
            looping).
        tolerance: convergence criterion for the inner ``estimate``.
        estimate_max_iterations: Gauss-Newton iteration limit for each
            ``estimate`` call.
    """
    if measurements is None:
        measurements = model.measurements
    assert measurements is not None  # for pyright

    removed: list[int] = []
    rn_history: list[float] = []
    critical: list[int] = []

    def _run_estimate() -> SEResult:
        return estimate(
            model,
            measurements=measurements,
            algorithm="wls",
            init="flat",
            tolerance=tolerance,
            max_iterations=estimate_max_iterations,
        )

    def _result(*, converged: bool) -> BadDataResult:
        # Snapshot the current se_result / removed / rn_history (late-bound).
        return BadDataResult(
            se_result=se_result,
            removed_meas_ids=removed,
            rn_max_history=rn_history,
            converged=converged,
            critical_meas_ids=critical,
        )

    # First run — before the loop; the loop only repeats estimate after a rejection.
    se_result = _run_estimate()

    for it in range(max_iterations + 1):
        if not se_result.success:
            logger.warning(
                "rn_max: SE did not converge at iteration %d — aborting the clean-up", it
            )
            return _result(converged=False)

        diag = compute_diagnostics(model, measurements)
        if diag.r.shape[0] == 0:
            logger.warning("rn_max: no active measurements — nothing to check")
            return _result(converged=True)

        rn = _normalized_residuals(diag)
        # Critical measurements (r^N = inf) are never candidates for removal.
        finite = np.isfinite(rn)
        critical = [int(i) for i in diag.meas_index.meas_id[~finite]]
        rn_max = float(np.max(rn[finite])) if finite.any() else 0.0
        rn_history.append(rn_max)

        if rn_max <= rn_max_threshold:
            logger.debug(
                "rn_max test passed at iteration %d: max(r^N)=%.3f ≤ %.3f",
                it,
                rn_max,
                rn_max_threshold,
            )
            if critical:
                logger.info(
                    "rn_max: %d critical measurement(s) kept (r^N undefined): ids=%s",
                    len(critical),
                    critical,
                )
            return _result(converged=True)

        # Otherwise — drop the worst measurement.
        bad_pos = int(np.argmax(np.where(finite, rn, -np.inf)))
        bad_id = int(diag.meas_index.meas_id[bad_pos])
        bad_meas = measurements.get_by_id(bad_id)
        if bad_meas is None:
            logger.error(
                "rn_max: measurement id=%d not found in the collection — aborting",
                bad_id,
            )
            return _result(converged=False)
        bad_meas.status = False
        bad_meas.quality = QUALITY_BAD
        removed.append(bad_id)
        logger.info(
            "rn_max: itr %d, removed measurement id=%d (r^N=%.2f > %.2f)",
            it,
            bad_id,
            rn_max,
            rn_max_threshold,
        )

        # Iteration limit reached — do not re-estimate after the last rejection
        # (the final se_result corresponds to the completed run, as before).
        if it == max_iterations:
            break
        se_result = _run_estimate()

    logger.warning(
        "rn_max: iteration limit reached (%d) and the test has not passed yet",
        max_iterations,
    )
    return _result(converged=False)
