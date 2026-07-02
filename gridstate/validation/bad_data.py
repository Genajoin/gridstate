"""Обнаружение и удаление плохих данных по тесту максимального нормированного
остатка (``rn_max``).

Adapted from pandapower:
    pandapower/estimation/state_estimation.py (function ``perform_rn_max_test``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Алгоритм::

    Ω = R − H G⁻¹ Hᵀ                       (ковариация невязок)
    r^N_i = |r_i| / √(diag(Ω)_i)            (нормированный остаток)

    while max(r^N) > threshold:
        пометить максимальный r^N как BAD (status=False, quality=BAD)
        перезапустить estimate()
        пересчитать r^N

Возвращает ``BadDataResult`` со списком удалённых ``meas_id`` и финальным
``SEResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gridstate.api import estimate
from gridstate.result import SEResult
from gridstate.validation._diagnostics import compute_diagnostics


if TYPE_CHECKING:
    from gridstate.validation._diagnostics import Diagnostics
    from gridstate.working import Working, _ArrayCollection


logger = logging.getLogger(__name__)


# Код качества измерения «BAD» в контракте (значение колонки ``quality``).
QUALITY_BAD = 2


class ResidualTopRecord(NamedTuple):
    """One entry of ``NormalizedResidualReport.top_records`` (named fields).

    A ``NamedTuple`` (tuple subclass): positional access is preserved for
    backward compatibility while giving the fields readable names.

    Attributes:
        rn: нормированный остаток ``r^N``.
        meas_id: ``meas_id`` измерения.
        object_kind: ``0`` — NODE, ``1`` — BRANCH.
        object_id: ID объекта измерения (``-1`` если не найден).
        kind: ``MeasurementType`` (0=POWER_P, 1=POWER_Q, 2=VOLTAGE, ...).
        value: значение измерения (``nan`` если не найдено).
        residual: невязка ``r``.
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
    """Audit-only отчёт по нормированным остаткам после SE.

    Не модифицирует measurements (в отличие от ``remove_bad_data``). Просто
    диагностический срез: какие измерения подозрительные (r^N > порог),
    топ-N по нормированному остатку.

    Attributes:
        rn: массив нормированных остатков (m,). ``inf`` для non-redundant
            (``diag(Ω)`` ≤ 0).
        meas_ids: ``meas_id`` для каждого элемента ``rn`` (порядок совпадает
            с z-vector в текущей model).
        top_records: топ-N записей по ``|rn|`` (исключая ``inf``) —
            список :class:`ResidualTopRecord` (именованные поля).
        n_above_threshold: сколько измерений (исключая ``inf``) дают
            ``rn > rn_threshold``.
        rn_max_finite: ``max(rn[rn<inf])`` — максимум по информативным.
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
    """Собрать отчёт по нормированным остаткам после SE.

    Работает на текущем состоянии ``model`` (запустил ``estimate()`` ранее).
    Считает ``r^N = |r| / √(diag(Ω))``, где ``Ω = R − H G⁻¹ Hᵀ``. Возвращает
    топ-``top_n`` подозрительных и счётчик `rn > rn_threshold`. **Не**
    модифицирует measurements — audit-only.

    Args:
        model: ``Working`` с выполненной оценкой.
        measurements: ``_ArrayCollection`` — ``None`` → ``model.measurements``.
        top_n: сколько топ-записей вернуть (по убыванию ``|rn|``).
        rn_threshold: порог для подсчёта подозрительных (классически 3.0).

    Returns:
        ``NormalizedResidualReport``. ``inf``-значения исключаются из топа.
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
    """Результат итеративной чистки плохих данных.

    Attributes:
        se_result: финальный ``SEResult`` после удаления всех плохих
            измерений.
        removed_meas_ids: список ``meas_id`` измерений, помеченных как BAD.
        rn_max_history: значения ``max(r^N)`` после каждой итерации
            (включая последнюю, прошедшую тест).
        converged: ``True``, если на финальной итерации ``max(r^N) ≤ threshold``.
    """

    se_result: SEResult
    removed_meas_ids: list[int] = field(default_factory=list)
    rn_max_history: list[float] = field(default_factory=list)
    converged: bool = False


def _normalized_residuals(diag: Diagnostics) -> np.ndarray:
    """``r^N = |r| / √(diag(Ω))``, где ``Ω = R − H G⁻¹ Hᵀ``.

    Если ``Ω_ii`` численно равно нулю или отрицательно (плохая обусловленность
    G), элемент возвращается как ``+∞`` — такое измерение фактически
    «non-redundant», и нормированный остаток для него не определён.

    Используем плотную линейную алгебру: ``n_state = 2·n_bus − 1`` обычно
    мало (≤ десятков тысяч), поэтому ``G⁻¹`` через ``np.linalg.solve``
    значительно проще, чем sparse-инверсия.

    Note:
        ``gridstate.quality_summary._normalized_residuals`` computes the same
        quantity through a faster Cholesky + triangular-solve path. The two are
        kept separate on purpose (not merged): results may differ in the tails
        due to differing numerical paths, and this iterative bad-data driver
        must not depend on the quality-summary module.
    """
    H_dense = diag.H.toarray()  # (m × (2n−1))
    R_inv_dense = diag.R_inv.toarray()
    G = H_dense.T @ R_inv_dense @ H_dense  # (n_state × n_state)

    try:
        X = np.linalg.solve(G, H_dense.T)  # (n_state × m), X = G⁻¹ · Hᵀ
    except np.linalg.LinAlgError as exc:
        logger.error("Сбой при инверсии G в rn_max-тесте: %s", exc)
        return np.full_like(diag.r, np.inf, dtype=np.float64)

    # diag(H · G⁻¹ · Hᵀ) = построчное произведение H_dense и X.T
    HGH_diag = np.einsum("ij,ji->i", H_dense, X)
    omega_diag = diag.sigma2 - HGH_diag

    # Численная защита: «−0» и почти нули → ∞.
    omega_diag = np.where(omega_diag > 1e-12, omega_diag, np.nan)
    rn = np.abs(diag.r) / np.sqrt(omega_diag)
    return np.where(np.isnan(rn), np.inf, rn)


def remove_bad_data(
    model: Working,
    measurements: _ArrayCollection | None = None,
    rn_max_threshold: float = 3.0,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    estimate_max_iterations: int = 50,
) -> BadDataResult:
    """Итеративно удалить плохие измерения и вернуть финальный ``SEResult``.

    На каждой итерации:
        1. Запускается ``estimate()``;
        2. Считается ``r^N``;
        3. Если ``max(r^N) > threshold`` — плохое измерение помечается
           ``status=False, quality=BAD`` и цикл повторяется.

    Args:
        model: ``Working`` — обновляется in-place.
        measurements: коллекция — если ``None``, ``model.measurements``.
        rn_max_threshold: порог для нормированного остатка (классически 3.0).
        max_iterations: максимальное число итераций отбраковки (защита от
            зацикливания).
        tolerance: критерий сходимости для внутреннего ``estimate``.
        estimate_max_iterations: лимит итераций Gauss-Newton на каждый
            вызов ``estimate``.
    """
    if measurements is None:
        measurements = model.measurements
    assert measurements is not None  # для pyright

    removed: list[int] = []
    rn_history: list[float] = []

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
        )

    # Первый прогон — до цикла; цикл только повторяет estimate после отбраковки.
    se_result = _run_estimate()

    for it in range(max_iterations + 1):
        if not se_result.success:
            logger.warning("rn_max: SE не сошлась на итерации %d — прерываем чистку", it)
            return _result(converged=False)

        diag = compute_diagnostics(model, measurements)
        if diag.r.shape[0] == 0:
            logger.warning("rn_max: нет активных измерений — нечего проверять")
            return _result(converged=True)

        rn = _normalized_residuals(diag)
        rn_max = float(np.max(rn))
        rn_history.append(rn_max)

        if rn_max <= rn_max_threshold:
            logger.debug(
                "rn_max-тест пройден на итерации %d: max(r^N)=%.3f ≤ %.3f",
                it,
                rn_max,
                rn_max_threshold,
            )
            return _result(converged=True)

        # Иначе — отбрасываем худшее измерение.
        bad_pos = int(np.argmax(rn))
        bad_id = int(diag.meas_index.meas_id[bad_pos])
        bad_meas = measurements.get_by_id(bad_id)
        if bad_meas is None:
            logger.error(
                "rn_max: не нашли измерение id=%d в коллекции — прерываем",
                bad_id,
            )
            return _result(converged=False)
        bad_meas.status = False
        bad_meas.quality = QUALITY_BAD
        removed.append(bad_id)
        logger.info(
            "rn_max: itr %d, удалено измерение id=%d (r^N=%.2f > %.2f)",
            it,
            bad_id,
            rn_max,
            rn_max_threshold,
        )

        # Лимит итераций достигнут — не переоцениваем после последней отбраковки
        # (финальный se_result соответствует прошедшему прогону, как раньше).
        if it == max_iterations:
            break
        se_result = _run_estimate()

    logger.warning(
        "rn_max: достигнут лимит итераций (%d), а тест ещё не пройден",
        max_iterations,
    )
    return _result(converged=False)
