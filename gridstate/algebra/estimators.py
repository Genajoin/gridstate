"""Robust SHGM (Schweppe-Huber GM) reweighting shared by the WLS and IPM solvers.

Extracted from ``gridstate.algorithms.wls`` / ``gridstate.algorithms.ipm``
(both adapted from pandapower, BSD 3-Clause — see the LICENSE file), where the
mask construction and the IRLS weight update used to be near-identical copies.

Two pieces live here:

* :func:`build_branch_pq_huber_mask` — which measurements are eligible for
  reweighting (branch P/Q flows minus transformer and leverage-Q exclusions);
* :class:`HuberReweighter` — the IRLS weight policy itself (sigma- or
  MAD-normalized residuals, one-shot adaptive tuning constant, weight floor).

References: Abur & Exposito ch. 6; Mili 1991 (SHGM), Mili 1996 (leverage).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from gridstate.z_vector import KIND_POWER_Q, OBJ_BRANCH


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.z_vector import MeasurementIndex


logger = logging.getLogger(__name__)


def build_branch_pq_huber_mask(
    meas_index: MeasurementIndex,
    network_pu: NetworkPU,
    *,
    m_total: int | None = None,
    skip_transformers: bool = True,
    leverage_b_threshold_pu: float = 2.0,
) -> np.ndarray:
    """Boolean mask of measurements eligible for SHGM reweighting.

    Only branch P/Q flows are reweighted: node V and P/Q injections are soft
    anchors and pseudo-priors — Huber on them wrecks max dV at terminals by
    downweighting pseudo-V rows with large residuals.

    Exclusions (both validated on regional models):

    * ``skip_transformers``: branch measurements on transformers
      (``|tap_ratio - 1| > 1e-3``). With imprecise RPN their flow residuals
      are routinely large, and downweighting them detaches the LV block bus
      from HV — V on the LV side drifts.
    * ``leverage_b_threshold_pu``: Q measurements on high-charging branches
      (``|B_pu| >= threshold``) are leverage points in the Mili 1996 sense.
      Their values are physically legitimate (``Q_charging ~ B*V^2``), and
      downweighting them tears V apart at the terminals. With S_base=100,
      ``|B| >= 2.0`` pu is ~200 Mvar of charging — typical for 500/750 kV
      lines longer than ~100 km.

    Args:
        meas_index: measurement metadata in z-order.
        network_pu: p.u. network (tap ratios and branch charging).
        m_total: total row count of the residual vector when it is longer
            than ``meas_index`` (IPM appends balance/prior rows); the extra
            rows are never reweighted. ``None`` -> ``len(meas_index)``.
        skip_transformers: apply the transformer exclusion.
        leverage_b_threshold_pu: leverage-Q threshold; ``<= 0`` disables it.

    Returns:
        Boolean array of length ``m_total`` (or ``len(meas_index)``).
    """
    n = len(meas_index)
    m = n if m_total is None else int(m_total)

    mask = np.zeros(m, dtype=bool)
    mask[:n] = np.asarray(meas_index.object_kind == OBJ_BRANCH, dtype=bool)
    if not mask.any():
        return mask

    branch_pos = np.zeros(m, dtype=np.int64)
    branch_pos[:n] = np.asarray(meas_index.object_pos, dtype=np.int64)
    kinds = -np.ones(m, dtype=np.int64)
    kinds[:n] = np.asarray(meas_index.kind, dtype=np.int64)

    n_branch = network_pu.n_branch
    if skip_transformers and n_branch > 0:
        is_xfmr_br = np.abs(network_pu.tap_ratio - 1.0) > 1e-3
        is_xfmr = np.zeros(m, dtype=bool)
        is_xfmr[mask] = is_xfmr_br[branch_pos[mask]]
        mask = mask & (~is_xfmr)
    if leverage_b_threshold_pu > 0 and n_branch > 0:
        is_leverage_br = np.abs(network_pu.branch_b) >= leverage_b_threshold_pu
        is_leverage_q = np.zeros(m, dtype=bool)
        q_mask = mask & (kinds == KIND_POWER_Q)
        is_leverage_q[q_mask] = is_leverage_br[branch_pos[q_mask]]
        mask = mask & (~is_leverage_q)
    return mask


class HuberReweighter:
    """IRLS weight policy of the SHGM estimator.

    Weights are ``w_i = min(1, c_eff / |r_n_i|)`` floored at ``w_floor`` and
    applied only inside ``mask``; rows outside the mask keep weight 1.

    Normalization of residuals: ``r_n = |r| / sigma`` by default, or MAD-based
    (``|r| / (1.4826 * median|r[mask]|)``) with ``use_mad``.

    The tuning constant is adaptive one-shot: on the first ``weights()`` call
    (and after every ``reset_adaptive()``) ``c_eff`` is raised to
    ``max(c, adaptive_k * median(|r_n|[mask]))``, so noisy telemetry with a
    large typical residual automatically gets a liberal constant. The IPM
    resets it each outer iteration because residuals drop sharply as the
    barrier weakens; the WLS applies it once.
    """

    def __init__(
        self,
        *,
        c: float,
        mask: np.ndarray,
        sigma: np.ndarray,
        w_floor: float,
        adaptive_k: float,
        use_mad: bool = False,
    ) -> None:
        self._c = float(c)
        self._mask = np.asarray(mask, dtype=bool)
        self._sigma = sigma
        self._w_floor = float(w_floor)
        self._adaptive_k = float(adaptive_k)
        self._use_mad = bool(use_mad)
        self._mask_any = bool(self._mask.any())
        self._c_eff = float(c)
        self._adaptive_applied = False

    @property
    def enabled(self) -> bool:
        """True when reweighting can have any effect (c > 0 and mask non-empty)."""
        return self._c > 0.0 and self._mask_any

    @property
    def c_eff(self) -> float:
        """Current effective tuning constant (after adaptive raise, if any)."""
        return self._c_eff

    def reset_adaptive(self) -> None:
        """Recompute ``c_eff`` from residuals on the next ``weights()`` call."""
        self._adaptive_applied = False

    def weights(self, r: np.ndarray) -> np.ndarray:
        """Weight vector (same length as ``r``) for the current residuals."""
        if self._use_mad and self._mask_any:
            ar = np.abs(r[self._mask])
            mad_scale = float(np.median(ar)) * 1.4826 + 1e-12
            r_n = np.abs(r) / mad_scale
        else:
            r_n = np.abs(r) / self._sigma
        if not self._adaptive_applied and self._mask_any:
            med = float(np.median(r_n[self._mask]))
            self._c_eff = max(self._c, self._adaptive_k * med)
            self._adaptive_applied = True
            logger.debug("SHGM adaptive c: median(|r_n|)=%.3f -> c_eff=%.2f", med, self._c_eff)
        w: np.ndarray = np.ones_like(r_n)
        sel = self._mask & (r_n > self._c_eff)
        w[sel] = np.maximum(self._c_eff / np.maximum(r_n[sel], 1e-30), self._w_floor)
        return w
