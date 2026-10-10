"""Measurement noise model: the variance rules of the telemetry layer.

One place for the sigma / variance formulas that the telemetry steps assign to
measurements. The steps still apply them in their own pipeline order, which is
part of the result:

1. ``apply_telemetry`` sets the base variance of every activated reading:
   :func:`variance_voltage` (node V), :func:`variance_branch_q` (branch Q,
   charging-aware, branch Vn of the *from* node), :func:`variance_power`
   (branch P) and :func:`variance_injection` (node injections), then scales
   QUESTIONABLE readings by :data:`QUESTIONABLE_SIGMA2_MULTIPLIER`, applies
   the producer's sigma^2 plans and the Q-loss downweight;
2. ``voltage_range_filter`` downweights out-of-range V readings by the same
   QUESTIONABLE multiplier;
3. ``widen_generation_only_injections`` adds the variance of a load uniform in
   its box (:func:`uniform_box_variance`);
4. ``gen_v_calibration`` sets the V variance of generating and slack nodes to
   :data:`GEN_V_SIGMA2` (:func:`gen_v_calibrated_variance`); the V mirror
   step then copies variances as they are;
5. ``flow_sigma_floor`` raises real branch-flow variances to the square of
   :func:`flow_sigma_floor` (branch Vn = the larger end).

The two branch nominal-voltage policies (``"from"`` for the charging floor,
``"max"`` for the channel floor) are kept apart in :func:`branch_nominal_kv`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "GEN_V_SIGMA2",
    "QUESTIONABLE_SIGMA2_MULTIPLIER",
    "branch_nominal_kv",
    "flow_sigma_floor",
    "gen_v_calibrated_variance",
    "uniform_box_variance",
    "variance_branch_q",
    "variance_injection",
    "variance_power",
    "variance_voltage",
]

#: Variance multiplier of a QUESTIONABLE reading (sigma x 10).
QUESTIONABLE_SIGMA2_MULTIPLIER = 100.0

#: Target V variance of generating and slack nodes (gen-V calibration), in the
#: units of the model's V readings (kV^2: sigma ~ 0.32 kV).
GEN_V_SIGMA2 = 0.1

_VARIANCE_V_FRAC = 0.01  # 1% of the nominal voltage
_VARIANCE_P_BASE = 1.0  # MW² floor
_VARIANCE_Q_BASE = 1.0  # MVAr² floor
_VARIANCE_P_FRAC = 0.02  # 2% of the power value
_VARIANCE_Q_FRAC = 0.02

# Node injection readings: sigma 5 % of |value| (not 2 % as branch flows): a net
# injection aggregates several components, so it is less certain than a
# directed branch flow; a tighter sigma broke convergence. Floor per component.
_INJECTION_SIGMA_FRAC = 0.05
_INJECTION_BASE = {"P": 1.0, "Q": 0.5}


def variance_voltage(value_kv: float, vn_kv: float) -> float:
    """σ²_V in kV²: 1% of nominal, at least 0.1 kV.

    Independent of the current value (the variance is a prior on nominal).

    ``value_kv`` is ignored; it stays in the signature for existing positional
    callers.
    """
    sigma = max(_VARIANCE_V_FRAC * vn_kv, 0.1)
    return float(sigma * sigma)


def variance_power(
    value_mva: float,
    base: float = _VARIANCE_P_BASE,
    sigma_frac: float = _VARIANCE_P_FRAC,
) -> float:
    """σ² in MW²: ``sigma_frac`` of |value|, at least 0.5 MW, plus the base floor.

    For P measurements the default is ``sigma_frac=0.02`` (2 % of |value|).
    For Q branch-flow measurements on large models a larger value
    (0.05–0.07) is sometimes justified: the Q flow depends strongly on V and
    δ, and a tight σ_Q forces the estimate to bend the voltage of a whole
    area to fit broken or inconsistent Q telemetry.
    """
    sigma = max(sigma_frac * abs(value_mva), 0.5)
    return float(sigma * sigma + base)


def variance_branch_q(
    value_mvar: float,
    *,
    charging_mvar: float = 0.0,
    charging_alpha: float = 0.0,
    base: float = _VARIANCE_Q_BASE,
    sigma_frac: float = _VARIANCE_Q_FRAC,
) -> float:
    """σ² for branch Q with a **charging-aware** floor (single source of truth).

    On long HV/EHV lines the model branch Q is dominated by the charging power
    of the π-model ``Q_charge ≈ V²·B`` (hundreds to thousands of MVAr at 750 kV), while
    the measured net Q at the line end is small. A flat ``σ_frac·|Q_meas|`` (1–3 MVAr
    for small Q) yields a normalized residual of tens to hundreds of σ and pulls the
    regional V down through the normal equations. Hence a minimal
    absolute floor ``σ_charge = charging_alpha·|charging_mvar|`` is introduced::

        σ² = max( variance_power(value, sigma_frac), (charging_alpha·|charging_mvar|)² )

    where ``charging_mvar = |B_si|·Vn²`` (full charging, see
    :func:`gridstate.telemetry.loss_filter.compute_expected_q_imbalance_mvar`).
    Used by ``apply_telemetry`` (``gridstate.telemetry.apply_resolved``) and by
    callers that build measurement variances themselves.

    Args:
        value_mvar: measured branch Q flow, MVAr.
        charging_mvar: branch ``|B_si|·Vn²``, MVAr (full charging). 0 = none.
        charging_alpha: share of the charging power that goes into the σ floor. 0 = off (= ``variance_power``).
        base: floor, MVAr².
        sigma_frac: relative σ (default 0.07).

    Returns:
        σ² in MVAr².
    """
    var_frac = variance_power(value_mvar, base=base, sigma_frac=sigma_frac)
    if charging_alpha > 0.0 and charging_mvar != 0.0:
        sigma_charge = charging_alpha * abs(charging_mvar)
        return max(var_frac, float(sigma_charge * sigma_charge))
    return var_frac


def variance_injection(value_mva: float, pq: Literal["P", "Q"]) -> float:
    """Variance of a node injection reading (MW^2 / Mvar^2).

    ``max(0.05·|value|, 0.5)^2`` plus a floor of 1.0 (P) or 0.5 (Q).
    """
    return variance_power(value_mva, base=_INJECTION_BASE[pq], sigma_frac=_INJECTION_SIGMA_FRAC)


def uniform_box_variance(lo: float, hi: float) -> float:
    """Variance of a quantity uniform in ``[lo, hi]``: ``(hi - lo)^2 / 12``."""
    return (hi - lo) ** 2 / 12.0


def gen_v_calibrated_variance(current: float, sigma2: float, *, tighten: bool) -> float | None:
    """New V variance of a calibrated generator/slack reading, ``None`` to keep it.

    ``tighten=False`` ("absolute" mode) always returns ``sigma2``; with
    ``tighten`` a finite variance already in ``(0, sigma2]`` is kept.
    """
    if tighten and np.isfinite(current) and 0.0 < current <= sigma2:
        return None
    return float(sigma2)


def flow_sigma_floor(vn_kv: np.ndarray, *, kv_frac: float, current_ka: float) -> np.ndarray:
    """Sigma floor of branch-flow readings from the channel scale, per branch Vn.

    ``kv_frac · √3 · Vn · I_nom`` (MVA): the accuracy class of the instrument
    transformers applies to the channel scale, not to the current value.
    """
    scale = kv_frac * np.sqrt(3.0) * current_ka
    floor: np.ndarray = scale * vn_kv
    return floor


def branch_nominal_kv(
    vn_by_node: Mapping[int, float],
    branches_arr: np.ndarray,
    *,
    policy: Literal["from", "max"],
) -> dict[int, float]:
    """Nominal voltage per branch id from its end nodes (``0.0`` if unknown).

    ``"from"``: the *from* node (charging floor of branch Q, Q-loss filter);
    ``"max"``: the larger of the two ends (channel sigma floor of flows).
    """
    if policy == "from":
        return {
            int(bid): vn_by_node.get(int(fn), 0.0)
            for bid, fn in zip(branches_arr["id"], branches_arr["from_node"], strict=True)
        }
    return {
        int(b["id"]): max(
            vn_by_node.get(int(b["from_node"]), 0.0),
            vn_by_node.get(int(b["to_node"]), 0.0),
        )
        for b in branches_arr
    }
