"""Single semantics of "unset" bounds for the node box fields.

Contract convention (``NODE_DTYPE``: ``load_p_min/max``,
``generation_q_min/max`` etc.): a pair of bounds is considered **unset**
if both values are zero (the numpy default of an unfilled field) or
both are sentinels (the EXACT value ±9999 means "no data" in the source
formats).

The sentinel is exactly ±9999, NOT a "large enough" threshold: real data
carries limits of the same magnitude (an equivalent node of a large power
plant: ``load_p ∈ [10000, 30000]`` MW; BUS equivalents: ±42500), and a
threshold rule |v| ≥ 9000 swallowed them, replacing the physical range with
the default box.

Before this module every subsystem interpreted bounds its own way:
``ipm_setup`` understood zeros and sentinels, while the WLS spread clip
(``post_processing._clip``) did not, so an unfilled (0, 0) pair
**zeroed** the estimate and a sentinel ±9999 pair produced garbage sums in
generator aggregation. Here there is one rule for everyone.
"""

from __future__ import annotations

import math


__all__ = [
    "SENTINEL_VALUE",
    "bounds_unset",
    "declared_box",
    "is_reactive_only_generation",
    "is_sentinel",
    "resolve_bounds",
]

#: Exact "no data" sentinel value (contract convention ±9999).
SENTINEL_VALUE = 9999.0

#: Tolerance of the sentinel comparison (the value may come from a text format).
_SENTINEL_TOL = 1e-6


def is_sentinel(value: float) -> bool:
    """Whether the value is the "no data" sentinel (exactly ±9999)."""
    return abs(abs(float(value)) - SENTINEL_VALUE) < _SENTINEL_TOL


def resolve_bounds(lo_raw: float, hi_raw: float) -> tuple[float, float]:
    """The pair ``(lo, hi)`` with unset sides replaced by ±inf.

    Rules:

    * both sides ≈0 → ``(-inf, +inf)`` — the pair was never filled
      (numpy default); a real "exactly [0, 0]" range does not occur on an
      active node — such a node carries no ``exist_*``. The exception is
      compensator generation (controlled shunt reactor, static var
      compensator, synchronous condenser): P exactly [0, 0] with a declared
      Q range, see :func:`is_reactive_only_generation`;
    * BOTH sides are sentinels (exactly ±9999) → the pair is unset →
      ``(-inf, +inf)``. A half-sentinel pair is kept **as is**
      (the other side is data);
    * ``lo > hi`` → ``(-inf, +inf)`` — the bounds are invalid and are not
      enforced (such pairs used to either zero the estimate or be
      silently skipped).

    Consumers: the WLS spread clip (``±inf`` = "do not clip"), the IPM
    box vars (``±inf`` = "substitute a wide default").
    """
    lo = float(lo_raw)
    hi = float(hi_raw)
    if abs(lo) < 1e-9 and abs(hi) < 1e-9:
        return (float("-inf"), float("inf"))
    if is_sentinel(lo) and is_sentinel(hi):
        return (float("-inf"), float("inf"))
    if lo > hi:
        return (float("-inf"), float("inf"))
    return (lo, hi)


def declared_box(
    lo_raw: float, hi_raw: float, *, allow_point: bool = False
) -> tuple[float, float] | None:
    """The pair resolved by :func:`resolve_bounds` when both sides are set, else ``None``.

    "Set" means both resolved bounds are finite and ``hi > lo``; with
    ``allow_point`` a point box ``hi == lo`` counts as well (finite sides only:
    :func:`resolve_bounds` never returns ``lo > hi``).
    """
    lo, hi = resolve_bounds(lo_raw, hi_raw)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if not allow_point and not hi > lo:
        return None
    return (lo, hi)


def bounds_unset(lo_raw: float, hi_raw: float) -> bool:
    """Neither side of the pair is set (both resolved bounds are infinite)."""
    lo, hi = resolve_bounds(lo_raw, hi_raw)
    return not (math.isfinite(lo) or math.isfinite(hi))


def is_reactive_only_generation(p_min: float, p_max: float, q_min: float, q_max: float) -> bool:
    """Reactive-only generation: ``P ∈ [0, 0]`` with a declared non-degenerate Q.

    Network models represent compensators (controlled shunt reactors, static
    var compensators, STATCOMs, synchronous condensers) as generators with no
    active power. Their P pair [0, 0] is data, not "bounds unset": treating it
    via :func:`resolve_bounds` would open up active power for the device.
    """
    if float(p_min) != 0.0 or float(p_max) != 0.0:
        return False
    return declared_box(q_min, q_max) is not None
