"""Piecewise static load characteristics (``load_characteristic_segments``).

A characteristic may define, separately for P and Q (``quantity`` 0 / 1), a set
of segments: segment ``k`` applies for ``v_lo[k] < V ≤ v_lo[k+1]``
(``V = |V|/Vnom``; a value on a boundary belongs to the lower segment), the
first one extends to ``V = 0`` whatever its ``v_lo``. The load factor on a
segment is ``c0 + c1·V + c2·V² + c3·V³ + c4·V⁴`` (1.0 = nominal).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


#: ``quantity`` codes of ``load_characteristic_segments``.
QUANTITY_P = 0
QUANTITY_Q = 1

#: ``(load_model_id, quantity) → (boundaries, coefficients)``: the upper
#: boundaries of all segments but the last (``v_lo[1:]``, increasing) and the
#: ``(n_segments, 5)`` coefficient matrix.
SegmentPlan = dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]


def segment_problems(segments: np.ndarray, n_models: int) -> list[str]:
    """Contract violations of a ``load_characteristic_segments`` table (empty = valid).

    ``n_models`` is the number of rows of ``load_characteristics`` (ids are
    0-based positions).
    """
    problems: list[str] = []
    if len(segments) == 0:
        return problems
    ids = np.asarray(segments["load_model_id"], dtype=np.int64)
    qty = np.asarray(segments["quantity"], dtype=np.int64)
    v_lo = np.asarray(segments["v_lo"], dtype=np.float64)
    unknown = sorted(set(ids[(ids < 0) | (ids >= n_models)].tolist()))
    if unknown:
        problems.append(f"load_model_id not in load_characteristics: {unknown}")
    bad_q = sorted(set(qty[(qty != QUANTITY_P) & (qty != QUANTITY_Q)].tolist()))
    if bad_q:
        problems.append(f"quantity must be 0 (P) or 1 (Q), got {bad_q}")
    coeffs = np.column_stack([np.asarray(segments[f"c{k}"], dtype=np.float64) for k in range(5)])
    if not np.all(np.isfinite(coeffs)):
        problems.append("non-finite coefficients")
    order = np.lexsort((v_lo, qty, ids))
    for a, b in pairwise(order):
        # The first segment's v_lo is ignored, so only later bounds must be
        # finite; equal bounds make an empty segment.
        same_pair = ids[a] == ids[b] and qty[a] == qty[b]
        if same_pair and (not np.isfinite(v_lo[b]) or v_lo[b] <= v_lo[a]):
            problems.append(
                f"v_lo must be finite and strictly increasing within "
                f"(load_model_id={ids[a]}, quantity={qty[a]})"
            )
    return problems


def segment_plan(segments: np.ndarray, n_models: int) -> SegmentPlan:
    """Group the segments by ``(load_model_id, quantity)``.

    Raises:
        ValueError: the table violates the contract (see :func:`segment_problems`).
    """
    problems = segment_problems(segments, n_models)
    if problems:
        raise ValueError("invalid load_characteristic_segments: " + "; ".join(problems))
    plan: SegmentPlan = {}
    if len(segments) == 0:
        return plan
    ids = np.asarray(segments["load_model_id"], dtype=np.int64)
    qty = np.asarray(segments["quantity"], dtype=np.int64)
    v_lo = np.asarray(segments["v_lo"], dtype=np.float64)
    coeffs = np.column_stack([np.asarray(segments[f"c{k}"], dtype=np.float64) for k in range(5)])
    order = np.lexsort((v_lo, qty, ids))
    for key in sorted(set(zip(ids.tolist(), qty.tolist(), strict=True))):
        rows = order[(ids[order] == key[0]) & (qty[order] == key[1])]
        plan[key] = (v_lo[rows][1:].copy(), coeffs[rows].copy())
    return plan


def segment_factor(bounds: np.ndarray, coeffs: np.ndarray, v: float) -> float:
    """Load factor at ``v`` (``bounds`` / ``coeffs`` of one plan entry)."""
    # Number of upper bounds strictly below v: on a boundary, the lower segment.
    k = int(np.searchsorted(bounds, v, side="left"))
    c = coeffs[k]
    return float(c[0] + v * (c[1] + v * (c[2] + v * (c[3] + v * c[4]))))


__all__ = [
    "QUANTITY_P",
    "QUANTITY_Q",
    "SegmentPlan",
    "segment_factor",
    "segment_plan",
    "segment_problems",
]
