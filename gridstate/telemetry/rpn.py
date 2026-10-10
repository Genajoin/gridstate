"""On-load / off-circuit tap changers: apply selected taps to tap_ratio/phase_shift/shunt.

The core ``_apply_tap_steps_on_arrays`` applies a ready ``tap_steps`` table to the
model branches through the ``apply_rpn_resolved`` wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridstate.utils import id_to_pos_map


if TYPE_CHECKING:
    from gridstate.working import Working


def _apply_tap_steps_on_arrays(
    branches_arr: np.ndarray, tap_steps_arr: np.ndarray
) -> dict[str, int | float]:
    """Apply the ``tap_steps`` contract to branches (tap/phase + shunt rescaling).

    Mutates ``branches_arr``.

    The tap choice is made outside the core: ``tap_steps`` carries the target ``tap_ratio``/
    ``phase_shift`` + ``shunt_factor`` (=(tap_new/tap_old)², 1.0 = no rescaling).
    The kernel only applies them: it writes tap/phase and rescales the branch
    shunt (fields ``!=0.0``).
    This is format-agnostic transformer physics.
    """
    by_id = id_to_pos_map(branches_arr["id"])
    stats: dict[str, int | float] = {"applied": 0, "shunt_recalc": 0}
    for ts in tap_steps_arr:
        idx = by_id.get(int(ts["branch_id"]))
        if idx is None:
            continue
        branches_arr[idx]["tap_ratio"] = float(ts["tap_ratio"])
        branches_arr[idx]["phase_shift"] = float(ts["phase_shift"])
        stats["applied"] += 1

        factor = float(ts["shunt_factor"])
        if factor != 1.0:  # 1.0: no rescaling requested; the product would be a no-op
            recalc_done = False
            for field in (
                "conductance",
                "susceptance",
                "conductance_from",
                "susceptance_from",
                "conductance_to",
                "susceptance_to",
            ):
                val = float(branches_arr[idx][field])
                if val != 0.0:
                    branches_arr[idx][field] = val * factor
                    recalc_done = True
            if recalc_done:
                stats["shunt_recalc"] += 1
    return stats


def apply_rpn_resolved(model: Working) -> dict[str, int | float]:
    """Apply the selected taps (the ``tap_steps`` table) to the model branches.

    The tap position is chosen upstream; the kernel :func:`_apply_tap_steps_on_arrays`
    writes tap/phase and rescales the branch shunt. The ``run()`` step calls it
    before ``apply_telemetry`` (the tap affects variances and the Y-bus). An empty
    ``tap_steps`` is a no-op.
    """
    tap_steps_coll = getattr(model, "tap_steps", None)
    tap_steps_arr = tap_steps_coll.to_numpy() if tap_steps_coll is not None else None
    if tap_steps_arr is None or len(tap_steps_arr) == 0:
        return {"applied": 0, "shunt_recalc": 0}
    branches_arr = model.branches.to_numpy().copy()
    stats = _apply_tap_steps_on_arrays(branches_arr, tap_steps_arr)
    model.branches.update_from_array(branches_arr)
    return stats
