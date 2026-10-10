"""σ floor for real-flow measurements from the measurement channel scale.

The typical telemetry σ of flow measurements is proportional to the value (≈ α·|z|): for
small flows σ is unphysically low (20 MW flow → σ 2 MW),
and the estimator over-trusts small flows, which gives a global downward bias of V.
Physically, the lower bound of the channel error is set by the CT/VT accuracy class
relative to the channel **scale**, not to the current value::

    σ_min = kv_frac · S_scale,   S_scale = √3 · Vn · I_nom   (MV·A)

where Vn is the branch voltage class = max(Vn of the ends), I_nom is the rated primary
current of the CT (typically 1 kA). The floor is applied ONLY to real (non-pseudo)
branch-flow P and Q measurements: ``variance := max(variance, floor²)``,
``weight := 1/variance``.

A/B on 4 regional models (IPM, kv_frac=0.010):
p50|V_se − V_ref| −8…−15 % on three of four (the fourth +2.5 % in p50,
but p95 5.64→5.25 and class-max is better); at 500 kV p50 up to −55 %; p95 better
than the baseline on all four; dV_max regresses by more than 5 % nowhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import id_value_map, meas_mask
from gridstate.constants import MeasurementObjectType, MeasurementType
from gridstate.telemetry.noise import branch_nominal_kv, flow_sigma_floor
from gridstate.working import table_copy, table_replace


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = ["apply_flow_sigma_floor"]


def apply_flow_sigma_floor(
    model: Working,
    *,
    kv_frac: float,
    current_ka: float = 1.0,
) -> dict:
    """Raise the variance of real branch-flow measurements to floor² of the channel scale.

    Args:
        model: ``Working`` after telemetry is applied (pseudo-measurements, if
            already added, are untouched: selector ``is_pseudo == 0``).
        kv_frac: share of the scale ``√3·Vn·current_ka`` taken as σ_min (channel
            accuracy class); 0.010 = 1 % of scale (110 kV → 1.9 MW, 500 kV → 8.7 MW).
        current_ka: rated primary CT current, kA (default 1.0, the typical value).

    Returns:
        ``{"checked": N, "floored": N}``: how many real-flow measurements were checked
        and for how many the variance was raised to floor².
    """
    meas_arr = table_copy(model.measurements)
    nodes_arr = model.nodes.to_numpy()
    branches_arr = model.branches.to_numpy()
    stats = _flow_sigma_floor_on_arrays(
        meas_arr,
        nodes_arr,
        branches_arr,
        ot_branch=int(MeasurementObjectType.BRANCH),
        mt_p=int(MeasurementType.POWER_P),
        mt_q=int(MeasurementType.POWER_Q),
        kv_frac=float(kv_frac),
        current_ka=float(current_ka),
    )
    table_replace(model.measurements, meas_arr)
    return stats


def _flow_sigma_floor_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    *,
    ot_branch: int,
    mt_p: int,
    mt_q: int,
    kv_frac: float,
    current_ka: float,
) -> dict:
    """CORE: σ floor from the channel scale over the contract arrays.

    Mutates ``meas_arr`` in place (``variance``/``weight``), reads
    ``nodes_arr`` (``voltage_nominal``) and ``branches_arr`` (``from_node``/
    ``to_node``). Vectorized; the python loop is only in the branch_id → Vn mapping.
    Measurements on unknown branches get floor 0 (left untouched).
    """
    vn_by_node = id_value_map(nodes_arr, "voltage_nominal")
    vn_by_branch = branch_nominal_kv(vn_by_node, branches_arr, policy="max")

    # No status filter: inactive real flow rows are floored as well.
    sel = meas_mask(meas_arr, kinds=[mt_p, mt_q], obj=ot_branch, active=False, pseudo=False)
    floor = flow_sigma_floor(
        np.fromiter(
            (vn_by_branch.get(int(oid), 0.0) for oid in meas_arr["object_id"]),
            dtype=float,
            count=len(meas_arr),
        ),
        kv_frac=kv_frac,
        current_ka=current_ka,
    )
    floor2 = floor * floor

    low = sel & (meas_arr["variance"] < floor2)
    meas_arr["variance"][low] = floor2[low]
    meas_arr["weight"][low] = 1.0 / meas_arr["variance"][low]
    return {"checked": int(sel.sum()), "floored": int(low.sum())}
