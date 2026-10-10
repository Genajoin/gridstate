"""V-measurement filter: a wide [V_lo, V_hi] range with a margin.

Implements the standard plausibility check of measurements against the
admissible and nominal values. The goal is to reject only readings that are
certainly broken (V<<V_critical, V>>V_max) without losing valid edge values.

Lower threshold (priority -- the most lenient):
* `voltage_critical`, the emergency lower limit; the first constraint,
  usually supplied for all nodes;
* fallback `V_nom · (1 − fallback_pct/100)` (default 20 %).

Upper threshold:
* `voltage_max · (1 + upper_margin_pct/100)` -- voltage_max itself is the
  "normal operating range", so a margin above it is needed for edge values;
* fallback `V_nom · (1 + fallback_pct/100)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from gridstate._arrays import meas_mask
from gridstate.bounds import is_sentinel
from gridstate.constants import (
    MeasurementObjectType,
    MeasurementQuality,
    MeasurementType,
    NodeType,
)
from gridstate.telemetry._filters import downweight_measurement, validate_downweight_action
from gridstate.telemetry.noise import (
    GEN_V_SIGMA2,
    QUESTIONABLE_SIGMA2_MULTIPLIER,
    gen_v_calibrated_variance,
)
from gridstate.utils import id_to_pos_map
from gridstate.working import table_copy, table_replace


if TYPE_CHECKING:
    from gridstate.working import Working


def apply_voltage_range_filter(
    model: Working,
    *,
    upper_margin_pct: float = 10.0,
    min_voltage_nominal_kv: float = 110.0,
    upper_fallback_factor: float = 1.4,
    action: str = "downweight",
    deactivate_nonphysical: bool = False,
    nonphysical_lower_factor: float = 0.5,
    nonphysical_upper_factor: float = 1.5,
    questionable_sigma2_multiplier: float = QUESTIONABLE_SIGMA2_MULTIPLIER,
) -> dict:
    """Deactivate/downweight V measurements outside the admissible range.

    The fallbacks handle placeholder limits that real inputs contain, e.g. a
    critical limit of 1.0 on 500 kV nodes, or a missing upper limit on a large
    share of high-voltage nodes:

    * Lower bound: ``voltage_critical`` if it is ≥ ``V_nom/2``
      (a valid value). Otherwise ``voltage_min`` if it is ≥ ``V_nom/2``.
      Otherwise ``V_nom/2`` (so that a placeholder critical limit of 1.0 does
      not let any V≥1 through).
    * Upper bound: ``voltage_max · (1 + upper_margin_pct/100)``;
      if ``voltage_max == 0`` -- fallback ``V_nom · upper_fallback_factor``.

    Args:
        model: Working with telemetry already applied.
        upper_margin_pct: margin above voltage_max (default 10 %).
            voltage_max is "admissible in the normal regime"; for filtering
            broken telemetry we take `voltage_max · 1.10`. On a node with
            Vn=10.5, voltage_max=11.6 (=1.1·Vn) this gives an upper bound of 12.76.
        upper_fallback_factor: multiplier of V_nom when `voltage_max=0`
            (default 1.4 = +40 %). Many inputs lack the upper limit on a
            large share of nodes ≥500 kV; a wide bound is used then.
        min_voltage_nominal_kv: minimum voltage class to check.
            Default 110 kV -- generator buses (6-21 kV) and low-voltage loads
            are ignored. Their V telemetry is usually at the edge of the range
            (voltage_max), and through transformers these measurements are
            important anchors for the 110-750 kV part. Setting it to 0 checks
            all nodes.
        action: what to do with out-of-range measurements:
            ``"downweight"`` (default) -- increase variance × multiplier
            (soft, WLS decides); ``"deactivate"`` -- `status=False`
            (hard, may make things worse if V measurements are the only anchor).
        deactivate_nonphysical: if True, V measurements that are physically
            impossible as a voltage (``V<=0``, ``V < V_nom·nonphysical_lower_factor``
            or ``V > V_nom·nonphysical_upper_factor``) are **deactivated**
            (``status=False``) regardless of ``action``. Rationale:
            ``downweight`` (σ×10) does not suppress grossly non-physical values --
            a broken U sensor with ``V=2533`` kV on a 20 kV bus (pu=127), a
            negative ``V`` or ``V=0.2`` pu (deep collapse / broken sensor) keeps
            a huge weighted residual and wrecks the solve (J~1e8, node collapse).
            A soft out-of-range in [lo, hi] is still handled by
            ``action``. **Default False** (keeps the previous behaviour bit for bit).
        nonphysical_lower_factor: lower physical limit of V as a multiple of
            V_nom (default 0.5 = 50 % of V_nom, matches the soft floor ``half_nom``).
            A V below this on an energized bus cannot be a valid voltage
            (symmetric to the upper limit). ``V<=0`` is always deactivated,
            regardless of the factor value.
        nonphysical_upper_factor: upper physical limit of V as a multiple of
            V_nom for ``deactivate_nonphysical`` (default 1.5 = 150 % of V_nom).
            A V above this cannot be a valid voltage.
        questionable_sigma2_multiplier: variance multiplier for
            QUESTIONABLE measurements (default 100.0 = σ × 10).

    Returns:
        ``{"checked": N, "out_of_range": N, "hard_deactivated": N,
        "by_vnom": {kv: N}}``
    """
    # The object/measurement-type enums are resolved here (gridstate.constants);
    # the core takes ready ints and reads only contract columns.
    validate_downweight_action(action)
    meas_arr = table_copy(model.measurements)
    nodes_arr = model.nodes.to_numpy()
    stats = _voltage_range_filter_on_arrays(
        meas_arr,
        nodes_arr,
        ot_node=int(MeasurementObjectType.NODE),
        mt_v=int(MeasurementType.VOLTAGE),
        upper_margin_pct=upper_margin_pct,
        min_voltage_nominal_kv=min_voltage_nominal_kv,
        upper_fallback_factor=upper_fallback_factor,
        action=action,
        deactivate_nonphysical=deactivate_nonphysical,
        nonphysical_lower_factor=nonphysical_lower_factor,
        nonphysical_upper_factor=nonphysical_upper_factor,
        questionable_sigma2_multiplier=questionable_sigma2_multiplier,
    )
    table_replace(model.measurements, meas_arr)
    return stats


def _voltage_range_filter_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    *,
    ot_node: int,
    mt_v: int,
    upper_margin_pct: float,
    min_voltage_nominal_kv: float,
    upper_fallback_factor: float,
    action: str,
    questionable_sigma2_multiplier: float,
    deactivate_nonphysical: bool = False,
    nonphysical_lower_factor: float = 0.5,
    nonphysical_upper_factor: float = 1.5,
) -> dict:
    """CORE: V-range filter over contract arrays.

    Mutates ``meas_arr`` in place (``status``/``variance``/``weight``/``quality``)
    and reads ``nodes_arr`` (``voltage_nominal``/``voltage_critical``/``voltage_min``/
    ``voltage_max``). The object and measurement type enums arrive as ready ints
    from the adapter. No external dependencies; a sequential loop in input order.
    """
    node_by_id = id_to_pos_map(nodes_arr["id"])

    counters: dict[str, int] = {
        "checked": 0,
        "out_of_range": 0,
        "hard_deactivated": 0,
    }
    by_vnom: dict[int, int] = {}

    # Only the current row is mutated in the loop, so the mask taken up front
    # selects exactly the rows the former per-row status/type checks did.
    for i in np.flatnonzero(meas_mask(meas_arr, kinds=mt_v, obj=ot_node)).tolist():
        m = meas_arr[i]
        node_id = int(m["object_id"])
        ni = node_by_id.get(node_id)
        if ni is None:
            continue
        v_nom = float(nodes_arr[ni]["voltage_nominal"])
        if v_nom <= 0 or v_nom < min_voltage_nominal_kv:
            continue
        v_meas = float(m["value"])

        # Hard deactivation of PHYSICALLY IMPOSSIBLE V (nonphysical): V<=0,
        # V < V_nom·lower_factor or V > V_nom·upper_factor. downweight (σ×10)
        # does not suppress them -- a broken U sensor with pu=127 ("2533 kV" on
        # a 20 kV bus), a negative V or V=0.2 pu (deep collapse) keeps a huge
        # weighted residual and wrecks the solve (J~1e8, collapse). Such values
        # cannot be a voltage -> status=False, not downweight. The limits are
        # symmetric (below and above); V<=0 is always deactivated.
        if deactivate_nonphysical and (
            v_meas <= 0
            or v_meas < v_nom * nonphysical_lower_factor
            or v_meas > v_nom * nonphysical_upper_factor
        ):
            counters["out_of_range"] += 1
            counters["hard_deactivated"] += 1
            v_nom_int = round(v_nom)
            by_vnom[v_nom_int] = by_vnom.get(v_nom_int, 0) + 1
            meas_arr[i]["status"] = False
            continue

        if v_meas <= 0:
            # V<=0 on an active node -- deactivate as certainly invalid.
            # apply_telemetry already filters V<50%Vn for TM formulas, but the
            # measurement may have come from an external source (manual seed,
            # SQL import, repeated apply); this guard closes the hole.
            counters["out_of_range"] += 1
            v_nom_int = round(v_nom)
            by_vnom[v_nom_int] = by_vnom.get(v_nom_int, 0) + 1
            if action == "deactivate":
                meas_arr[i]["status"] = False
            else:
                downweight_measurement(
                    meas_arr, i, questionable_sigma2_multiplier, mark_questionable=True
                )
            continue

        # Lower threshold: voltage_critical -> voltage_min -> V_nom/2.
        # Some inputs carry a placeholder critical limit of 1.0 on nodes
        # ≥500 kV; the validity criterion "X ≥ V_nom/2" discards 1.0 as a
        # placeholder (1 ≪ 250 for 500 kV).
        half_nom = v_nom * 0.5
        v_crit = float(nodes_arr[ni]["voltage_critical"])
        v_min = (
            float(nodes_arr[ni]["voltage_min"])
            if "voltage_min" in (nodes_arr.dtype.names or ())
            else 0.0
        )
        if v_crit >= half_nom:
            lo = v_crit
        elif v_min >= half_nom:
            lo = v_min
        else:
            lo = half_nom

        # Upper threshold: voltage_max·(1+upper_margin_pct) -> fallback V_nom·factor.
        # Many inputs lack the upper limit on a large share of nodes ≥500 kV;
        # fallback 1.4·V_nom.
        v_max = float(nodes_arr[ni]["voltage_max"])
        hi = (
            v_max * (1.0 + upper_margin_pct / 100.0) if v_max > 0 else v_nom * upper_fallback_factor
        )

        counters["checked"] += 1
        v_nom_int = round(v_nom)
        by_vnom[v_nom_int] = by_vnom.get(v_nom_int, 0) + 1

        if v_meas < lo or v_meas > hi:
            counters["out_of_range"] += 1
            if action == "deactivate":
                meas_arr[i]["status"] = False
            else:  # downweight
                downweight_measurement(
                    meas_arr, i, questionable_sigma2_multiplier, mark_questionable=True
                )
            continue

    return {
        "checked": counters["checked"],
        "out_of_range": counters["out_of_range"],
        "hard_deactivated": counters["hard_deactivated"],
        "by_vnom": by_vnom,
    }


def apply_voltage_meas_calibration_for_gen_nodes(
    model: Working,
    *,
    sigma2: float = GEN_V_SIGMA2,
    mode: Literal["absolute", "tighten"] = "absolute",
) -> dict:
    """Calibrate σ² of V measurements on generating and slack nodes.

    Voltage transducers on generator buses (110-750 kV) are far more accurate
    than on passive load buses (σ_V of tenths of a kV against several kV).
    Without this step the default variance leaves the V measurement a weak
    anchor; on 750 kV clusters of nuclear plants that gave a systematic
    voltage sag of a few percent.

    Targets:

    * active nodes with ``generation_p_max`` set after
      :func:`aggregate_generators_to_node`. The ±9999 sentinel means "no
      data" (:func:`gridstate.bounds.is_sentinel`), not generation, so such
      nodes are skipped;
    * slack nodes (``node_type == NodeType.SLACK``).

    Only measurements of ``quality == GOOD`` are touched. A measurement an
    earlier stage marked QUESTIONABLE (telemetry quality, the voltage range
    filter) keeps its inflated variance: calibrating it would turn a suspect
    reading into the strongest anchor of the estimate.

    Modes:

    * ``"absolute"`` (default): the variance is **set** to ``sigma2``
      regardless of its current value, and the weight to ``1 / sigma2``. This
      is a tightening only where the current variance is larger than
      ``sigma2``. On low-voltage generator buses (6-16 kV) the telemetry
      variance is already smaller (``max(1 % · Vn, 0.1)² ≈ 0.01-0.025``), so
      ``"absolute"`` *loosens* those measurements, by up to a factor of 10,
      and it discards any variance scaling applied by earlier stages.
    * ``"tighten"``: the variance becomes ``min(current, sigma2)``, i.e. it is
      never loosened and a measurement already tighter than ``sigma2`` is left
      untouched (variance and weight). The weight of a tightened measurement
      is ``1 / sigma2``.

    Args:
        model: ``Working`` after aggregate_generators_to_node
            and apply_voltage_range_filter.
        sigma2: target σ² of the V measurement, in the units of the model's
            V measurements. Default 0.1 (σ ≈ 0.32 kV).
        mode: ``"absolute"`` or ``"tighten"``, see above.

    Returns:
        ``{"updated_meas": N, "target_nodes": N}``; in ``"tighten"`` mode also
        ``"kept_tighter": N`` -- targeted GOOD measurements whose variance was
        already at or below ``sigma2`` and was left as is.

    Raises:
        ValueError: if ``mode`` is not one of the two listed.
    """
    if mode not in ("absolute", "tighten"):
        raise ValueError(f"mode must be 'absolute' or 'tighten', got {mode!r}")
    meas_arr = table_copy(model.measurements)
    nodes_arr = model.nodes.to_numpy()
    stats = _voltage_meas_calibration_on_arrays(
        meas_arr,
        nodes_arr,
        slack_type=int(NodeType.SLACK),
        ot_node=int(MeasurementObjectType.NODE),
        mt_v=int(MeasurementType.VOLTAGE),
        sigma2=sigma2,
        mode=mode,
    )
    table_replace(model.measurements, meas_arr)
    return stats


def _voltage_meas_calibration_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    *,
    slack_type: int,
    ot_node: int,
    mt_v: int,
    sigma2: float,
    mode: Literal["absolute", "tighten"] = "absolute",
) -> dict:
    """Core of :func:`apply_voltage_meas_calibration_for_gen_nodes` on arrays.

    Targets are active nodes with a non-zero, non-sentinel
    ``generation_p_max`` or ``node_type == SLACK`` (enum values as ints).
    Mutates ``variance``/``weight`` of GOOD measurements in ``meas_arr`` in
    place; reads ``nodes_arr``. ``mode`` is validated by the public wrapper.
    """
    has_gen_max = "generation_p_max" in (nodes_arr.dtype.names or ())
    target_ids: set[int] = set()
    for n in nodes_arr:
        if not bool(n["status"]):
            continue
        if int(n["node_type"]) == slack_type:
            target_ids.add(int(n["id"]))
            continue
        if has_gen_max:
            p_max = float(n["generation_p_max"])
            if p_max != 0.0 and not is_sentinel(p_max):
                target_ids.add(int(n["id"]))

    tighten = mode == "tighten"
    n_updated = 0
    n_kept = 0
    for i in np.flatnonzero(meas_mask(meas_arr, kinds=mt_v, obj=ot_node)).tolist():
        if int(meas_arr[i]["object_id"]) not in target_ids:
            continue
        if int(meas_arr[i]["quality"]) != int(MeasurementQuality.GOOD):
            continue
        new_var = gen_v_calibrated_variance(float(meas_arr[i]["variance"]), sigma2, tighten=tighten)
        if new_var is None:
            n_kept += 1
            continue
        meas_arr[i]["variance"] = new_var
        meas_arr[i]["weight"] = 1.0 / float(sigma2) if sigma2 > 0 else 0.0
        n_updated += 1

    stats = {"updated_meas": n_updated, "target_nodes": len(target_ids)}
    if tighten:
        stats["kept_tighter"] = n_kept
    return stats
