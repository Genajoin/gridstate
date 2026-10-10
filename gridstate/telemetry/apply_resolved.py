"""Application of precomputed numeric telemetry/materialization plans to the regime.

Contract cores of the z-vector (``_apply_telemetry_on_arrays``) and of regime
materialization (``_materialize_area_on_arrays``) plus thin adapters that apply
ready-made numeric plans (``apply_telemetry_resolved`` /
``apply_materialize_resolved``). The plans are computed outside the core by the
data producer; this module only applies them to ``SE_INPUT`` arrays (no source
format layer). The measurement kind maps (``_KIND_MAP`` / ``_NODE_INJ_MAP`` /
``_INJ_MT``) live in ``gridstate.telemetry._specs``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import id_value_map
from gridstate.constants import FilterFlag, MeasurementObjectType, MeasurementType
from gridstate.measurement_ids import IdAllocator
from gridstate.telemetry._filters import build_measurement_index
from gridstate.telemetry._specs import _INJ_MT, _KIND_MAP, _NODE_INJ_MAP
from gridstate.telemetry.loss_filter import is_branch_q_consistent_with_physics
from gridstate.telemetry.noise import (
    QUESTIONABLE_SIGMA2_MULTIPLIER,
    branch_nominal_kv,
    variance_branch_q,
    variance_injection,
    variance_power,
    variance_voltage,
)
from gridstate.telemetry.quality import QUALITY_BAD, QUALITY_QUESTIONABLE, aggregate_qualities
from gridstate.working import table_copy, table_replace
from gridstate.z_vector import SIDE_NONE


if TYPE_CHECKING:
    from gridstate.working import Working


_OT_NODE = int(MeasurementObjectType.NODE)
_OT_BRANCH = int(MeasurementObjectType.BRANCH)
_MT_Q = int(MeasurementType.POWER_Q)
_MT_V = int(MeasurementType.VOLTAGE)


@dataclass(frozen=True)
class TelemetryApplyConfig:
    """Tunable thresholds/factors for :func:`_apply_telemetry_on_arrays`.

    Single source of truth for the defaults that were previously duplicated
    between the public wrapper :func:`apply_telemetry_resolved` and the core.
    The wrapper keeps its individual keyword arguments for backward
    compatibility and assembles this config to hand down.
    """

    questionable_sigma2_multiplier: float = QUESTIONABLE_SIGMA2_MULTIPLIER
    v_sigma2_scale_by_node: dict[int, float] | None = None
    flow_sigma2_scale_by_branch: dict[tuple[int, str], float] | None = None
    branch_p_sigma_frac: float = 0.02
    branch_q_sigma_frac: float = 0.07
    branch_q_sigma_charging_alpha: float = 0.10
    sign_inconsistency_threshold_mw: float | None = 100.0
    q_loss_filter_enabled: bool = True
    q_loss_filter_floor_mvar: float = 50.0
    q_loss_filter_rel_pct: float = 30.0
    q_loss_filter_action: str = "downweight"
    q_loss_filter_downweight_factor: float = 100.0


def apply_telemetry_resolved(
    model: Working,
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    arg_keys: list[tuple[int, str]],
    *,
    total_args: int,
    questionable_sigma2_multiplier: float = QUESTIONABLE_SIGMA2_MULTIPLIER,
    v_sigma2_scale_by_node: dict[int, float] | None = None,
    flow_sigma2_scale_by_branch: dict[tuple[int, str], float] | None = None,
    branch_p_sigma_frac: float = 0.02,
    branch_q_sigma_frac: float = 0.07,
    branch_q_sigma_charging_alpha: float = 0.10,
    sign_inconsistency_threshold_mw: float | None = 100.0,
    q_loss_filter_enabled: bool = True,
    q_loss_filter_floor_mvar: float = 50.0,
    q_loss_filter_rel_pct: float = 30.0,
    q_loss_filter_action: str = "downweight",
    q_loss_filter_downweight_factor: float = 100.0,
) -> dict[str, int]:
    """Apply a ready ``resolved`` z-vector to ``measurements``.

    A pure application (no source format, snapshot or formulas): snapshot of
    ``measurements``/``nodes``/``branches`` → kernel
    :func:`_apply_telemetry_on_arrays` → write-back + ``.add()`` of NODE
    injections. Called by the ``run()`` step at its own position (after
    ``apply_topology``/``apply_rpn``: the core reads branch ``status``/``susceptance``).
    """
    config = TelemetryApplyConfig(
        questionable_sigma2_multiplier=questionable_sigma2_multiplier,
        v_sigma2_scale_by_node=v_sigma2_scale_by_node,
        flow_sigma2_scale_by_branch=flow_sigma2_scale_by_branch,
        branch_p_sigma_frac=branch_p_sigma_frac,
        branch_q_sigma_frac=branch_q_sigma_frac,
        branch_q_sigma_charging_alpha=branch_q_sigma_charging_alpha,
        sign_inconsistency_threshold_mw=sign_inconsistency_threshold_mw,
        q_loss_filter_enabled=q_loss_filter_enabled,
        q_loss_filter_floor_mvar=q_loss_filter_floor_mvar,
        q_loss_filter_rel_pct=q_loss_filter_rel_pct,
        q_loss_filter_action=q_loss_filter_action,
        q_loss_filter_downweight_factor=q_loss_filter_downweight_factor,
    )
    meas_arr = table_copy(model.measurements)
    nodes_arr = model.nodes.to_numpy()
    branches_arr = model.branches.to_numpy()

    stats, new_rows = _apply_telemetry_on_arrays(
        meas_arr,
        nodes_arr,
        branches_arr,
        arg_keys,
        resolved,
        total_args=total_args,
        config=config,
    )

    table_replace(model.measurements, meas_arr)
    # Batch insert in a single concatenation (per-row .add() = O(n²)).
    model.measurements.add_many(new_rows)

    return stats


def _detect_sign_inconsistent_branches(
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    threshold_mw: float | None,
) -> set[int]:
    """Pre-pass: detect branches with sign-inconsistent P measurements
    (PBEG and PEND both flowing into the line, i.e. of the same sign after
    accounting for invert) -- this indicates a measurement bound to the wrong
    branch or stale telemetry. The branch measurements of such branches are
    not activated.
    """
    inconsistent_branches: set[int] = set()
    if threshold_mw is None:
        return inconsistent_branches
    for obj_id, kind in arg_keys:
        if kind != "PBEG":
            continue
        if (obj_id, "PEND") not in resolved:
            continue
        v_beg = resolved[(obj_id, "PBEG")][0]
        v_end = resolved[(obj_id, "PEND")][0]
        if v_beg is None or v_end is None:
            continue
        # PBEG/PEND values arrive with the sign already applied (the inversion
        # is encoded in the input numeric plan by the adapter). On a physically
        # consistent line ``v_beg ≈ -v_end + losses`` holds, i.e. they have
        # **opposite signs**. If |v_beg + v_end| exceeds the threshold, it is
        # either a sign anomaly or a wrong binding.
        if abs(v_beg + v_end) >= threshold_mw:
            inconsistent_branches.add(int(obj_id))
    return inconsistent_branches


def _detect_q_inconsistent_branches(
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    branch_vn: dict[int, float],
    branch_b: dict[int, float],
    branch_x: dict[int, float],
    config: TelemetryApplyConfig,
) -> tuple[set[int], set[int]]:
    """Pre-pass for Q: physical consistency of |Q_BEG + Q_END| with the branch pi model.

    With ``q_loss_filter_enabled`` (default) the observed |Q_beg + Q_end| is
    compared with the expectation from V_flat = V_nom and
    ``branch.{susceptance, reactance}``; an outlier is a deviation above
    ``max(floor, rel_pct/100 * expected)``. See
    :func:`gridstate.telemetry.loss_filter.is_branch_q_consistent_with_physics`.
    Branches without a nominal voltage are not checked.

    Returns ``(q_inconsistent_branches, q_inconsistent_hv_branches)``: branches
    to drop and branches to downweight (by ``q_loss_filter_action``).
    """
    q_inconsistent_branches: set[int] = set()  # branches to drop
    q_inconsistent_hv_branches: set[int] = set()  # branches to downweight

    if not config.q_loss_filter_enabled:
        return q_inconsistent_branches, q_inconsistent_hv_branches

    for obj_id, kind in arg_keys:
        if kind != "QBEG":
            continue
        if (obj_id, "QEND") not in resolved:
            continue
        v_beg = resolved[(obj_id, "QBEG")][0]
        v_end = resolved[(obj_id, "QEND")][0]
        if v_beg is None or v_end is None:
            continue
        vn = branch_vn.get(int(obj_id), 0.0)
        if vn <= 0:
            continue
        # Take P_typical from PBEG/PEND to estimate the X-loss.
        p_typical = 0.0
        for p_kind in ("PBEG", "PEND"):
            if (obj_id, p_kind) not in resolved:
                continue
            pv = resolved[(obj_id, p_kind)][0]
            if pv is not None:
                p_typical = max(p_typical, abs(pv))
        is_ok, _, _, _ = is_branch_q_consistent_with_physics(
            q_observed_mvar=v_beg + v_end,
            vn_kv=vn,
            susceptance_si=branch_b.get(int(obj_id), 0.0),
            reactance_si=branch_x.get(int(obj_id), 0.0),
            p_typical_mw=p_typical,
            floor_mvar=config.q_loss_filter_floor_mvar,
            rel_pct=config.q_loss_filter_rel_pct,
        )
        if not is_ok:
            if config.q_loss_filter_action == "downweight":
                q_inconsistent_hv_branches.add(int(obj_id))
            else:
                q_inconsistent_branches.add(int(obj_id))
    return q_inconsistent_branches, q_inconsistent_hv_branches


def _measurement_variance(
    meas_arr: np.ndarray,
    idx: int,
    *,
    mt: int,
    ot: int,
    obj_id: int,
    vn_by_node: dict[int, float],
    branch_vn: dict[int, float],
    branch_b: dict[int, float],
    config: TelemetryApplyConfig,
) -> float:
    """Variance for the just-written measurement, by measurement type.

    V uses the node nominal voltage; branch Q is charging-aware; branch P
    scales with magnitude.
    """
    # variance: V -- by the node vn, P/Q -- by magnitude
    value = float(meas_arr[idx]["value"])
    if mt == _MT_V:  # NODE V
        obj_node = int(meas_arr[idx]["object_id"])
        vn = vn_by_node.get(obj_node, 220.0)
        return variance_voltage(value, vn)
    if mt == _MT_Q:  # BRANCH Q
        # Charging-aware σ_Q (single source: variance_branch_q): on
        # long HV/EHV lines with a large B the legitimate Q ≈ V²·B; a static
        # σ_frac underestimates the noise. σ_min = α·|B|·Vn² (default α=0.10), so Q readings
        # of 750 kV lines get σ ≈ 50–100 Mvar. A parameter sweep confirmed
        # α=0.10 as the best value on four large production models.
        vn_q = branch_vn.get(int(obj_id), 0.0) if ot == _OT_BRANCH else 0.0
        b_si_q = branch_b.get(int(obj_id), 0.0) if ot == _OT_BRANCH else 0.0
        return variance_branch_q(
            value,
            charging_mvar=abs(b_si_q) * vn_q * vn_q,
            charging_alpha=config.branch_q_sigma_charging_alpha,
            sigma_frac=config.branch_q_sigma_frac,
        )
    # BRANCH P (mt=0)
    return variance_power(value, sigma_frac=config.branch_p_sigma_frac)


def _apply_measurement_loop(
    meas_arr: np.ndarray,
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    meas_idx: dict[tuple[int, int, int, int], int],
    vn_by_node: dict[int, float],
    branch_status_by_id: dict[int, bool],
    branch_vn: dict[int, float],
    branch_b: dict[int, float],
    inconsistent_branches: set[int],
    q_inconsistent_branches: set[int],
    q_inconsistent_hv_branches: set[int],
    config: TelemetryApplyConfig,
    stats: dict[str, int],
    inj_acc: dict[tuple[int, str], list[tuple[float, str, int]]],
) -> None:
    """Main application loop over ``arg_keys``.

    Mutates ``meas_arr`` (write-back of activated measurements), ``stats``
    (counters) and ``inj_acc`` (accumulated NODE injections) in place.
    """
    for obj_id, kind in arg_keys:
        # NODE injection (PG/PN/QG/QN) or GENERATOR (PG_G<n>, QG_G<n>) ->
        # accumulate for the later .add(). PG_G* and QG_G* count as a
        # positive contribution of P_gen/Q_gen to the node (like a plain PG/QG).
        kind_base = kind.split("_G", 1)[0] if kind.startswith(("PG_G", "QG_G")) else kind
        if kind_base in _NODE_INJ_MAP:
            value, n_res, guid_first, q = resolved[(obj_id, kind)]
            if value is None:
                if n_res == 0:
                    stats["skipped_no_value"] += 1
                else:
                    stats["skipped_formula_error"] += 1
                continue
            if q == QUALITY_BAD:
                stats["skipped_bad_quality"] += 1
                continue
            pq, mult = _NODE_INJ_MAP[kind_base]
            val = value * mult
            inj_acc.setdefault((obj_id, pq), []).append((val, guid_first, q))
            continue

        if kind not in _KIND_MAP:
            stats["skipped_kind_unsupported"] += 1
            continue
        # A branch with sign-inconsistent P-meas: none of the 4 measurements of
        # this LINE-OBJ_ID are activated (the data is broken).
        if kind in ("PBEG", "PEND", "QBEG", "QEND") and obj_id in inconsistent_branches:
            stats.setdefault("skipped_sign_inconsistent", 0)
            stats["skipped_sign_inconsistent"] += 1
            ot_f, mt_f, side_f, _ = _KIND_MAP[kind]
            idx_f = meas_idx.get((ot_f, mt_f, side_f, obj_id))
            if idx_f is not None:
                meas_arr[idx_f]["filter_flag"] = int(FilterFlag.P_SIGN_INCONSISTENCY)
            continue
        # Q-inconsistent branches: skip only the Q measurements, keep P.
        if kind in ("QBEG", "QEND") and obj_id in q_inconsistent_branches:
            stats.setdefault("skipped_q_inconsistent", 0)
            stats["skipped_q_inconsistent"] += 1
            ot_f, mt_f, side_f, _ = _KIND_MAP[kind]
            idx_f = meas_idx.get((ot_f, mt_f, side_f, obj_id))
            if idx_f is not None:
                meas_arr[idx_f]["filter_flag"] = int(FilterFlag.Q_INCONSISTENCY)
            continue
        # Q-inconsistent on an HV line with action=downweight: do NOT drop; the
        # marker for the σ² ×factor increase is applied after the value is
        # activated (logic right below the val/var computation).
        ot, mt, side, sign = _KIND_MAP[kind]
        # Do not activate a measurement on a disconnected branch -- otherwise
        # WLS gets a P/Q residual ≠0 on a branch the Y-bus has zeroed.
        if ot == _OT_BRANCH and not branch_status_by_id.get(int(obj_id), False):
            stats.setdefault("skipped_branch_off", 0)
            stats["skipped_branch_off"] += 1
            continue
        key = (ot, mt, side, obj_id)
        idx = meas_idx.get(key)
        if idx is None:
            stats["skipped_no_meas"] += 1
            continue
        value, n_res, guid_first, q = resolved[(obj_id, kind)]
        if value is None:
            if n_res == 0:
                stats["skipped_no_value"] += 1
            else:
                stats["skipped_formula_error"] += 1
            continue
        if q == QUALITY_BAD:
            stats["skipped_bad_quality"] += 1
            meas_arr[idx]["filter_flag"] = int(FilterFlag.BAD_QUALITY)
            continue
        val = value * sign
        # V-measurement < 50% Vnom is a "sensor not working" / disconnected
        # equipment sentinel. It must not be accepted as a hard anchor: SE would pull
        # V→0 at the node and around it (a non-physical minimum of J). It is not
        # activated; add_pseudo_measurements adds pseudo-V=Vnom instead.
        if mt == _MT_V and ot == _OT_NODE:
            obj_node = int(meas_arr[idx]["object_id"])
            vn = vn_by_node.get(obj_node, 220.0)
            if vn > 0 and abs(val) < 0.5 * vn:
                stats["skipped_v_below_half_nominal"] += 1
                meas_arr[idx]["filter_flag"] = int(FilterFlag.V_BELOW_HALF_NOMINAL)
                continue
        if meas_arr[idx]["status"]:
            meas_arr[idx]["value"] = float(meas_arr[idx]["value"]) + val
            # Worst-case accumulation of quality on repeated hits.
            meas_arr[idx]["quality"] = max(int(meas_arr[idx]["quality"]), q)
        else:
            meas_arr[idx]["value"] = val
            meas_arr[idx]["status"] = True
            meas_arr[idx]["source_guid"] = guid_first
            meas_arr[idx]["quality"] = q
        var = _measurement_variance(
            meas_arr,
            idx,
            mt=mt,
            ot=ot,
            obj_id=obj_id,
            vn_by_node=vn_by_node,
            branch_vn=branch_vn,
            branch_b=branch_b,
            config=config,
        )
        if int(meas_arr[idx]["quality"]) == QUALITY_QUESTIONABLE:
            var *= config.questionable_sigma2_multiplier
        # Per-measurement σ² scale of a node V measurement from the data
        # producer's plan ({node_id: factor}; factor<1 increases trust in the
        # measurement -- an "anchor" sensor, factor>1 reduces it). Node V only
        # (mt=2, ot=0).
        if mt == _MT_V and ot == _OT_NODE and config.v_sigma2_scale_by_node:
            scale = config.v_sigma2_scale_by_node.get(int(meas_arr[idx]["object_id"]))
            if scale is not None:
                var *= float(scale)
                stats.setdefault("v_sigma2_scaled", 0)
                stats["v_sigma2_scaled"] += 1
        # A similar plan for branch flow measurements: {(branch_id, kind): factor},
        # kind ∈ PBEG/PEND/QBEG/QEND. factor>1 reduces trust (a measurement that is
        # a defect candidate per the producer's data), factor<1 increases it.
        if ot == _OT_BRANCH and config.flow_sigma2_scale_by_branch:
            scale = config.flow_sigma2_scale_by_branch.get((int(obj_id), kind))
            if scale is not None:
                var *= float(scale)
                stats.setdefault("flow_sigma2_scaled", 0)
                stats["flow_sigma2_scaled"] += 1
        # Q-inconsistent branch (physical filter, action="downweight"):
        # downweight σ² × factor (a loose anchor: the reading stays with a low weight).
        # Applied after the default σ is computed.
        if kind in ("QBEG", "QEND") and obj_id in q_inconsistent_hv_branches:
            var *= config.q_loss_filter_downweight_factor
            meas_arr[idx]["filter_flag"] = int(FilterFlag.Q_INCONSISTENCY)  # q_inconsistency mark
        meas_arr[idx]["variance"] = float(var)
        meas_arr[idx]["weight"] = 1.0 / float(var)
        if int(meas_arr[idx]["quality"]) == QUALITY_QUESTIONABLE:
            stats["applied_questionable"] += 1
        else:
            stats["applied"] += 1


def _build_node_injection_rows(
    inj_acc: dict[tuple[int, str], list[tuple[float, str, int]]],
    meas_arr: np.ndarray,
    config: TelemetryApplyConfig,
    stats: dict[str, int],
) -> list[dict]:
    """Build NODE injections for the later .add() by the adapter.

    The variance is :func:`gridstate.telemetry.noise.variance_injection` (sigma
    5 % of the net value, see there), scaled for a QUESTIONABLE aggregate. The ids
    are assigned here (bit-identical to the former .add()).
    """
    new_rows: list[dict] = []
    ids = IdAllocator.after_max(meas_arr["id"], name="apply_telemetry")
    for (obj_id, pq), entries in inj_acc.items():
        net = sum(v for v, _, _ in entries)
        guid_first = entries[0][1]
        net_q = aggregate_qualities([q for _, _, q in entries])
        mt = _INJ_MT[pq]
        var = variance_injection(net, "P" if pq == "P" else "Q")
        if net_q == QUALITY_QUESTIONABLE:
            var *= config.questionable_sigma2_multiplier
        new_rows.append(
            {
                "id": ids.next(),
                "object_type": _OT_NODE,
                "object_id": int(obj_id),
                "measurement_type": mt,
                "branch_side": SIDE_NONE,
                "value": float(net),
                "variance": float(var),
                "weight": 1.0 / float(var),
                "status": True,
                "quality": int(net_q),
                "source_guid": guid_first,
            }
        )
        stats["node_inj_added"] += 1
        if net_q == QUALITY_QUESTIONABLE:
            stats["applied_questionable"] += 1

    return new_rows


def _apply_telemetry_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    arg_keys: list[tuple[int, str]],
    resolved: dict[tuple[int, str], tuple[float | None, int, str, int]],
    *,
    total_args: int,
    config: TelemetryApplyConfig,
) -> tuple[dict[str, int], list[dict]]:
    """CORE: telemetry application over contract numpy arrays.

    Pure work on ``SE_INPUT.measurements`` (``meas_arr``, mutated in place) /
    ``SE_INPUT.nodes`` (``nodes_arr``) / ``SE_INPUT.branches``
    (``branches_arr``). NO external dependencies, NO source formulas: all numeric
    measurement values were already computed by the adapter and passed in
    ``resolved`` -- ``{(obj_id, kind): (value|None, n_resolved, guid_first, quality)}``.
    The iteration order is given by ``arg_keys`` (= ``list(args.keys())``).

    Returns ``(stats, new_rows)``: ``new_rows`` is a list of dicts for the
    later ``model.measurements.add()`` (NODE injections PG/PN/QG/QN); their
    ``id`` values are assigned here so as to be bit-identical to the former
    in-loop ``.add()`` block.

    Tuning arrives via ``config`` (:class:`TelemetryApplyConfig`).
    """
    # Index measurements: (ot, mt, side, oid) → row idx in numpy array
    meas_idx = build_measurement_index(meas_arr)

    # node_id -> voltage_nominal for the sigma_V computation.
    vn_by_node: dict[int, float] = id_value_map(nodes_arr, "voltage_nominal")
    # branch_id -> status. A measurement must not be placed on a disconnected
    # branch: WLS would get a P/Q residual ≠0 on a branch the Y-bus zeroes out,
    # and the iteration would compensate through V/δ of neighbouring nodes and
    # spoil the solution. Checked when activating a branch measurement (ot=1).
    branch_status_by_id: dict[int, bool] = id_value_map(branches_arr, "status", bool)

    # First disable ALL measurements (on input they arrive with status=True by
    # default). After apply, only those covered by the snapshot get status=True.
    meas_arr["status"] = False

    stats = {
        "applied": 0,
        "applied_questionable": 0,
        "skipped_no_value": 0,
        "skipped_no_meas": 0,
        "skipped_bad_quality": 0,
        "total_args": total_args,
        "skipped_kind_unsupported": 0,
        "skipped_formula_error": 0,
        "skipped_v_below_half_nominal": 0,
        "node_inj_added": 0,
    }

    # Accumulator for NODE injections:
    #   (obj_id, "P"|"Q") → list of (signed_val, guid, quality).
    # Net P_inj = sum(values), quality = max(qualities) (worst-case).
    inj_acc: dict[tuple[int, str], list[tuple[float, str, int]]] = {}

    # Per-branch nominal voltage / susceptance / reactance: feed the Q-physics
    # pre-pass and the charging-aware σ_Q. Vn is taken from the branch's from-node.
    branch_vn = branch_nominal_kv(vn_by_node, branches_arr, policy="from")
    branch_b: dict[int, float] = id_value_map(branches_arr, "susceptance")
    branch_x: dict[int, float] = id_value_map(branches_arr, "reactance")

    inconsistent_branches = _detect_sign_inconsistent_branches(
        arg_keys, resolved, config.sign_inconsistency_threshold_mw
    )
    if config.sign_inconsistency_threshold_mw is not None:
        stats["sign_inconsistent_branches"] = len(inconsistent_branches)

    q_inconsistent_branches, q_inconsistent_hv_branches = _detect_q_inconsistent_branches(
        arg_keys, resolved, branch_vn, branch_b, branch_x, config
    )
    if config.q_loss_filter_enabled:
        stats["q_inconsistent_branches"] = len(q_inconsistent_branches)
        stats["q_inconsistent_hv_branches"] = len(q_inconsistent_hv_branches)

    _apply_measurement_loop(
        meas_arr,
        arg_keys,
        resolved,
        meas_idx,
        vn_by_node,
        branch_status_by_id,
        branch_vn,
        branch_b,
        inconsistent_branches,
        q_inconsistent_branches,
        q_inconsistent_hv_branches,
        config,
        stats,
        inj_acc,
    )

    new_rows = _build_node_injection_rows(inj_acc, meas_arr, config, stats)

    return stats, new_rows


def _materialize_area_on_arrays(
    nodes_arr: np.ndarray,
    obs: dict[int, float],
    *,
    max_col: str,
    min_col: str,
    exist_col: str,
    set_col: str,
    max_cap: float,
    skip_neg_min: float,
    global_k_fallback: float,
    fill: bool = True,
) -> dict[str, float]:
    """CORE: materialization of one node field over a ``SE_INPUT`` array.

    Contract float math: reads ``status``/``id``/``area_id``/
    ``<max_col>``/``<min_col>``/``<exist_col>`` from ``nodes_arr`` (the structured
    ``SE_INPUT.nodes.input_dtype()``) and mutates ``nodes_arr[<set_col>]`` in place.
    The observed regime is resolved outside the core -- a ready numeric ``obs``
    (``{node_id -> value}``) arrives here, like ``resolved_taps`` in apply_rpn.

    * **observed** (``nid in obs``) -> ``nodes_arr[set_col][i] = obs[nid]`` verbatim;
    * **unobserved** with ``<exist_col>`` -> (if ``fill``) ``clamp(k_area·<max_col>)``,
      where ``k_area = median(obs/<max_col>)`` over the observed non-sentinel nodes
      of the area (``area_id``); fallback ``global_k_fallback``.

    Sentinel sanitization (for k calibration and fill): ``0 < max ≤ max_cap`` and
    ``min ≥ skip_neg_min``.

    ``fill=False`` -> verbatim only (for generation: the area spread ``k·gen_max``
    is bimodal and unreliable -- a generator is either dispatched or not; an A/B
    comparison on a regional model showed WLS dV_max worsening 0.25 -> 0.43).
    Verbatim is sufficient given rich PG telemetry.

    Returns:
        ``{"n_obs", "n_fill", "sum"}`` (``sum`` rounded to 0.1; the node values
        feeding the solver are written per node and do not depend on the
        iteration order).
    """
    obs_ids = set(obs)
    status = nodes_arr["status"]
    ids = nodes_arr["id"]
    areas = nodes_arr["area_id"]
    maxv = nodes_arr[max_col]
    minv = nodes_arr[min_col]
    exist = nodes_arr[exist_col]
    setv = nodes_arr[set_col]
    n = len(nodes_arr)

    def _ok(wmax: float, wmin: float) -> bool:
        return 0.0 < wmax <= max_cap and wmin >= skip_neg_min

    k_area: dict[int, float] = {}
    glob = global_k_fallback
    if fill:
        rat: dict[int, list[float]] = defaultdict(list)
        for i in range(n):
            if not bool(status[i]):
                continue
            nid = int(ids[i])
            if nid not in obs_ids:
                continue
            wmax = float(maxv[i])
            wmin = float(minv[i])
            if _ok(wmax, wmin):
                rat[int(areas[i])].append(obs[nid] / wmax)
        k_area = {a: float(median(v)) for a, v in rat.items() if v}
        all_rat = [r for v in rat.values() for r in v]
        glob = float(median(all_rat)) if all_rat else global_k_fallback

    cnt = {"n_obs": 0, "n_fill": 0, "sum": 0.0}
    for i in range(n):
        if not bool(status[i]):
            continue
        nid = int(ids[i])
        if nid in obs_ids:
            setv[i] = float(obs[nid])
            cnt["n_obs"] += 1
            cnt["sum"] += float(obs[nid])
        elif fill and int(exist[i]):
            wmax = float(maxv[i])
            wmin = float(minv[i])
            if _ok(wmax, wmin):
                a = int(areas[i])
                v = min(max(k_area.get(a, glob) * wmax, wmin), wmax)
                setv[i] = float(v)
                cnt["n_fill"] += 1
                cnt["sum"] += v
    cnt["sum"] = round(cnt["sum"], 1)
    return cnt


def apply_materialize_resolved(
    model: Working,
    obs: dict[str, dict[int, float]],
    *,
    fill_q: bool = True,
    materialize_generation: bool = True,
    materialize_generation_fill: bool = False,
    max_load: float = 5e4,
    max_gen: float = 5e4,
    skip_neg_min: float = -100.0,
    global_k_fallback: float = 0.40,
    gen_global_k_fallback: float = 0.80,
) -> dict[str, dict[str, float]]:
    """Apply a ready observed node regime ``obs`` to the model nodes.

    Pure application (no snapshot/formulas): ``obs`` (``{"load_p"/"load_q"/"gen_p"/
    "gen_q" -> {node_id -> value}}``, a ready numeric plan of observed injections)
    -> contract cores :func:`_materialize_area_on_arrays` -> write-back. Called by
    the ``run()`` step at its own position (after ``aggregate_generators``: the 4
    fields are written per node, generation_* overrides the aggregation -- the
    order is preserved).

    Contract node snapshot: the 4 fields are materialized on ONE array (each call
    has its own ``set_col`` -- load_p/load_q/generation_p/generation_q; the k
    calibration reads only the max/min/exist columns, which are not mutated, so
    the calls are independent), followed by a single write-back.
    """
    nodes_arr = model.nodes.to_numpy().copy()
    out: dict[str, dict[str, float]] = {}
    out["load_p"] = _materialize_area_on_arrays(
        nodes_arr,
        obs["load_p"],
        max_col="load_p_max",
        min_col="load_p_min",
        exist_col="exist_load",
        set_col="load_p",
        max_cap=max_load,
        skip_neg_min=skip_neg_min,
        global_k_fallback=global_k_fallback,
    )
    if fill_q:
        out["load_q"] = _materialize_area_on_arrays(
            nodes_arr,
            obs["load_q"],
            max_col="load_q_max",
            min_col="load_q_min",
            exist_col="exist_load",
            set_col="load_q",
            max_cap=max_load,
            skip_neg_min=skip_neg_min,
            global_k_fallback=global_k_fallback,
        )
    if materialize_generation:
        out["gen_p"] = _materialize_area_on_arrays(
            nodes_arr,
            obs["gen_p"],
            max_col="generation_p_max",
            min_col="generation_p_min",
            exist_col="exist_gen",
            set_col="generation_p",
            max_cap=max_gen,
            skip_neg_min=-1e18,  # generation_*_min<0 is normal, not a boundary sentinel
            global_k_fallback=gen_global_k_fallback,
            fill=materialize_generation_fill,
        )
        if fill_q:
            out["gen_q"] = _materialize_area_on_arrays(
                nodes_arr,
                obs["gen_q"],
                max_col="generation_q_max",
                min_col="generation_q_min",
                exist_col="exist_gen",
                set_col="generation_q",
                max_cap=max_gen,
                skip_neg_min=-1e18,
                global_k_fallback=gen_global_k_fallback,
                fill=materialize_generation_fill,
            )
    model.nodes.update_from_array(nodes_arr)
    return out
