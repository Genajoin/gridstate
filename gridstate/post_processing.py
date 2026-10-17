"""Post-processing of SE results — write-back into ``model.measurements``.

After the WLS converges, the ``z`` vector (measurements in p.u.) and the
estimated values ``h(x)`` are known, but in ``Measurement`` the
``estimated_si`` and ``residual`` fields remain zero. This module fills them
in the original units (MW / MVAr / kV / A) for further analytics (bad-data
detection, reports, plugins).

The fields are filled for **active** measurements that made it into
``meas_index`` (i.e. were used by the WLS). Measurements outside ``meas_index``
get ``NaN`` — an explicit "not estimated" mark.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate._arrays import meas_mask
from gridstate.algebra.base import (
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    SIDE_FROM,
    SIDE_TO,
    BaseAlgebra,
)
from gridstate.bounds import is_reactive_only_generation, resolve_bounds
from gridstate.load_characteristics import QUANTITY_P, QUANTITY_Q, segment_factor, segment_plan
from gridstate.measurement_ids import ANTI_OVERSHOOT_ID_BASE, IdAllocator
from gridstate.preprocessing.meas_rows import pseudo_node_measurement
from gridstate.resolve_pass import (
    REVERT_REASON_NOT_CONVERGED,
    REVERT_REASON_OUT_OF_RANGE,
    ResolvePass,
    state_range_violation,
)
from gridstate.units import BASE_MVA, NetworkPU
from gridstate.utils import id_to_pos_map
from gridstate.working import table_copy, table_replace


if TYPE_CHECKING:
    from gridstate.result import SEResult
    from gridstate.working import Working, _ArrayCollection
    from gridstate.z_vector import MeasurementIndex


def _pu_to_si(
    values_pu: np.ndarray, meas_index: MeasurementIndex, network_pu: NetworkPU
) -> np.ndarray:
    """Convert values of the ``z`` rows from p.u. to the original units.

    * V (kind=2): p.u. × ``voltage_nominal`` → kV.
    * P/Q (kind=0/1/4/5): p.u. × ``BASE_MVA`` → MW/MVAr.
    * I (kind=3): p.u. × ``i_base`` (A), where
      ``i_base = BASE_MVA·1000/(√3·vn)`` for the corresponding branch side.
    """
    out = np.array(values_pu, dtype=np.float64, copy=True)
    if out.size == 0:
        return out
    kind = meas_index.kind
    op = meas_index.object_pos
    side = meas_index.branch_side

    # V at a node
    m = (kind == KIND_VOLTAGE) & (meas_index.object_kind == OBJ_NODE)
    if m.any():
        out[m] = values_pu[m] * network_pu.bus_vn_kv[op[m]]

    # P/Q injections and P/Q branch flows
    m_pq = (
        (kind == KIND_POWER_P)
        | (kind == KIND_POWER_Q)
        | (kind == KIND_POWER_INJECTION_P)
        | (kind == KIND_POWER_INJECTION_Q)
    )
    if m_pq.any():
        out[m_pq] = values_pu[m_pq] * BASE_MVA

    # Branch current — the side determines vn for i_base
    m_i = (kind == KIND_CURRENT) & (meas_index.object_kind == OBJ_BRANCH)
    if m_i.any():
        sqrt3 = float(np.sqrt(3.0))
        # vn of the corresponding end node
        from_idx = network_pu.from_idx[op[m_i]]
        to_idx = network_pu.to_idx[op[m_i]]
        vn_from = network_pu.bus_vn_kv[from_idx]
        vn_to = network_pu.bus_vn_kv[to_idx]
        side_arr = side[m_i]
        vn_used = np.where(
            side_arr == SIDE_TO, vn_to, np.where(side_arr == SIDE_FROM, vn_from, vn_from)
        )
        i_base = BASE_MVA * 1000.0 / (sqrt3 * vn_used)
        out[m_i] = values_pu[m_i] * i_base

    return out


def write_measurement_estimates(
    *,
    model: Working,
    measurements: _ArrayCollection,
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    meas_index: MeasurementIndex,
    z: np.ndarray,
) -> dict[str, int]:
    """Fill ``estimated_si``/``residual`` in measurements.

    Args:
        model: ``Working``, updated in place.
        measurements: the collection used in SE (the same reference from ``model``).
        v_pu: voltage magnitudes from the SE solution, p.u.
        delta_rad: voltage angles, rad.
        network_pu: internal p.u. representation.
        ybus, yf, yt: admittance matrices, needed to recompute h(x).
        meas_index: metadata of the ``z`` rows.
        z: measurement vector in p.u. (for residual = z − h).

    Fills (only for measurements that made it into meas_index):
        * ``estimated_si`` — h(x) in the original units;
        * ``residual`` — value (the original measurement) − estimated_si.

    Measurements OUTSIDE ``meas_index`` (inactive / not used by the WLS) get
    ``estimated_si = residual = NaN`` — explicit "not estimated" semantics
    instead of a silent zero from the zero-filled collection. Consumers of
    estimates/residuals (UI, promote, analytics) must be NaN-aware. The internal
    bad-data/χ² code computes residuals from the live ``z − h(x)`` vector in
    ``quality_summary`` / ``validation.bad_data``, NOT from this column, so NaN
    does not affect it.

    Returns:
        ``{"updated": N, "missing": N}``.
    """
    base = BaseAlgebra(
        ybus=ybus,
        yf=yf,
        yt=yt,
        meas_index=meas_index,
        layout=None,  # type: ignore[arg-type]  # evaluate_h does not use layout
        network_pu=network_pu,
    )
    h_pu = base.evaluate_h(v_pu, delta_rad)

    # h(x) and z in the original units.
    h_si = _pu_to_si(h_pu, meas_index, network_pu)

    meas_id_arr = meas_index.meas_id

    # Vectorised write through the backing structured array (a bit-for-bit
    # replacement of the by_id dict + two per-row proxy loops; removes 2.2M
    # __getattr__ calls on large models). First mark ALL measurements as
    # "not estimated" (NaN), then overwrite the estimated ones by id — without
    # the NaN fill, measurements outside meas_index would carry 0 from the
    # zero-filled collection, indistinguishable from a zero estimate.
    arr = table_copy(measurements)
    arr["estimated_si"] = np.nan
    arr["residual"] = np.nan
    id_to_row = id_to_pos_map(arr["id"])  # duplicate ids -> last wins
    rows = np.fromiter(
        (id_to_row.get(int(m), -1) for m in meas_id_arr.tolist()),
        dtype=np.int64,
        count=len(meas_id_arr),
    )
    valid = rows >= 0
    vr = rows[valid]
    # Duplicate id → last-write-wins (as the former dict by_id): fancy-assign with
    # repeated indices takes the last write in meas_index order.
    est = arr["estimated_si"]
    res = arr["residual"]
    est[vr] = h_si[valid]
    # residual in the original units: value (original) − estimated; f8−f8.
    res[vr] = arr["value"][vr] - h_si[valid]
    table_replace(measurements, arr)

    return {"updated": int(valid.sum()), "missing": int((~valid).sum())}


def write_part_estimates(measurements: _ArrayCollection, estimates: dict[int, float]) -> None:
    """Overwrite ``estimated_si`` / ``residual`` of the part rows of node injections.

    ``estimates`` maps a measurement id to the estimate of its part (MW / MVAr;
    ``nan`` — not estimated on its own). Written after
    :func:`write_measurement_estimates`, which evaluates every injection row as a
    net injection.
    """
    arr = table_copy(measurements)
    row_of = id_to_pos_map(arr["id"])
    for mid, value in estimates.items():
        i = row_of.get(int(mid))
        if i is None:
            continue
        arr["estimated_si"][i] = value
        arr["residual"][i] = arr["value"][i] - value
    table_replace(measurements, arr)


def write_node_estimates(
    model: Working,
    *,
    node_ids: np.ndarray,
    load_p: np.ndarray | None = None,
    load_q: np.ndarray | None = None,
    generation_p: np.ndarray | None = None,
    generation_q: np.ndarray | None = None,
) -> dict[str, int]:
    """Write load/generation estimates after IPM-SE into the node table.

    Fills ``load_p_estimated``, ``load_q_estimated``,
    ``generation_p_estimated``, ``generation_q_estimated`` for the nodes in
    ``node_ids``. The semantics are the actual values at the current V
    (the voltage-dependent load and generation of the solved state); a
    conversion to nominal values (if PF needs it) is done separately through
    ``load_model_id`` + ``load_characteristics``.

    ``None`` for any of the value arrays → the corresponding field is not
    updated. For nodes not in ``node_ids`` the fields stay as they were
    (0.0 by default).

    Args:
        model: ``Working``, updated in place.
        node_ids: (N,) — IDs of the nodes the estimates are written for.
        load_p: (N,) MW or ``None``.
        load_q: (N,) MVAr or ``None``.
        generation_p: (N,) MW or ``None``.
        generation_q: (N,) MVAr or ``None``.

    Returns:
        ``{"updated": N, "missing": M}`` — how many nodes were written, how many
        IDs were not found in the model.
    """
    arrays = {
        "load_p_estimated": load_p,
        "load_q_estimated": load_q,
        "generation_p_estimated": generation_p,
        "generation_q_estimated": generation_q,
    }
    n = len(node_ids)
    for name, arr in arrays.items():
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} has length {len(arr)}, expected {n} (as node_ids)")

    updated = 0
    missing = 0
    for i, nid in enumerate(np.asarray(node_ids).tolist()):
        update: dict[str, float] = {}
        for name, arr in arrays.items():
            if arr is not None:
                update[name] = float(arr[i])
        if not update:
            continue
        try:
            model.nodes.update(int(nid), update)
            updated += 1
        except (KeyError, ValueError):
            missing += 1

    return {"updated": updated, "missing": missing}


def _clip(value: float, lo: float, hi: float) -> float:
    """Safe clip that handles an "inverted" pair (lo > hi → return value)."""
    if not np.isfinite(lo) and not np.isfinite(hi):
        return value
    if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
        # Invalid bounds — do not clip, otherwise we would end up with lo>hi.
        return value
    if np.isfinite(hi) and value > hi:
        return hi
    if np.isfinite(lo) and value < lo:
        return lo
    return value


def write_node_estimates_from_inj(model: Working) -> dict[str, int]:
    """Split ``p_inj_calc``/``q_inj_calc`` into
    ``load_*_estimated`` / ``generation_*_estimated`` for the WLS mode.

    After a WLS solve we only have ``p_inj_calc``/``q_inj_calc``
    (written in ``write_results_to_model``). This post-pass fills the
    load and generation estimates based on the ``exist_load`` /
    ``exist_gen`` flags and the physical bounds ``*_min`` / ``*_max``. For IPM
    this is already done through the box vars (``write_node_estimates``), so it
    does not need to be called additionally.

    Bounds are interpreted through :func:`gridstate.bounds.resolve_bounds`:
    an unfilled ``(0, 0)`` pair and ±9999 sentinels = "unset" →
    the estimate is not clipped (previously (0, 0) zeroed the estimate — nodes
    without declared limits lost all their load/generation).

    The split logic (for each active node):

    * **transit** (``exist_load=0`` AND ``exist_gen=0``) →
      ``load_*_estimated`` = ``generation_*_estimated`` = 0.
    * **gen-only** (``exist_gen=1``, ``exist_load=0``) →
      ``generation_*_estimated`` = ``p_inj_calc`` (q_inj_calc for Q),
      clipped to ``[generation_*_min, generation_*_max]``;
      ``load_*_estimated`` = 0.
    * **load-only** (``exist_load=1``, ``exist_gen=0``) →
      ``load_*_estimated`` = ``-p_inj_calc`` (load is consumption),
      clipped to ``[load_*_min, load_*_max]``;
      ``generation_*_estimated`` = 0.
    * **both** (``exist_load=1`` AND ``exist_gen=1``) — the load is FIXED
      at its nominal (``cur_pn=load_p`` from the input data) and the injection
      difference is attributed to generation: for ``p_inj≥0`` (generation
      surplus) ``gen = p_inj + cur_pn``, ``load = cur_pn``; for ``p_inj<0``
      (load surplus) ``gen = cur_pg``, ``load = cur_pg − p_inj``.
      If the node is not tied to the input data (``cur_pg=cur_pn=0``) — split
      by the sign of the injection (``gen=max(p_inj,0)``, ``load=max(−p_inj,0)``).
      The Q branch is analogous. Clipped to the corresponding bounds. This is a
      rough heuristic: the exact gen/load split on a mixed node from a single
      injection is underdetermined — without separate pg/pn measurements it
      yields only a plausible, not a unique, solution.

    Returns the counters ``{"updated": ..., "missing": ...,
    "transit": ..., "gen_only": ..., "load_only": ..., "both": ...,
    "clipped": ...}`` for diagnostics.
    """
    nodes_arr = model.nodes.to_numpy()
    if len(nodes_arr) == 0:
        return {
            "updated": 0,
            "missing": 0,
            "transit": 0,
            "gen_only": 0,
            "load_only": 0,
            "both": 0,
            "clipped": 0,
        }

    updated = 0
    missing = 0
    transit = 0
    gen_only = 0
    load_only = 0
    both = 0
    clipped = 0

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        nid = int(row["id"])
        exist_load = bool(row["exist_load"])
        exist_gen = bool(row["exist_gen"])
        p_inj = float(row["p_inj_calc"])
        q_inj = float(row["q_inj_calc"])

        # Unset/sentinel bounds → ±inf ("do not clip"). Without this the
        # (0, 0) pair — the numpy default of an unfilled field — zeroed
        # the estimate, while ±9999 on load_q practically never occurred,
        # yet garbage sums of generator limits clipped Q to junk.
        load_p_min, load_p_max = resolve_bounds(row["load_p_min"], row["load_p_max"])
        load_q_min, load_q_max = resolve_bounds(row["load_q_min"], row["load_q_max"])
        gen_p_min, gen_p_max = resolve_bounds(row["generation_p_min"], row["generation_p_max"])
        gen_q_min, gen_q_max = resolve_bounds(row["generation_q_min"], row["generation_q_max"])

        load_p_est = 0.0
        load_q_est = 0.0
        gen_p_est = 0.0
        gen_q_est = 0.0

        if not exist_load and not exist_gen:
            transit += 1
            # all zeros — already initialised.
        elif exist_gen and not exist_load:
            gen_only += 1
            gen_p_raw = p_inj
            gen_q_raw = q_inj
            gen_p_est = _clip(gen_p_raw, gen_p_min, gen_p_max)
            gen_q_est = _clip(gen_q_raw, gen_q_min, gen_q_max)
            if gen_p_est != gen_p_raw or gen_q_est != gen_q_raw:
                clipped += 1
        elif exist_load and not exist_gen:
            load_only += 1
            # Convention: load_p is consumption, p_inj = gen - load → load = -p_inj.
            load_p_raw = -p_inj
            load_q_raw = -q_inj
            load_p_est = _clip(load_p_raw, load_p_min, load_p_max)
            load_q_est = _clip(load_q_raw, load_q_min, load_q_max)
            if load_p_est != load_p_raw or load_q_est != load_q_raw:
                clipped += 1
        else:
            both += 1
            # Keep the generation share in (gen + |load|), as in the node input data.
            cur_pg = float(row["generation_p"])
            cur_pn = float(row["load_p"])
            denom_p = cur_pg + abs(cur_pn)
            cur_qg = float(row["generation_q"])
            cur_qn = float(row["load_q"])
            denom_q = abs(cur_qg) + abs(cur_qn)

            # p_inj = gen_p - load_p is UNDERDETERMINED for the (gen, load) pair:
            # one injection, two unknowns. A proportional split
            # gen_p_raw = frac_pg·S_p is singular at frac_pg≈0.5
            # (S_p = p_inj/(2·frac_pg−1)), so a robust heuristic is used:
            # FIX the nominal load (cur_pn) and attribute the injection
            # difference to generation (for p_inj≥0 — generation surplus
            # over the load; for p_inj<0 — the opposite, gen is fixed).
            if p_inj >= 0.0:
                gen_p_raw = p_inj + cur_pn  # gen covers inj+load
                load_p_raw = cur_pn
            else:
                gen_p_raw = cur_pg
                load_p_raw = cur_pg - p_inj  # |p_inj| is the load surplus over gen
            if q_inj >= 0.0:
                gen_q_raw = q_inj + cur_qn
                load_q_raw = cur_qn
            else:
                gen_q_raw = cur_qg
                load_q_raw = cur_qg - q_inj

            # Fallback when the node is not tied to the input data (cur_pg=cur_pn=0): there is no
            # nominal anchor → split strictly by the sign of the injection
            # (gen=max(p_inj,0), load=max(−p_inj,0)).
            if denom_p < 1e-9:
                gen_p_raw = max(p_inj, 0.0)
                load_p_raw = max(-p_inj, 0.0)
            if denom_q < 1e-9:
                gen_q_raw = max(q_inj, 0.0)
                load_q_raw = max(-q_inj, 0.0)

            gen_p_est = _clip(gen_p_raw, gen_p_min, gen_p_max)
            gen_q_est = _clip(gen_q_raw, gen_q_min, gen_q_max)
            load_p_est = _clip(load_p_raw, load_p_min, load_p_max)
            load_q_est = _clip(load_q_raw, load_q_min, load_q_max)
            if (
                gen_p_est != gen_p_raw
                or gen_q_est != gen_q_raw
                or load_p_est != load_p_raw
                or load_q_est != load_q_raw
            ):
                clipped += 1

        try:
            model.nodes.update(
                nid,
                {
                    "load_p_estimated": load_p_est,
                    "load_q_estimated": load_q_est,
                    "generation_p_estimated": gen_p_est,
                    "generation_q_estimated": gen_q_est,
                },
            )
            updated += 1
        except (KeyError, ValueError):
            missing += 1

    return {
        "updated": updated,
        "missing": missing,
        "transit": transit,
        "gen_only": gen_only,
        "load_only": load_only,
        "both": both,
        "clipped": clipped,
    }


def reconcile_node_balance(
    model: Working, *, respect_bounds: bool = False
) -> dict[str, float | int]:
    """Close the nodal imbalance of the estimates: ``gen_est − load_est ≡ *_inj_calc``.

    A soft IPM (and the WLS split with clips / load-characteristic overlay)
    leaves a residual imbalance ``r = inj_calc − (gen_est − load_est)`` at the nodes —
    the load/generation estimates are NOT KCL-consistent with the state voltages.
    That is acceptable for analytics, but the SE output as an operating POINT
    (a PF input, promotion into the input data) must be balanced: the classical SE
    convention is state = V/δ, injections = Sbus(V), and gen/load are their split.
    This post-pass finalises the split by draining the residual according to the
    node markup:

    * ``exist_gen=1, exist_load=0`` → into generation (``gen += r``);
    * otherwise (load nodes, mixed, transits) → into load
      (``load −= r``; a transit node gets a pseudo-load ``−inj``, the usual
      way to materialise an area's load on its nodes).

    By default (``respect_bounds=False``) there are no clips to the bounds:
    consistency accuracy matters more, after the pass ``gen_est − load_est ==
    inj_calc`` element-wise, and a standalone PF from the promoted estimates
    reproduces the SE operating point.

    ``respect_bounds=True`` — the target value is clipped to the physical
    node bounds (``load_*_min/max`` through :func:`resolve_bounds`;
    unset pairs are not clipped). Rationale: a consistent operating point
    never violates ``pn_min`` (negative loads are legal only within the
    limits), while an unconditional split of the soft balance residual r
    fabricates non-physical negative loads on nodes
    with ``lo=0`` (the IPM box var cannot go negative — the negative is created by
    reconcile itself). Cost: on clipped nodes ``gen−load ≠ inj_calc`` —
    the unclosed residual accumulates in ``sum_unclosed_p_mw`` (diagnostics).

    Only solved nodes are processed (``solved=1`` — they received
    ``*_inj_calc`` in ``write_results_to_model``). The ``imbalance_p/q``
    columns are not touched — they are computed against the input gen/load.

    Returns:
        Counters: ``updated``, ``to_gen``/``to_load`` (nodes per rule
        branch), ``sum_abs_p_mw``/``max_abs_p_mw`` — the scale of the closed
        P imbalance (diagnostics of "how soft" the IPM was); with
        ``respect_bounds`` additionally ``clipped`` and
        ``sum_unclosed_p_mw``.
    """
    nodes_arr = model.nodes.to_numpy()
    out: dict[str, float | int] = {
        "updated": 0,
        "to_gen": 0,
        "to_load": 0,
        "sum_abs_p_mw": 0.0,
        "max_abs_p_mw": 0.0,
        "clipped": 0,
        "sum_unclosed_p_mw": 0.0,
    }
    if len(nodes_arr) == 0:
        return out

    eps = 1e-9  # MW/MVAr: below this is numerical noise, the node is left alone
    for row in nodes_arr:
        if not bool(row["status"]) or not bool(row["solved"]):
            continue
        r_p = float(row["p_inj_calc"]) - (
            float(row["generation_p_estimated"]) - float(row["load_p_estimated"])
        )
        r_q = float(row["q_inj_calc"]) - (
            float(row["generation_q_estimated"]) - float(row["load_q_estimated"])
        )
        if abs(r_p) <= eps and abs(r_q) <= eps:
            continue
        update: dict[str, float]
        if bool(row["exist_gen"]) and not bool(row["exist_load"]):
            out["to_gen"] = int(out["to_gen"]) + 1
            update = {
                "generation_p_estimated": float(row["generation_p_estimated"]) + r_p,
                "generation_q_estimated": float(row["generation_q_estimated"]) + r_q,
            }
            if respect_bounds:
                gp_lo, gp_hi = resolve_bounds(row["generation_p_min"], row["generation_p_max"])
                if is_reactive_only_generation(
                    row["generation_p_min"],
                    row["generation_p_max"],
                    row["generation_q_min"],
                    row["generation_q_max"],
                ):
                    # Compensator: no active power, the [0, 0] pair is data.
                    gp_lo, gp_hi = 0.0, 0.0
                gq_lo, gq_hi = resolve_bounds(row["generation_q_min"], row["generation_q_max"])
                gp = _clip(update["generation_p_estimated"], gp_lo, gp_hi)
                gq = _clip(update["generation_q_estimated"], gq_lo, gq_hi)
                if gp != update["generation_p_estimated"] or gq != update["generation_q_estimated"]:
                    out["clipped"] = int(out["clipped"]) + 1
                    out["sum_unclosed_p_mw"] = float(out["sum_unclosed_p_mw"]) + abs(
                        update["generation_p_estimated"] - gp
                    )
                    update["generation_p_estimated"] = gp
                    update["generation_q_estimated"] = gq
        else:
            out["to_load"] = int(out["to_load"]) + 1
            update = {
                "load_p_estimated": float(row["load_p_estimated"]) - r_p,
                "load_q_estimated": float(row["load_q_estimated"]) - r_q,
            }
            if respect_bounds:
                if not bool(row["exist_load"]):
                    # Transit (exist_load=0 in the else branch): there is no load on the node —
                    # draining the residual here would be fabricating a pseudo-load.
                    lp, lq = 0.0, 0.0
                else:
                    lp_lo, lp_hi = resolve_bounds(row["load_p_min"], row["load_p_max"])
                    lq_lo, lq_hi = resolve_bounds(row["load_q_min"], row["load_q_max"])
                    lp = _clip(update["load_p_estimated"], lp_lo, lp_hi)
                    lq = _clip(update["load_q_estimated"], lq_lo, lq_hi)
                if lp != update["load_p_estimated"] or lq != update["load_q_estimated"]:
                    out["clipped"] = int(out["clipped"]) + 1
                    out["sum_unclosed_p_mw"] = float(out["sum_unclosed_p_mw"]) + abs(
                        update["load_p_estimated"] - lp
                    )
                    update["load_p_estimated"] = lp
                    update["load_q_estimated"] = lq
        model.nodes.update(int(row["id"]), update)
        out["updated"] = int(out["updated"]) + 1
        out["sum_abs_p_mw"] = float(out["sum_abs_p_mw"]) + abs(r_p)
        out["max_abs_p_mw"] = max(float(out["max_abs_p_mw"]), abs(r_p))

    return out


def apply_load_characteristic(model: Working) -> dict[str, int]:
    """Recompute ``load_*_estimated`` through the static load characteristics
    P(V) / Q(V) after the gen/load split.

    After the WLS pass ``write_node_estimates_from_inj`` has written
    ``load_p_estimated`` / ``load_q_estimated`` as the actual values at the
    current V. For nodes bound to a load characteristic
    (``node.load_model_id >= 0``) a more physical estimate scales the
    **nominal** load (``node.load_p`` / ``node.load_q``, the value supplied with
    the input, at ``V_pu = 1.0``) by a polynomial in V:

    .. code-block:: text

        load_p_estimated = load_p_nominal × (a0 + a1·V_pu + a2·V_pu²)
        load_q_estimated = load_q_nominal × (b0 + b1·V_pu + b2·V_pu²)

    where ``V_pu = voltage_magnitude / voltage_nominal`` (after SE).
    The coefficients come from the typed ``load_characteristics`` table
    (a node refers to a row via the 0-based index ``load_model_id``, ``-1`` =
    no binding). For PQ-const models (a0=1, a1=a2=0, b0=1, b1=b2=0) the recomputation
    yields exactly ``load_*_nominal``.

    A characteristic with rows in ``load_characteristic_segments`` for P (resp.
    Q) is piecewise: the segment containing ``V_pu`` (a value on a boundary
    belongs to the lower segment) replaces the quadratic P (resp. Q) row, with
    a polynomial up to ``V_pu⁴`` (see :mod:`gridstate.load_characteristics`).
    A piecewise P (Q) is applied only where the nominal ``load_p`` (``load_q``)
    is positive; elsewhere the estimate is kept. The quadratic rows keep their
    previous behaviour.

    Applied **only in the WLS mode** (see ``gridstate/api.py::estimate``)
    after ``write_node_estimates_from_inj``. The IPM box vars already account for
    V at the final state — no repeated recomputation is needed.

    Args:
        model: ``Working``, updated in place.

    Returns:
        Counters:
            * ``updated`` — number of nodes whose
              ``load_p_estimated`` (and/or ``load_q_estimated``) was recomputed.
            * ``skipped_no_sxn`` — active nodes with ``exist_load=True`` but
              ``load_model_id < 0`` (no binding to a load characteristic).
            * ``skipped_no_load`` — active nodes with ``load_model_id >= 0`` but
              ``exist_load=False`` (no load — nothing to recompute).
            * ``skipped_bad_sxn`` — ``load_model_id`` points outside the
              ``load_characteristics`` table (a dangling reference).
            * ``no_load_models`` — 1 if the model has no
              ``load_characteristics`` table (then zeros are returned).
            * ``piecewise_p`` / ``piecewise_q`` — loads recomputed through a
              piecewise P / Q characteristic.

    Raises:
        ValueError: ``load_characteristic_segments`` violates the contract
            (unknown ``load_model_id``, bad ``quantity``, non-increasing ``v_lo``).
    """
    out = {
        "updated": 0,
        "skipped_no_sxn": 0,
        "skipped_no_load": 0,
        "skipped_bad_sxn": 0,
        "no_load_models": 0,
        "piecewise_p": 0,
        "piecewise_q": 0,
    }

    # The source of the load-characteristic coefficients is the ``load_characteristics`` table; nodes carry
    # a 0-based reference ``load_model_id`` (``-1`` = none).
    nodes_arr = model.nodes.to_numpy()
    lc_coll = getattr(model, "load_characteristics", None)
    coeff = lc_coll.to_numpy() if lc_coll is not None else None
    if coeff is None or len(coeff) == 0:
        out["no_load_models"] = 1
        return out

    n_lm = len(coeff)
    a0 = np.asarray(coeff["coeff_p_a0"], dtype=np.float64)
    a1 = np.asarray(coeff["coeff_p_a1"], dtype=np.float64)
    a2 = np.asarray(coeff["coeff_p_a2"], dtype=np.float64)
    b0 = np.asarray(coeff["coeff_q_b0"], dtype=np.float64)
    b1 = np.asarray(coeff["coeff_q_b1"], dtype=np.float64)
    b2 = np.asarray(coeff["coeff_q_b2"], dtype=np.float64)
    seg_coll = getattr(model, "load_characteristic_segments", None)
    seg_arr = seg_coll.to_numpy() if seg_coll is not None else None
    plan = segment_plan(seg_arr, n_lm) if seg_arr is not None else {}

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        exist_load = bool(row["exist_load"])

        # idx — 0-based coefficient row from load_model_id (-1=none). The order
        # of the skip checks does not affect the recomputation (no skip branch
        # recomputes) — only the stat counters.
        idx = int(row["load_model_id"])
        if idx < 0:
            if exist_load:
                out["skipped_no_sxn"] += 1
            continue

        if not exist_load:
            out["skipped_no_load"] += 1
            continue
        if idx < 0 or idx >= n_lm:
            out["skipped_bad_sxn"] += 1
            continue

        vn = float(row["voltage_nominal"])
        vm = float(row["voltage_magnitude"])
        if vn <= 0.0 or vm <= 0.0:
            # Without a valid V we cannot scale — leave as is.
            continue
        v_pu = vm / vn

        load_p_nom = float(row["load_p"])
        load_q_nom = float(row["load_q"])

        factor_p = float(a0[idx] + a1[idx] * v_pu + a2[idx] * v_pu * v_pu)
        factor_q = float(b0[idx] + b1[idx] * v_pu + b2[idx] * v_pu * v_pu)

        load_p_est = load_p_nom * factor_p
        load_q_est = load_q_nom * factor_q
        seg_p = plan.get((idx, QUANTITY_P))
        if seg_p is not None:
            if load_p_nom > 0.0:
                load_p_est = load_p_nom * segment_factor(*seg_p, v_pu)
                out["piecewise_p"] += 1
            else:
                load_p_est = float(row["load_p_estimated"])
        seg_q = plan.get((idx, QUANTITY_Q))
        if seg_q is not None:
            if load_q_nom > 0.0:
                load_q_est = load_q_nom * segment_factor(*seg_q, v_pu)
                out["piecewise_q"] += 1
            else:
                load_q_est = float(row["load_q_estimated"])

        try:
            model.nodes.update(
                int(row["id"]),
                {
                    "load_p_estimated": load_p_est,
                    "load_q_estimated": load_q_est,
                },
            )
            out["updated"] += 1
        except (KeyError, ValueError):
            # ID not found — theoretically impossible, we iterate over this same table.
            pass

    return out


def _max_voltage_ratio(model: Working) -> float:
    """max(voltage_magnitude / voltage_nominal) over active nodes (vn>0)."""
    nd = model.nodes.to_numpy()
    mask = nd["status"] & (nd["voltage_nominal"] > 0)
    if not np.any(mask):
        return 0.0
    return float(np.max(nd["voltage_magnitude"][mask] / nd["voltage_nominal"][mask]))


def refine_anti_overshoot(
    model: Working,
    result: SEResult,
    resolve: Callable[[], SEResult],
    *,
    ceiling: float = 1.15,
    inj_sigma: float = 2.0,
    max_iters: int = 5,
    mid_start: int = ANTI_OVERSHOOT_ID_BASE,
) -> tuple[SEResult, dict[str, int | float | bool | str]]:
    """Anti-overshoot post-solve V refinement with SELF-VALIDATION (revert).

    Weakly observable radial nodes (no real V measurement, a loose Q pseudo) can get
    an unphysical V overshoot (up to 1.3+ pu): a high-X branch without telemetry drives
    a phantom circulating reactive power. The step finds active nodes with
    ``V/Vnom > ceiling`` and WITHOUT a real (non-pseudo) V measurement, adds tight
    P/Q injection priors on them = the materialised value (``generation−load``, σ=``inj_sigma`` MVA) and
    re-solves (warm) through ``resolve()``. This removes the phantom Q → V settles at the
    physically consistent value (``V_from/tap``) rather than at an arbitrary ceiling.

    SELF-VALIDATION: the refinement is accepted ONLY if ``max(V/Vnom)`` dropped
    (a runtime criterion, no outside solution needed). Otherwise V/δ are
    **rolled back** to the base solution.
    This guarantees no regression: where the overshoot is real and removable — a gain;
    where the refinement destabilises — a rollback to the base (no-op). Universally safe.

    Args:
        model: the model AFTER ``estimate`` (with V/δ in ``voltage_magnitude/angle``).
        result: the current ``SEResult`` (returned on rollback).
        resolve: callable ``() -> SEResult`` — re-solve SE on ``model`` (warm,
            ``init="results"``). Decoupling from ``gridstate.api.estimate`` (no cycle).
        ceiling: overshoot threshold (pu); nodes above it are candidates.
        inj_sigma: σ (MVA) of the tight injection prior. Small → pins the injection ≈ 0 hard.
        max_iters: max outer passes (each catches new overshoot nodes).
            The loop stops after the first pass that did not lower
            ``max(V/Vnom)`` relative to the base.
        mid_start: starting id for the added pseudo-measurements.

    The refinement is accepted only if the last re-solve converged
    (``refined.success``) and left the state in range
    (:func:`~gridstate.resolve_pass.state_range_violation`; an out-of-range
    state also ends the rounds, since no further solve can start from it); a
    rejected re-solve is reverted even when it lowered ``max(V/Vnom)``, and
    the stats then carry ``revert_reason``.

    Returns:
        ``(result, stats)`` — the final ``SEResult`` (refined, or base on rollback),
        ``stats = {"tightened": N, "accepted": bool, "max_ratio_before/after": ...}``
        (+ ``revert_reason`` when the re-solve did not converge or left the
        state out of range, + ``state_violation`` in the latter case).
    """
    # Snapshots for a FULL rollback: nodes (V/δ/estimated), branches (flows) and
    # measurements (the added tight priors + estimated_si). Otherwise a revert would leave
    # the measurements and branch/measurement estimates of the rejected solution in
    # the model — inconsistent with the base V/δ (which would break both the post-hoc quality
    # summary and the bad-data re-pass that read estimated_si).
    # The pass is armed eagerly: the real-V set is read from the snapshot, and
    # the acceptance criterion is its own (revert() instead of settle()).
    rp = ResolvePass(model, result)
    rp.arm()
    maxr0 = _max_voltage_ratio(model)

    me = rp.snapshot.measurements
    real_v: set[int] = {
        int(x)
        for x in me["object_id"][
            meas_mask(me, kinds=KIND_VOLTAGE, obj=OBJ_NODE, pseudo=False)
        ].tolist()
    }

    tightened: set[int] = set()
    # Ids are checked against the measurements at the start; the rows added
    # below come from this allocator only.
    ids = IdAllocator(me["id"], mid_start, name="anti_overshoot")
    var = float(inj_sigma) * float(inj_sigma)
    refined = result
    violation: str | None = None
    for _ in range(max_iters):
        nd = model.nodes.to_numpy()
        newly: list[tuple[int, float, float]] = []
        for i in range(len(nd)):
            nid = int(nd[i]["id"])
            vn = float(nd[i]["voltage_nominal"])
            if not bool(nd[i]["status"]) or vn <= 0 or nid in real_v or nid in tightened:
                continue
            if float(nd[i]["voltage_magnitude"]) / vn > ceiling:
                pinj = float(nd[i]["generation_p"]) - float(nd[i]["load_p"])
                qinj = float(nd[i]["generation_q"]) - float(nd[i]["load_q"])
                newly.append((nid, pinj, qinj))
        if not newly:
            break
        for nid, pinj, qinj in newly:
            for kind, val in (
                (KIND_POWER_INJECTION_P, pinj),
                (KIND_POWER_INJECTION_Q, qinj),
            ):
                model.measurements.add(pseudo_node_measurement(ids.next(), nid, kind, val, var))
            tightened.add(nid)
        refined = resolve()
        violation = state_range_violation(model.nodes.to_numpy())
        if violation is not None:
            break
        # The round did not push max V/Vnom below the base — further rounds
        # would add priors to an already destabilised solution; the result
        # would be rolled back anyway unless it becomes better than the base.
        if _max_voltage_ratio(model) >= maxr0 - 1e-4:
            break

    maxr1 = _max_voltage_ratio(model)
    # A non-converged re-solve is never accepted: the V it stopped at says
    # nothing about the overshoot being resolved.
    converged = not tightened or bool(refined.success)
    accepted = bool(tightened) and converged and violation is None and maxr1 < maxr0 - 1e-4
    stats: dict[str, int | float | bool | str] = {
        "tightened": len(tightened),
        "accepted": accepted,
        "max_ratio_before": round(maxr0, 4),
        "max_ratio_after": round(maxr1, 4),
    }
    if violation is not None:
        stats["revert_reason"] = REVERT_REASON_OUT_OF_RANGE
        stats["state_violation"] = violation
    elif not converged:
        stats["revert_reason"] = REVERT_REASON_NOT_CONVERGED
    if accepted:
        return refined, stats
    # Full rollback to the base: V/δ + branch flows + measurements (the added
    # tight priors are removed, estimated_si returns to the base solution).
    return rp.revert(), stats
