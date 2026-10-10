"""Setup assembly for the SE IPM mode.

Helper functions for:

1. Extracting box bounds from ``NODE_DTYPE`` (``load_p_min/max``,
   ``generation_p_min/max``, etc.) of active nodes with ``exist_load=1``
   or ``exist_gen=1``;
2. Extending ``StateLayout`` for the IPM mode with these box sections;
3. Augmenting ``z`` / ``R`` / ``MeasurementIndex`` with node
   balance equations (``KIND_NODE_BALANCE_P/Q``) for all active
   nodes, linking V/δ to the box variables through the physics.

All functions are pure (they do not modify ``model``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.bounds import declared_box, is_reactive_only_generation, resolve_bounds
from gridstate.state import StateLayout
from gridstate.units import BASE_MVA
from gridstate.utils import id_to_pos_map
from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_BOX_PRIOR_QGEN,
    KIND_BOX_PRIOR_QNAG,
    KIND_NODE_BALANCE_P,
    KIND_NODE_BALANCE_Q,
    OBJ_NODE,
    SIDE_NONE,
    MeasurementIndex,
)


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working


__all__ = [
    "IPMSetup",
    "build_ipm_setup",
]


@dataclass
class _BoxSection:
    """Accumulator + column spec for one box-var family (pgen/qgen/pnag/qnag).

    ``exist_col`` gates which nodes contribute (``exist_gen`` for the gen families,
    ``exist_load`` for the load families). ``prior_kind`` is the ``KIND_BOX_PRIOR_*`` used when emitting prior-meas.
    The list fields accumulate one entry per contributing node, in node order.
    """

    exist_col: str
    min_col: str
    max_col: str
    init_col: str
    prior_kind: int
    pos: list[int] = field(default_factory=list)
    lo: list[float] = field(default_factory=list)
    hi: list[float] = field(default_factory=list)
    init: list[float] = field(default_factory=list)
    prior_s2: list[float] = field(default_factory=list)


def _collect_box_sections(
    nodes_arr: np.ndarray,
    bus_id_to_pos: dict[int, int],
    *,
    default_box_halfwidth_pu: float,
    prior_sigma2_normal_pu: float,
    prior_sigma2_bus_equiv_pu: float,
    bus_equiv_width_threshold_pu: float,
) -> list[_BoxSection]:
    """Collect box-var bounds/init/prior per section for exist_load/exist_gen nodes.

    Returns the four sections in canonical order ``[pgen, qgen, pnag, qnag]``. Each
    section's lists are filled in ``nodes_arr`` order, which is the ordering every
    downstream concatenation depends on.
    """

    def _resolve_prior_sigma2(
        bounds: tuple[float, float],
        *,
        is_default_box: bool = False,
    ) -> float:
        """σ²_prior for a box variable.

        * wide box (width > bus_equiv_threshold) -> bus_equiv_pu
          (tight, keeps the solver from spreading the residual over a fictitious
          5-10 GW equivalent); NOT applied to default boxes
          (``is_default_box=True``): they are wide not because the node is a
          BUS equivalent but because no bounds are set in the data;
        * ordinary box -> normal_pu (default 0: no prior is created,
          balance/data measurements pull the value).
        """
        width = abs(bounds[1] - bounds[0])
        if not is_default_box and width > float(bus_equiv_width_threshold_pu):
            return float(prior_sigma2_bus_equiv_pu)
        return float(prior_sigma2_normal_pu)

    def _bound_pair(
        lo_raw: float, hi_raw: float, default_halfwidth_pu: float
    ) -> tuple[float, float, bool]:
        """``(lo, hi, is_default)`` in p.u.; unset bounds -> a wide default.

        "Unset" (both ~0 / sentinels ±9999 / a degenerate or
        inverted pair) is interpreted per :mod:`gridstate.bounds` and
        replaced by a symmetric box ``±default_halfwidth_pu``.
        A half-set pair (one sentinel) keeps the valid side.

        Previously unset pairs gave ``None`` -> the box variable was **not
        created** while the node's balance equation remained,
        pinning its injection to zero as for a transit node. For Q this is a
        mass case (Q limits are rarely filled): the reactive output of
        generator nodes was forcibly zeroed.
        """
        lo_res, hi_res = resolve_bounds(lo_raw, hi_raw)
        lo_pu = lo_res / BASE_MVA if np.isfinite(lo_res) else -float(default_halfwidth_pu)
        hi_pu = hi_res / BASE_MVA if np.isfinite(hi_res) else float(default_halfwidth_pu)
        is_default = not (np.isfinite(lo_res) and np.isfinite(hi_res))
        if hi_pu - lo_pu < 1e-9:
            # A degenerate valid pair (lo==hi) is a narrow "nail" the barrier
            # would not survive; widen symmetrically to the default.
            lo_pu -= float(default_halfwidth_pu)
            hi_pu += float(default_halfwidth_pu)
            is_default = True
        return lo_pu, hi_pu, is_default

    def _init_in_box(value_pu: float, bounds: tuple[float, float]) -> float:
        """Init box-var: the current value from the node table clamped to [lo, hi]
        with a micro-margin from the bounds.

        If ``value_pu==0``, init=0 (if 0 ∈ [lo+margin, hi-margin]),
        otherwise the nearest bound with margin. This is a soft prior towards 0
        for nodes without an explicit pn₀/pg₀ in the input: the injection
        starts at 0 and pn/pg are estimated from the branch telemetry.
        Without it IPM started from the box midpoint (e.g. 42 500 MW for a
        wide BUS equivalent) and spread the mismatch over it.
        """
        lo, hi = bounds
        width = hi - lo
        # Micro-margin from the bounds: 1% of width or 1e-6 p.u.
        margin = max(width * 0.01, 1e-6)
        if abs(value_pu) < 1e-9:
            # Start from 0 if 0 ∈ box; otherwise the nearest bound with margin.
            if lo + margin > 0.0:
                return lo + margin
            if hi - margin < 0.0:
                return hi - margin
            return 0.0
        if value_pu < lo + margin:
            return lo + margin
        if value_pu > hi - margin:
            return hi - margin
        return value_pu

    def _is_reactive_only(row: np.void) -> bool:
        """Generating node with reactive power only: ``P ∈ [0, 0]``,
        ``P = 0`` and a set, non-degenerate Q range."""
        return float(row["generation_p"]) == 0.0 and is_reactive_only_generation(
            row["generation_p_min"],
            row["generation_p_max"],
            row["generation_q_min"],
            row["generation_q_max"],
        )

    sections = [
        _BoxSection(
            "exist_gen",
            "generation_p_min",
            "generation_p_max",
            "generation_p",
            KIND_BOX_PRIOR_PGEN,
        ),
        _BoxSection(
            "exist_gen",
            "generation_q_min",
            "generation_q_max",
            "generation_q",
            KIND_BOX_PRIOR_QGEN,
        ),
        _BoxSection(
            "exist_load",
            "load_p_min",
            "load_p_max",
            "load_p",
            KIND_BOX_PRIOR_PNAG,
        ),
        _BoxSection(
            "exist_load",
            "load_q_min",
            "load_q_max",
            "load_q",
            KIND_BOX_PRIOR_QNAG,
        ),
    ]

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        nid = int(row["id"])
        if nid not in bus_id_to_pos:
            continue
        pos = bus_id_to_pos[nid]
        exist_flags = {
            "exist_gen": bool(row["exist_gen"]),
            "exist_load": bool(row["exist_load"]),
        }

        # Init box-vars are taken from the node row (load_p/q, generation_p/q).
        # EVERY exist_* node gets a box variable (unset bounds ->
        # a wide default): the balance equation is written for all active
        # nodes, and a node with exist_* but without a variable would enter
        # it as zero, so its P/Q injection would be pinned to zero as for a
        # transit node. Full coverage also guarantees that
        # ``write_node_estimates`` fills all 4 ``*_estimated`` fields.
        for sec in sections:
            if not exist_flags[sec.exist_col]:
                continue
            if sec.min_col == "generation_p_min" and _is_reactive_only(row):
                # Compensator (controllable shunt reactor, SVC, synchronous condenser):
                # no active power. A [0, 0] pair would otherwise read as "bounds not set"
                # and get a wide default box, and the estimate would produce phantom
                # active generation on the device.
                continue
            lo, hi, dflt = _bound_pair(
                float(row[sec.min_col]),
                float(row[sec.max_col]),
                default_box_halfwidth_pu,
            )
            sec.pos.append(pos)
            sec.lo.append(lo)
            sec.hi.append(hi)
            sec.init.append(_init_in_box(float(row[sec.init_col]) / BASE_MVA, (lo, hi)))
            sec.prior_s2.append(_resolve_prior_sigma2((lo, hi), is_default_box=dflt))

    return sections


@dataclass
class IPMSetup:
    """Extended layout + box bounds + augmented (z, R, meas_index)."""

    layout: StateLayout
    # Box bounds in p.u. (P/Q in MW are divided by BASE_MVA to match z).
    box_idx_in_state: np.ndarray  # positions in the state vector (length n_box)
    box_lo: np.ndarray  # lower bounds
    box_hi: np.ndarray  # upper bounds
    # Initial box-var values in p.u. (taken from the current generation_p/load_p).
    pgen_init: np.ndarray
    qgen_init: np.ndarray
    pnag_init: np.ndarray
    qnag_init: np.ndarray
    # Augmented z/R/meas_index with the added balance measurements.
    z: np.ndarray
    r_matrix: csr_matrix
    meas_index: MeasurementIndex


def _boxed_balance_masks(
    nodes_arr: np.ndarray, node_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """P / Q masks over ``node_ids``: the node's injection range is fully set by data."""
    rows = {int(r["id"]): r for r in nodes_arr}
    boxed = np.zeros((2, len(node_ids)), dtype=bool)
    for k, nid in enumerate(node_ids):
        r = rows[nid]
        has_gen, has_load = bool(r["exist_gen"]), bool(r["exist_load"])
        if not (has_gen or has_load):
            continue
        reactive_only = is_reactive_only_generation(
            r["generation_p_min"],
            r["generation_p_max"],
            r["generation_q_min"],
            r["generation_q_max"],
        )
        for axis, sfx in ((0, "p"), (1, "q")):
            pairs = []
            if has_gen and not (sfx == "p" and reactive_only):
                pairs.append((r[f"generation_{sfx}_min"], r[f"generation_{sfx}_max"]))
            if has_load:
                pairs.append((r[f"load_{sfx}_min"], r[f"load_{sfx}_max"]))
            # Both sides set; a point box counts (no hi > lo requirement here).
            boxed[axis, k] = all(
                declared_box(lo, hi, allow_point=True) is not None for lo, hi in pairs
            )
    return boxed[0], boxed[1]


def _median_measurement_sigma2(sigma2: np.ndarray) -> float | None:
    """Median variance (p.u.²) of the data rows; ``None`` when there are none."""
    positive = sigma2[sigma2 > 1e-15]
    return float(np.median(positive)) if positive.size > 0 else None


def _adaptive_balance_sigma2(sigma2: np.ndarray, balance_weight_factor: float) -> float:
    """Soft balance-row variance: median data variance / ``balance_weight_factor``."""
    median_sigma2 = _median_measurement_sigma2(sigma2)
    adaptive_sigma2 = (
        1e-6 if median_sigma2 is None else median_sigma2 / float(balance_weight_factor)
    )
    # Floor to avoid singularity at very small data σ².
    return max(adaptive_sigma2, 1e-12)


def build_ipm_setup(
    model: Working,
    network_pu: NetworkPU,
    z: np.ndarray,
    r_matrix: csr_matrix,
    meas_index: MeasurementIndex,
    *,
    layout_base: StateLayout,
    balance_sigma2: float | None = None,
    balance_weight_factor: float = 0.1,
    boxed_balance_weight_factor: float = 0.0,
    default_box_halfwidth_pu: float = 50.0,
    prior_sigma2_normal_pu: float = 0.0,
    prior_sigma2_bus_equiv_pu: float = 0.01,
    bus_equiv_width_threshold_pu: float = 100.0,
) -> IPMSetup:
    """Assemble the IPM setup on top of ready WLS data.

    Args:
        model: ``Working`` (for reading NODE_DTYPE).
        network_pu: internal p.u. representation.
        z, r_matrix, meas_index: WLS data (from ``build_z_and_r``).
        layout_base: WLS layout (without box-vars). The returned ``layout``
            inherits ``n_bus, slack_idx, non_slack_idx`` and adds the
            box sections.
        balance_sigma2: variance (p.u.²) for the balance pseudo-measurements. If
            ``None`` (default), computed adaptively:
            ``σ²_balance = median(σ²_data) / balance_weight_factor``
            (median, not min: the minimum is often an outlier and produces a
            balance weight that overrides the data by orders of magnitude).
        balance_weight_factor: ratio of the balance weight (1/σ²) to the weight of
            the median data measurement. Used when ``balance_sigma2=None``.
            Default 0.1: the balance is 10 times SOFTER than the median TM by σ²:
            calibrated in favor of data fit; the node balance is reached
            by the solver as a stationary point rather than hammered in by weight.
            Values >1 make the balance stiffer than the median measurement.
        boxed_balance_weight_factor: balance weight for nodes whose admissible
            injection is bounded by data: every declared part (``exist_gen`` /
            ``exist_load``) has both bounds set. Their variance is
            ``median(σ²_data) / boxed_balance_weight_factor`` (never looser than
            the soft one): the box already carries the node's freedom, so a
            loose balance only hides data conflicts as an imbalance. ``0`` — off.
        default_box_halfwidth_pu: half-width (p.u.) of the default box
            ``[-hw, +hw]`` for an exist_* node with unset bounds.
            Default 50 p.u. (±5 GW/GVAr), deliberately wider than any
            real node, so the barrier is effectively inactive and the value is
            governed by balance + TM. Default boxes get no
            BUS-equivalent prior (they are wide because of missing data,
            not because of a fictitious equivalent).
        prior_sigma2_normal_pu: σ² (p.u.²) of the prior measurement for the box-var
            of a node with an ordinary box (width <= ``bus_equiv_width_threshold_pu``).
            Default ``0``: no prior is created, data measurements via TM on
            branches govern the values. Transit nodes (``exist_=0``)
            have no box-var; their `Pgen-Pload = 0` is enforced
            through the balance measurement Sbus=0.
        prior_sigma2_bus_equiv_pu: σ² (p.u.²) of the prior measurement for BUS
            equivalents: nodes with a box width greater than
            ``bus_equiv_width_threshold_pu`` (typical external equivalents
            with a fictitious wide box of the order of tens of GW).
            Default ``0.01`` (≈ σ=10 MW): tight so as not to spread the
            residual, yet soft enough for TM on branches to be able to
            pull the values.
        bus_equiv_width_threshold_pu: box width threshold (p.u.,
            BASE_MVA=100 -> 100 p.u. = 10 GW of MW equivalent).
            Default ``100`` p.u. covers large BUS equivalents
            and does not affect ordinary nodes (typical width 5–50 p.u.).

    Returns:
        ``IPMSetup`` with the extended layout and augmented z/R/meas_index.
    """
    nodes_arr = model.nodes.to_numpy()
    bus_ids = network_pu.bus_ids
    bus_id_to_pos = id_to_pos_map(bus_ids)

    # ---- Collecting box-vars: for every active node with exist_load/exist_gen ----
    pgen_sec, qgen_sec, pnag_sec, qnag_sec = _collect_box_sections(
        nodes_arr,
        bus_id_to_pos,
        default_box_halfwidth_pu=default_box_halfwidth_pu,
        prior_sigma2_normal_pu=prior_sigma2_normal_pu,
        prior_sigma2_bus_equiv_pu=prior_sigma2_bus_equiv_pu,
        bus_equiv_width_threshold_pu=bus_equiv_width_threshold_pu,
    )
    pgen_node_pos = np.asarray(pgen_sec.pos, dtype=np.int64)
    qgen_node_pos = np.asarray(qgen_sec.pos, dtype=np.int64)
    pnag_node_pos = np.asarray(pnag_sec.pos, dtype=np.int64)
    qnag_node_pos = np.asarray(qnag_sec.pos, dtype=np.int64)

    layout = StateLayout(
        n_bus=layout_base.n_bus,
        slack_idx=layout_base.slack_idx,
        non_slack_idx=layout_base.non_slack_idx,
        pgen_node_pos=pgen_node_pos,
        qgen_node_pos=qgen_node_pos,
        pnag_node_pos=pnag_node_pos,
        qnag_node_pos=qnag_node_pos,
    )

    # ---- box bounds in state-vector coordinates ----
    box_idx: list[int] = []
    box_lo: list[float] = []
    box_hi: list[float] = []
    box_layout = (
        (layout.offset_pgen, pgen_sec),
        (layout.offset_qgen, qgen_sec),
        (layout.offset_pnag, pnag_sec),
        (layout.offset_qnag, qnag_sec),
    )
    for offset, sec in box_layout:
        for k in range(len(sec.lo)):
            box_idx.append(offset + k)
            box_lo.append(sec.lo[k])
            box_hi.append(sec.hi[k])

    box_idx_arr = np.asarray(box_idx, dtype=np.int64)
    box_lo_arr = np.asarray(box_lo, dtype=np.float64)
    box_hi_arr = np.asarray(box_hi, dtype=np.float64)

    # ---- Balance measurements: for every active node (P + Q) ----
    # z=0, σ²=balance_sigma2. Nodes without exist_load/gen are transit
    # (Sbus = 0 in brackets). Nodes with exist: link Sbus = Pgen-Pload.
    active_mask = nodes_arr["status"].astype(bool)
    active_nids = nodes_arr["id"][active_mask].astype(np.int64)
    active_positions: list[int] = []
    active_ids: list[int] = []
    for nid in active_nids.tolist():
        active_pos = bus_id_to_pos.get(int(nid))
        if active_pos is not None:
            active_positions.append(active_pos)
            active_ids.append(int(nid))
    n_balance = len(active_positions)

    # ---- Prior measurements: z=0, individual σ². Only for box-vars with σ²>0 ----
    pgen_prior_arr = np.asarray(pgen_sec.prior_s2, dtype=np.float64)
    qgen_prior_arr = np.asarray(qgen_sec.prior_s2, dtype=np.float64)
    pnag_prior_arr = np.asarray(pnag_sec.prior_s2, dtype=np.float64)
    qnag_prior_arr = np.asarray(qnag_sec.prior_s2, dtype=np.float64)
    pgen_prior_mask = pgen_prior_arr > 0
    qgen_prior_mask = qgen_prior_arr > 0
    pnag_prior_mask = pnag_prior_arr > 0
    qnag_prior_mask = qnag_prior_arr > 0
    n_prior = int(
        pgen_prior_mask.sum()
        + qgen_prior_mask.sum()
        + pnag_prior_mask.sum()
        + qnag_prior_mask.sum()
    )

    n_old = int(z.shape[0])
    n_new = n_old + 2 * n_balance + n_prior

    z_aug = np.zeros(n_new, dtype=np.float64)
    z_aug[:n_old] = z
    # Prior rows are anchored to the INIT box-var (materialized load/gen in p.u.),
    # NOT to zero: a BUS equivalent with a fictitious gross pair (10 GW load +
    # 12 GW generation at one node) lost its anchor at z=0: a tight prior
    # pinned both variables to zero, and the solver split the pure flow
    # symmetrically (±1.9 GW instead of 10/12). Balances + TM still steer
    # NET; the prior holds the GROSS level.
    if n_prior > 0:
        prior_z = np.concatenate(
            [
                np.asarray(pgen_sec.init, dtype=np.float64)[pgen_prior_mask],
                np.asarray(qgen_sec.init, dtype=np.float64)[qgen_prior_mask],
                np.asarray(pnag_sec.init, dtype=np.float64)[pnag_prior_mask],
                np.asarray(qnag_sec.init, dtype=np.float64)[qnag_prior_mask],
            ]
        )
        z_aug[n_old + 2 * n_balance :] = prior_z

    # R: extend the diagonal.
    # Adaptive balance_sigma2: take the median (not min) σ² of the data measurements as a
    # representative base, divided by balance_weight_factor. The minimum is
    # often an outlier (one tight measurement) and produces a balance weight that
    # overrides the other data by 4 orders of magnitude, so the solver sacrifices
    # data fit for the balance.
    sigma2_old = r_matrix.diagonal()
    if balance_sigma2 is None:
        balance_sigma2_eff = _adaptive_balance_sigma2(sigma2_old, balance_weight_factor)
    else:
        balance_sigma2_eff = float(balance_sigma2)

    sigma2_new = np.empty(n_new, dtype=np.float64)
    sigma2_new[:n_old] = sigma2_old
    sigma2_new[n_old : n_old + 2 * n_balance] = balance_sigma2_eff
    if boxed_balance_weight_factor > 0.0 and n_balance > 0:
        median_sigma2 = _median_measurement_sigma2(sigma2_old)
        if median_sigma2 is not None:
            boxed_s2 = min(median_sigma2 / float(boxed_balance_weight_factor), balance_sigma2_eff)
            boxed_p, boxed_q = _boxed_balance_masks(nodes_arr, active_ids)
            p_rows = sigma2_new[n_old : n_old + n_balance]
            q_rows = sigma2_new[n_old + n_balance : n_old + 2 * n_balance]
            p_rows[boxed_p] = np.minimum(p_rows[boxed_p], boxed_s2)
            q_rows[boxed_q] = np.minimum(q_rows[boxed_q], boxed_s2)
    if n_prior > 0:
        prior_sigmas = np.concatenate(
            [
                pgen_prior_arr[pgen_prior_mask],
                qgen_prior_arr[qgen_prior_mask],
                pnag_prior_arr[pnag_prior_mask],
                qnag_prior_arr[qnag_prior_mask],
            ]
        )
        sigma2_new[n_old + 2 * n_balance :] = prior_sigmas
    r_aug = csr_matrix(
        (sigma2_new, (np.arange(n_new), np.arange(n_new))),
        shape=(n_new, n_new),
    )

    # MeasurementIndex: add rows (balance + prior)
    balance_pos_arr = np.asarray(active_positions, dtype=np.int64)
    kind_blocks = [
        meas_index.kind,
        np.full(n_balance, KIND_NODE_BALANCE_P, dtype=meas_index.kind.dtype),
        np.full(n_balance, KIND_NODE_BALANCE_Q, dtype=meas_index.kind.dtype),
    ]
    obj_kind_blocks = [
        meas_index.object_kind,
        np.full(2 * n_balance, OBJ_NODE, dtype=meas_index.object_kind.dtype),
    ]
    obj_pos_blocks = [meas_index.object_pos, balance_pos_arr, balance_pos_arr]
    side_blocks = [
        meas_index.branch_side,
        np.full(2 * n_balance, SIDE_NONE, dtype=meas_index.branch_side.dtype),
    ]

    if n_prior > 0:
        # Prior measurement: 1 per box-var with σ²>0. Kind by section type, pos=node_pos.
        prior_kind_list: list[int] = []
        prior_pos_list: list[int] = []
        for prior_kind, mask, sec_pos_arr in (
            (KIND_BOX_PRIOR_PGEN, pgen_prior_mask, pgen_node_pos),
            (KIND_BOX_PRIOR_QGEN, qgen_prior_mask, qgen_node_pos),
            (KIND_BOX_PRIOR_PNAG, pnag_prior_mask, pnag_node_pos),
            (KIND_BOX_PRIOR_QNAG, qnag_prior_mask, qnag_node_pos),
        ):
            for i in np.where(mask)[0]:
                prior_kind_list.append(prior_kind)
                prior_pos_list.append(int(sec_pos_arr[i]))
        kind_blocks.append(np.asarray(prior_kind_list, dtype=meas_index.kind.dtype))
        obj_kind_blocks.append(np.full(n_prior, OBJ_NODE, dtype=meas_index.object_kind.dtype))
        obj_pos_blocks.append(np.asarray(prior_pos_list, dtype=meas_index.object_pos.dtype))
        side_blocks.append(np.full(n_prior, SIDE_NONE, dtype=meas_index.branch_side.dtype))

    new_kind = np.concatenate(kind_blocks)
    new_object_kind = np.concatenate(obj_kind_blocks)
    new_object_pos = np.concatenate(obj_pos_blocks)
    new_branch_side = np.concatenate(side_blocks)
    # Negative ids for balance + prior (do not collide with real measurement.id).
    pseudo_ids = -(np.arange(2 * n_balance + n_prior, dtype=meas_index.meas_id.dtype) + 1)
    new_meas_id = np.concatenate([meas_index.meas_id, pseudo_ids])

    meas_index_aug = MeasurementIndex(
        kind=new_kind,
        object_kind=new_object_kind,
        object_pos=new_object_pos,
        branch_side=new_branch_side,
        meas_id=new_meas_id,
    )

    return IPMSetup(
        layout=layout,
        box_idx_in_state=box_idx_arr,
        box_lo=box_lo_arr,
        box_hi=box_hi_arr,
        pgen_init=np.asarray(pgen_sec.init, dtype=np.float64),
        qgen_init=np.asarray(qgen_sec.init, dtype=np.float64),
        pnag_init=np.asarray(pnag_sec.init, dtype=np.float64),
        qnag_init=np.asarray(qnag_sec.init, dtype=np.float64),
        z=z_aug,
        r_matrix=r_aug,
        meas_index=meas_index_aug,
    )
