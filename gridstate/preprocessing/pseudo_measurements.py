"""Pseudo-measurements to remove under-observability in WLS.

A node without telemetry measurements makes the corresponding part of the state vector
unobservable, so the gain matrix becomes singular. To avoid this, for every
active node without a `V/P_inj/Q_inj` measurement we add a weak prior:

* `V ≈ V_nominal` (either `V_magnitude` from the loaded model, or `vzd`
  for a PV node) with σ=10% of Vnom (slack 0.1%, PV 5%). With
  ``voltage_scope="generation"`` the V prior of a node that cannot hold its
  voltage (no generation, not the slack) is loosened to an observability prior;
  with ``voltage_scope="none"`` every V prior is.
* `P_inj = pg - pn`, `Q_inj = qg - qn` from the node table (for transit
  nodes this gives 0). σ depends on the node type and empty-model detection.

A node that already has a measurement of a kind (real, mirrored or an external
pseudo-measurement, :func:`add_external_pseudo_measurements`) gets no prior of
that kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from gridstate._arrays import meas_mask
from gridstate.bounds import bounds_unset, declared_box
from gridstate.constants import NodeType
from gridstate.measurement_ids import EXTERNAL_PSEUDO_ID_BASE, PSEUDO_ID_BASE, IdAllocator
from gridstate.preprocessing._scan import scan_node_meas_by_kind
from gridstate.preprocessing.meas_rows import pseudo_node_measurement
from gridstate.preprocessing.node_props import extract_node_load_props_from_model
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from gridstate.working import Working


__all__ = ["add_external_pseudo_measurements", "add_pseudo_measurements"]

#: Kind of an external pseudo-measurement → ``measurement_type``.
_EXTERNAL_PSEUDO_KINDS = {
    "v": KIND_VOLTAGE,
    "p_inj": KIND_POWER_INJECTION_P,
    "q_inj": KIND_POWER_INJECTION_Q,
}
#: ``source_code`` of the external pseudo-measurements.
SOURCE_EXTERNAL_PSEUDO = "external_pseudo"


@dataclass(frozen=True)
class PseudoMeasConfig:
    """Tuning knobs for pseudo-measurement generation (defaults live here once).

    Field semantics mirror the ``add_pseudo_measurements`` kwargs one-to-one; the
    public adapter keeps its explicit signature and assembles this config. Defaults
    are declared in a single place so the core and adapter no longer duplicate them.
    """

    add_zero_injections: bool = True
    add_v_priors: bool = True
    slack_voltage_sigma_frac: float = 0.001
    zero_inj_variance: float = 100.0
    empty_model_loose_factor: float = 100.0
    load_inj_loose_factor: float = 10.0
    mid_start: int = PSEUDO_ID_BASE
    boundary_inj_loose_factor: float = 10000.0
    unobservable_v_sigma_frac: float | None = None
    unobservable_v_exclude_real_v_neighbor: bool = True
    unobservable_v_min_vm_deviation: float = 0.0
    # Nodes whose load-only injection measurement was released: generation is
    # unmeasured there, so the P/Q injection prior spans the generation range.
    unmeasured_gen_p_nodes: frozenset[int] = frozenset()
    unmeasured_gen_q_nodes: frozenset[int] = frozenset()
    # Loosen the P/Q injection prior of a node with load but no load data (no
    # box, nothing materialized) when the injection is fixed by measurements.
    loosen_unknown_load: bool = False
    # Nodes whose V prior keeps its σ (generation and slack nodes, or none);
    # ``None`` = every node. Outside it the V prior variance is multiplied by
    # ``v_out_of_scope_loose_factor``.
    v_scope_nodes: frozenset[int] | None = None
    v_out_of_scope_loose_factor: float = 1e4


def _generation_range_variance(row: np.void, lo_col: str, hi_col: str, fallback: float) -> float:
    """Squared half-width of the node generation range; ``fallback`` if unset."""
    box = declared_box(float(row[lo_col]), float(row[hi_col]))
    if box is not None:
        lo, hi = box
        return max(fallback, (0.5 * (hi - lo)) ** 2)
    return fallback * 100.0


def _detect_empty_model(nodes_arr: np.ndarray) -> bool:
    """Auto-detection of an "empty" model.

    Some snapshots have zeros in pg/pn/qg/qn in model.json (the mode is stored separately).
    On such models the transit criterion pg=pn=qg=qn=0 fires on ALL nodes, so a
    hard prior σ²=zero_inj_variance overpowers TM and the SE diverges. For an empty model
    we loosen all priors by empty_model_loose_factor.
    """
    n_active = 0
    n_loaded = 0
    for row in nodes_arr:
        if not row["status"]:
            continue
        n_active += 1
        if (
            abs(float(row["generation_p"])) > 1e-3
            or abs(float(row["load_p"])) > 1e-3
            or abs(float(row["generation_q"])) > 1e-3
            or abs(float(row["load_q"])) > 1e-3
        ):
            n_loaded += 1
    return n_active > 0 and n_loaded * 100 < n_active  # <1% loaded


def _real_v_neighbors(
    branches_arr: np.ndarray,
    have_v: set[int],
    config: PseudoMeasConfig,
) -> set[int]:
    """Nodes with an active-branch neighbour carrying a V measurement.

    Filled only when ``unobservable_v_sigma_frac`` is requested and
    ``unobservable_v_exclude_real_v_neighbor`` is set: such nodes are observable
    through the line and keep the loose V prior.
    """
    real_v_neighbor: set[int] = set()
    if (
        config.unobservable_v_sigma_frac is not None
        and config.unobservable_v_exclude_real_v_neighbor
    ):
        for r in branches_arr:
            if not r["status"]:
                continue
            f, t = int(r["from_node"]), int(r["to_node"])
            if f in have_v:
                real_v_neighbor.add(t)
            if t in have_v:
                real_v_neighbor.add(f)
    return real_v_neighbor


def _unknown_load_determined_nodes(
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    meas_arr: np.ndarray,
) -> tuple[set[int], set[int]]:
    """Nodes whose unknown load is fixed by measurements: ``(for_p, for_q)``.

    A node qualifies when it has load (``exist_load``), no load box (unset per
    :func:`gridstate.bounds.resolve_bounds`) and nothing materialized for that
    component: the zero injection prior there is an absence of data, not data.
    Such a prior is loosened only where the injection is determined anyway:

    * P: every active incident branch carries a real P flow measurement;
    * Q: every active incident branch carries a real Q flow measurement or
      real V measurements at both ends.
    """

    # A missing ``is_pseudo`` column means every row is real.
    def _real(kind: int, obj: int) -> set[int]:
        sel = meas_mask(meas_arr, kinds=kind, obj=obj, pseudo=False, missing_pseudo="real")
        return set(meas_arr["object_id"][sel].tolist())

    flow_p = _real(KIND_POWER_P, OBJ_BRANCH)
    flow_q = _real(KIND_POWER_Q, OBJ_BRANCH)
    real_v = _real(KIND_VOLTAGE, OBJ_NODE)

    incident: dict[int, list[int]] = {}
    covered_q = set(flow_q)
    for r in branches_arr:
        if not r["status"]:
            continue
        bid, f, t = int(r["id"]), int(r["from_node"]), int(r["to_node"])
        incident.setdefault(f, []).append(bid)
        incident.setdefault(t, []).append(bid)
        if f in real_v and t in real_v:
            covered_q.add(bid)

    for_p: set[int] = set()
    for_q: set[int] = set()
    for r in nodes_arr:
        if not r["status"] or not r["exist_load"]:
            continue
        nid = int(r["id"])
        branches = incident.get(nid)
        if not branches:
            continue
        for lo_col, hi_col, val_col, covered, out in (
            ("load_p_min", "load_p_max", "load_p", flow_p, for_p),
            ("load_q_min", "load_q_max", "load_q", covered_q, for_q),
        ):
            unset = bounds_unset(float(r[lo_col]), float(r[hi_col]))
            if not unset or abs(float(r[val_col])) > 1e-6:
                continue
            if all(b in covered for b in branches):
                out.add(nid)
    return for_p, for_q


def _build_pseudo_rows(
    nodes_arr: np.ndarray,
    node_load_props: dict[int, dict] | None,
    config: PseudoMeasConfig,
    *,
    have_v: set[int],
    have_pinj: set[int],
    have_qinj: set[int],
    is_empty_model: bool,
    real_v_neighbor: set[int],
    unknown_load_p: set[int] | None = None,
    unknown_load_q: set[int] | None = None,
    existing_ids: Iterable[int] | np.ndarray = (),
) -> tuple[list[dict], dict]:
    """Main loop: build pseudo V/P_inj/Q_inj rows for the active nodes.

    A sequential pass over ``nodes_arr`` in the original order, with the same row order
    within a node (V prior -> P_inj -> Q_inj) and monotonic id allocation
    from ``config.mid_start`` (checked against ``existing_ids``). Returns
    ``(new_rows, stats)``.
    """
    ids = IdAllocator(existing_ids, config.mid_start, name="add_pseudo_measurements")
    n_v_added = 0
    n_zinj_added = 0
    n_unobs_v = 0
    n_unknown_load = 0
    n_v_loose = 0

    new_rows: list[dict] = []
    for row in nodes_arr:
        if not row["status"]:
            continue
        nid = int(row["id"])
        vn = float(row["voltage_nominal"])
        vm = float(row["voltage_magnitude"])
        is_slack = int(row["node_type"]) == int(NodeType.SLACK)

        if config.add_v_priors and nid not in have_v and vn > 0:
            v_prior_value = vm if vm > 0 else vn
            is_pv_with_vzd = False
            if node_load_props is not None and nid in node_load_props:
                prop = node_load_props[nid]
                vzd = float(prop.get("vzd", 0.0))
                # Slack and PV nodes with a set U_ZAD setpoint use it as
                # the V prior (input-format semantics: the balancing node holds V=vzd,
                # a PV generator controls Q for V=vzd). Previously slack
                # was excluded and the pseudo took V=Vnom (flat), which differed from
                # vzd by 2-5 kV and pulled the SE toward nominal rather than the setpoint.
                if vzd > 0 and (prop.get("exist_gen") or is_slack):
                    v_prior_value = vzd
                    if not is_slack:
                        is_pv_with_vzd = True
            if is_slack:
                sigma = (config.slack_voltage_sigma_frac * vn) ** 2
            elif is_pv_with_vzd:
                sigma = (0.05 * vn) ** 2  # PV node is stiffer, vzd≈vras
            elif (
                config.unobservable_v_sigma_frac is not None
                and nid not in real_v_neighbor
                and abs(v_prior_value - vn) >= config.unobservable_v_min_vm_deviation * vn
            ):
                # An unobservable node without real V (and without a real-V
                # neighbour, per the gate) whose prior is a non-trivial operating
                # point supplied with the input (not a nominal placeholder): keep V
                # at vm with a hard anchor instead of 5 %, a loose σ lets it drift.
                sigma = (config.unobservable_v_sigma_frac * vn) ** 2
                n_unobs_v += 1
            else:
                # 5% Vnom: stiffer than the former 10%. On weakly observable
                # nodes (degree=1, empty-model P_inj σ=100) 10% allowed the SE
                # to drag V down to 0.3-0.5 p.u.: the pseudo-V did not hold the anchor.
                sigma = (0.05 * vn) ** 2
            if config.v_scope_nodes is not None and nid not in config.v_scope_nodes:
                # Outside the scope a prior at the nominal or input V would pull
                # the estimate there; keep it for observability only.
                sigma *= config.v_out_of_scope_loose_factor
                n_v_loose += 1
            new_rows.append(
                pseudo_node_measurement(ids.next(), nid, KIND_VOLTAGE, v_prior_value, float(sigma))
            )
            n_v_added += 1

        # The Q pseudo requires the node to have neither a real P_inj NOR a real Q_inj: a node with a
        # real P_inj is skipped entirely (want_pinj=False), so it does NOT get
        # a Q pseudo either, even if it has no Q_inj; this is the recorded behavior.
        want_pinj = config.add_zero_injections and nid not in have_pinj
        want_qinj = want_pinj and nid not in have_qinj
        if want_pinj:
            pg = float(row["generation_p"])
            pn = float(row["load_p"])
            qg = float(row["generation_q"])
            qn = float(row["load_q"])
            p_inj_prior = pg - pn
            q_inj_prior = qg - qn

            if node_load_props is not None and nid in node_load_props:
                prop = node_load_props[nid]
                transit = not bool(prop.get("exist_load")) and not bool(prop.get("exist_gen"))
            else:
                transit = abs(pg) < 1e-3 and abs(pn) < 1e-3 and abs(qg) < 1e-3 and abs(qn) < 1e-3

            if is_empty_model:
                if node_load_props is not None and transit:
                    var_p = config.zero_inj_variance * 10.0
                else:
                    var_p = config.zero_inj_variance * config.empty_model_loose_factor
            else:
                # Load/generation nodes are looser than transit ones by load_inj_loose_factor
                # (default 10.0 = historical behavior). On radial terminals
                # with materialized load≈0 and no real TM this loosening lets
                # V overshoot (the branch Q drives a phantom Q).
                var_p = (
                    config.zero_inj_variance
                    if transit
                    else config.zero_inj_variance * config.load_inj_loose_factor
                )
            var_q = var_p
            if nid in config.unmeasured_gen_p_nodes:
                var_p = _generation_range_variance(
                    row, "generation_p_min", "generation_p_max", var_p
                )
            if nid in config.unmeasured_gen_q_nodes:
                var_q = _generation_range_variance(
                    row, "generation_q_min", "generation_q_max", var_q
                )
            # Load without data, injection fixed by measurements: the prior
            # value is an absence of data, keep it out of the estimate.
            if unknown_load_p and nid in unknown_load_p:
                var_p *= config.boundary_inj_loose_factor
                n_unknown_load += 1
            if unknown_load_q and nid in unknown_load_q:
                var_q *= config.boundary_inj_loose_factor
            new_rows.append(
                pseudo_node_measurement(
                    ids.next(), nid, KIND_POWER_INJECTION_P, float(p_inj_prior), float(var_p)
                )
            )
            if want_qinj:
                new_rows.append(
                    pseudo_node_measurement(
                        ids.next(), nid, KIND_POWER_INJECTION_Q, float(q_inj_prior), float(var_q)
                    )
                )
                n_zinj_added += 1

    stats = {
        "v_priors_added": n_v_added,
        "zero_inj_added": n_zinj_added,
        "unobservable_v_tight": n_unobs_v,
        "unknown_load_loose": n_unknown_load,
    }
    if config.v_scope_nodes is not None:
        stats["v_priors_loose"] = n_v_loose
    return new_rows, stats


def _add_pseudo_measurements_on_arrays(
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    meas_arr: np.ndarray,
    node_load_props: dict[int, dict] | None,
    config: PseudoMeasConfig | None = None,
) -> tuple[list[dict], dict]:
    """Build pseudo V/P_inj/Q_inj rows on the contract arrays (does NOT mutate the inputs).

    Core (append pattern): reads ONLY the contract columns
    nodes/branches/measurements + a ready ``node_load_props`` (resolved by the adapter);
    RETURNS ``(new_rows, stats)``. For the meaning of ``config`` see the adapter
    ``add_pseudo_measurements`` (``PseudoMeasConfig`` duplicates the same fields with shared
    defaults). ``config=None`` means all defaults.

    **Must remain a sequential Python loop** over ``nodes_arr`` in the original
    order, with the same row order within a node (V prior -> P_inj -> Q_inj)
    and monotonic ``mid`` allocation: the σ²/p_inj_prior arithmetic is scalar and
    deterministic.
    """
    if config is None:
        config = PseudoMeasConfig()

    have = scan_node_meas_by_kind(
        meas_arr,
        (KIND_VOLTAGE, KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q),
    )
    have_v = have[KIND_VOLTAGE]
    have_pinj = have[KIND_POWER_INJECTION_P]
    have_qinj = have[KIND_POWER_INJECTION_Q]

    is_empty_model = _detect_empty_model(nodes_arr)
    real_v_neighbor = _real_v_neighbors(branches_arr, have_v, config)

    unknown_load_p: set[int] = set()
    unknown_load_q: set[int] = set()
    if config.loosen_unknown_load:
        unknown_load_p, unknown_load_q = _unknown_load_determined_nodes(
            nodes_arr, branches_arr, meas_arr
        )

    return _build_pseudo_rows(
        nodes_arr,
        node_load_props,
        config,
        have_v=have_v,
        have_pinj=have_pinj,
        have_qinj=have_qinj,
        is_empty_model=is_empty_model,
        real_v_neighbor=real_v_neighbor,
        unknown_load_p=unknown_load_p,
        unknown_load_q=unknown_load_q,
        existing_ids=meas_arr["id"],
    )


def add_pseudo_measurements(
    model: Working,
    *,
    add_zero_injections: bool = True,
    add_v_priors: bool = True,
    slack_voltage_sigma_frac: float = 0.001,
    zero_inj_variance: float = 100.0,
    empty_model_loose_factor: float = 100.0,
    load_inj_loose_factor: float = 10.0,
    node_load_props: dict[int, dict] | None = None,
    mid_start: int = PSEUDO_ID_BASE,
    boundary_inj_loose_factor: float = 10000.0,
    unobservable_v_sigma_frac: float | None = None,
    unobservable_v_exclude_real_v_neighbor: bool = True,
    unobservable_v_min_vm_deviation: float = 0.0,
    unmeasured_gen_p_nodes: frozenset[int] = frozenset(),
    unmeasured_gen_q_nodes: frozenset[int] = frozenset(),
    loosen_unknown_load: bool = False,
    voltage_scope: Literal["all", "generation", "none"] = "all",
    v_out_of_scope_loose_factor: float = 1e4,
) -> dict:
    """Augment the model with pseudo-measurements to remove under-observability.

    1. **V priors**: for every active node without a VOLTAGE measurement
       add ``V = voltage_magnitude/nominal`` with variance:
       slack: ``(slack_voltage_sigma_frac·Vnom)²``; the others:
       ``(0.1·Vnom)²`` (weak prior).
    2. **Zero-injection**: for active nodes without an INJECTION measurement
       add ``P_inj = Q_inj = (pg-pn)`` (from the node table) with variance:

       * **transit node** (``exist_load=0 AND exist_gen=0`` if ``node_load_props``
         is given, otherwise fallback ``pg=pn=qg=qn=0`` in the node table):
         hard prior ``variance=zero_inj_variance``.
       * **load/generation node**: weak prior
         ``variance = 10·zero_inj_variance``.
       * **empty-model** (fewer than 1% of nodes filled in the node table):
         all nodes get ``variance = zero_inj_variance·empty_model_loose_factor``,
         except true transits (``EXIST_PN=0 AND EXIST_PG=0`` from
         ``node_load_props``), whose variance is
         ``10·zero_inj_variance``.

    Args:
        model: Working to modify (in place).
        add_zero_injections: whether to add P/Q injections.
        add_v_priors: whether to add the V priors.
        slack_voltage_sigma_frac: σ_V fraction for the slack node.
        zero_inj_variance: base σ² for transit nodes (MW²).
        empty_model_loose_factor: multiplier for an empty model.
        node_load_props: ``ny → {exist_load, exist_gen, vzd, ...}`` —
            input-format semantics of EXIST_PN/PG. Without it, a fallback to
            zeros in the node table.
        mid_start: starting ID for the new measurements.
        boundary_inj_loose_factor: variance multiplier of the P/Q injection
            prior on the nodes selected by ``loosen_unknown_load``. Default
            10000.0 (σ × 100) keeps that prior in the system only for
            observability.
        unobservable_v_sigma_frac: if set, for **unobservable** nodes
            without a real V the pseudo-V is anchored hard: ``σ = (frac·Vnom)²`` instead of
            the usual 5%. Rationale: an unobservable node has no information
            beyond the operating point supplied with the input
            (``voltage_magnitude``), and a loose 5 % prior lets the solver drift
            (the dominant top-dV contribution on large models: terminal tails
            drift by 10-22 %). A hard anchor to the vm prior keeps the node at
            that operating point.
            ``None`` (default): behavior unchanged (5%).
        unobservable_v_exclude_real_v_neighbor: when tightening, do NOT anchor a node
            that has a neighbor over an active branch with a **real** V measurement:
            such a node is observable through the line (its V follows the
            neighbour instead of being held at vm). Default True.
        unobservable_v_min_vm_deviation: minimum relative deviation
            of the prior from nominal ``|v_prior - Vnom|/Vnom`` for tightening. Default
            0.0 (no restriction). Motivation: when ``vm`` ≈ ``Vnom`` (or
            ``vm=0`` → fallback to Vnom), the prior is a nominal placeholder,
            not a real operating point, and the true V on such nodes may be
            far from nominal (on large models about half of these nodes
            deviate by more than 2 %), so a hard anchor to nominal worsens the
            median. A threshold > 0 anchors only nodes with a non-trivial
            input V.
        unmeasured_gen_p_nodes / unmeasured_gen_q_nodes: nodes whose load-only
            injection measurement was released (see
            :mod:`gridstate.telemetry.load_only_injection`). Their generation is
            unmeasured, so the P (Q) injection prior gets the variance of the
            generation range half-width instead of pulling it to ``pg - pn``.
        loosen_unknown_load: a node with load but no load data (no box, nothing
            materialized) gets a zero injection prior that states the absence of
            data, not a value. Where the injection is fixed by measurements anyway
            (P: real P flows on every incident branch; Q: real Q flows or real V
            at both ends of every incident branch) that prior's variance is
            multiplied by ``boundary_inj_loose_factor``. Typical nodes: cut ends of
            lines leaving the model and substations of neighbouring systems.
        voltage_scope: nodes whose V prior keeps its σ. ``"all"`` (default):
            every node. ``"generation"``: only the nodes that can hold their
            voltage, i.e. the slack and the nodes with generation (``exist_gen``
            or an active generator); elsewhere the V prior variance is multiplied
            by ``v_out_of_scope_loose_factor``. A V prior at the nominal (or
            input) V on a load node without a V measurement pulls the estimate
            there, and the solver makes up the difference with the load Q.
            ``"none"``: every V prior is loosened, the slack and the generation
            nodes included; V is then held by the network, the measurements and
            the voltage limits. Meant for ``ipm`` / ``ipm_pd``, whose barrier
            keeps V within the limits; WLS has no such bound, and V of a node seen
            by no measurement rests on the loosened prior alone.
        v_out_of_scope_loose_factor: variance multiplier of the V prior outside
            ``voltage_scope``. Default 1e4 (σ × 100) keeps the prior in the
            system only for observability.

    Returns:
        ``{"v_priors_added": N, "zero_inj_added": N, "unobservable_v_tight": N,
        "unknown_load_loose": N}`` — counters, plus ``"v_priors_loose": N`` with
        ``voltage_scope`` other than ``"all"``.
    """
    nodes_arr = model.nodes.to_numpy()
    meas_arr = model.measurements.to_numpy()
    branches_arr = model.branches.to_numpy()

    # exist_load/exist_gen are filled by the input-format loader;
    # if present, node_load_props is derived from the model automatically.
    if node_load_props is None and (
        np.any(nodes_arr["exist_load"]) or np.any(nodes_arr["exist_gen"])
    ):
        node_load_props = extract_node_load_props_from_model(model)

    if voltage_scope not in ("all", "generation", "none"):
        raise ValueError(
            f"voltage_scope={voltage_scope!r} is not one of ['all', 'generation', 'none']"
        )
    v_scope_nodes: frozenset[int] | None = None
    if voltage_scope == "generation":
        gen = model.generators.to_numpy()
        holds_v = nodes_arr["exist_gen"].astype(bool)
        holds_v |= nodes_arr["node_type"] == int(NodeType.SLACK)
        v_scope_nodes = frozenset(
            nodes_arr["id"][holds_v].tolist() + gen["node_id"][gen["status"].astype(bool)].tolist()
        )
    elif voltage_scope == "none":
        v_scope_nodes = frozenset()

    config = PseudoMeasConfig(
        add_zero_injections=add_zero_injections,
        add_v_priors=add_v_priors,
        slack_voltage_sigma_frac=slack_voltage_sigma_frac,
        zero_inj_variance=zero_inj_variance,
        empty_model_loose_factor=empty_model_loose_factor,
        load_inj_loose_factor=load_inj_loose_factor,
        mid_start=mid_start,
        boundary_inj_loose_factor=boundary_inj_loose_factor,
        unobservable_v_sigma_frac=unobservable_v_sigma_frac,
        unobservable_v_exclude_real_v_neighbor=unobservable_v_exclude_real_v_neighbor,
        unobservable_v_min_vm_deviation=unobservable_v_min_vm_deviation,
        unmeasured_gen_p_nodes=frozenset(unmeasured_gen_p_nodes),
        unmeasured_gen_q_nodes=frozenset(unmeasured_gen_q_nodes),
        loosen_unknown_load=loosen_unknown_load,
        v_scope_nodes=v_scope_nodes,
        v_out_of_scope_loose_factor=v_out_of_scope_loose_factor,
    )

    new_rows, stats = _add_pseudo_measurements_on_arrays(
        nodes_arr,
        branches_arr,
        meas_arr,
        node_load_props,
        config,
    )
    # Batch insertion in a single concatenation: per-row .add() gives O(n²) on
    # thousands of pseudo-measurements (a bottleneck of large models).
    model.measurements.add_many(new_rows)
    return stats


def add_external_pseudo_measurements(
    model: Working,
    plan: Mapping[tuple[int, str], tuple[float, float]],
) -> dict:
    """Add the pseudo-measurements of ``plan`` as given.

    ``plan`` is ``{(node_id, kind): (value, sigma2)}`` (see
    :attr:`gridstate.contract.derived.DerivedInputs.pseudo_measurements`):
    ``kind`` ∈ ``v`` / ``p_inj`` / ``q_inj``, the value in kV / MW / Mvar
    (injections generation-positive), the variance in their squares. Each
    entry becomes a pseudo row (``is_pseudo``, ``source_code``
    ``"external_pseudo"``, ids from
    :data:`~gridstate.measurement_ids.EXTERNAL_PSEUDO_ID_BASE`). Entries on
    nodes that are missing or inactive are skipped (``"skipped_inactive"``).

    Raises:
        ValueError: an unknown kind, or a value or variance that is not finite
            (the variance must also be positive).
    """
    nodes_arr = model.nodes.to_numpy()
    active = {int(i) for i, on in zip(nodes_arr["id"], nodes_arr["status"], strict=True) if on}
    ids = IdAllocator(
        model.measurements.to_numpy()["id"],
        EXTERNAL_PSEUDO_ID_BASE,
        name="external_pseudo",
        skip_taken_start=True,
    )
    new_rows: list[dict] = []
    stats = {"added_v": 0, "added_p_inj": 0, "added_q_inj": 0, "skipped_inactive": 0}
    for (node_id, kind), (value, sigma2) in plan.items():
        mt = _EXTERNAL_PSEUDO_KINDS.get(kind)
        if mt is None:
            raise ValueError(
                f"pseudo_measurements[({node_id}, {kind!r})]: unknown kind; expected one "
                f"of {list(_EXTERNAL_PSEUDO_KINDS)}"
            )
        if not (np.isfinite(value) and np.isfinite(sigma2) and sigma2 > 0.0):
            raise ValueError(
                f"pseudo_measurements[({node_id}, {kind!r})] = ({value!r}, {sigma2!r}): "
                "expected a finite value and a finite positive variance"
            )
        if int(node_id) not in active:
            stats["skipped_inactive"] += 1
            continue
        new_rows.append(
            pseudo_node_measurement(
                ids.next(),
                int(node_id),
                mt,
                float(value),
                float(sigma2),
                source_code=SOURCE_EXTERNAL_PSEUDO,
            )
        )
        stats[f"added_{kind}"] += 1
    model.measurements.add_many(new_rows)
    return stats
