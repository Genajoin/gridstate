"""V-mirror: the value of blind-cluster pseudo-V from the measured boundary level.

A cluster of active nodes without a single real measurement (single-branch 110/35 kV
taps, midpoints without readings) is observed only through its pseudo-V prior. On a
flat input the loader sets that prior to nominal (``value = Vnom``, σ 5 %) and the
estimate sags to nominal, while physically a blind cluster sits at the level of its
neighbouring MEASURED boundary (often 1.05–1.12 Vnom). The root of the blind-tail
bias is not the prior weight but its placeholder VALUE.

Mechanism (two-pass): from the first-pass solution take the median pu (``V/Vnom``)
of the measured nodes on the boundary of the blind cluster and move the cluster's
pseudo-V to ``pu · Vnom``; then warm re-solve. ONLY the pseudo-prior value changes,
the variance is kept (tightening a blind node was tried and rejected: with σ → tight
the estimate drifts).

Two physical gates (without them the lever is a saddle between input kinds: a flat
input likes an aggressive transfer, an input with a real operating point gets the
observed network pushed up):

1. **Same voltage class**: pu is taken only from a boundary of the same ``Vnom``.
   Across a transformer (a 220/110 autotransformer) pu is not preserved because of
   the tap and the winding drop; transferring the pu of a neighbouring class
   overshoots.
2. **Lift gate** (``pu_boundary − pu_node > min_lift``): only nodes whose solution
   is SYSTEMATICALLY below their boundary are moved. Nodes already at the level
   (vm ≈ Vnom by chance at a real operating point) stay put, otherwise lifting
   them pushes the observed network up through the Q balances.

The ``|pu−1| ≤ max_pu_dev`` gate also discards a garbage boundary. Only placeholder
priors (``|value − Vnom| < 1e-6 · Vnom``) are touched: the loader has already
anchored nodes with a meaningful operating point or a PV setpoint.

The configuration (max_pu_dev=0.25, min_lift=0.01) is validated on four large
production models: p50 / p95 improve by 13 % / 16 %, 2.4 % / 7 %, 3 % / 11 % and
1 %; the mean bias shrinks everywhere and the per-class maximum nowhere regresses
beyond noise. Values and statuses of real measurements are not touched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import active_adjacency, active_ids, id_value_map, meas_mask
from gridstate.constants import BranchType
from gridstate.utils import branch_endpoints_map
from gridstate.working import table_copy, table_replace
from gridstate.z_vector import KIND_VOLTAGE, OBJ_BRANCH, OBJ_NODE


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)

_FLAT_REL_TOL = 1e-6  # |value − Vnom| < tol · Vnom -> prior placeholder
_CROSS_AT_MIN_VNOM = 110.0  # cross-AT fallback only for EHV/HV (>=110 kV) on both sides


@dataclass(frozen=True)
class VMirrorPlan:
    """Plan for reassigning pseudo-V values of blind clusters."""

    new_values: tuple[tuple[int, float], ...]  # (node_id, new pseudo-V value)
    n_clusters: int  # blind clusters that passed the boundary gate

    @property
    def empty(self) -> bool:
        return not self.new_values


def _node_maps(nodes: np.ndarray) -> tuple[dict[int, float], dict[int, float], set[int]]:
    """``(vn_map, vse_map, active)``: nominal and solved V by node id, active node ids."""
    return (
        id_value_map(nodes, "voltage_nominal"),
        id_value_map(nodes, "voltage_magnitude"),
        active_ids(nodes),
    )


def _flat_pseudo_v(measurements: np.ndarray, vn_map: dict[int, float]) -> dict[int, float]:
    """Nodes whose active pseudo-V is a nominal placeholder -> their ``Vnom``.

    A placeholder is ``|value − Vnom| < _FLAT_REL_TOL · Vnom`` on a node with
    ``Vnom > 0``.
    """
    psv_flat: dict[int, float] = {}
    sel_ps = meas_mask(measurements, kinds=KIND_VOLTAGE, obj=OBJ_NODE, pseudo=True)
    for j in np.where(sel_ps)[0]:
        nid = int(measurements["object_id"][j])
        vnom = vn_map.get(nid, 0.0)
        if vnom > 0 and abs(float(measurements["value"][j]) - vnom) < _FLAT_REL_TOL * vnom:
            psv_flat[nid] = vnom
    return psv_flat


def _measured_nodes(measurements: np.ndarray, branches: np.ndarray) -> set[int]:
    """Nodes covered by a real TM (a node measurement or the side of a branch measurement)."""
    b2n = branch_endpoints_map(branches)
    sel_real = meas_mask(measurements, pseudo=False)
    measured: set[int] = set()
    for j in np.where(sel_real)[0]:
        ot = int(measurements["object_type"][j])
        if ot == OBJ_NODE:
            measured.add(int(measurements["object_id"][j]))
        elif ot == OBJ_BRANCH:
            ends = b2n.get(int(measurements["object_id"][j]))
            if ends:
                s = int(measurements["branch_side"][j])
                measured.update(ends[s : s + 1] if s in (0, 1) else ends)
    return measured


def classify_v_mirror(
    measurements: np.ndarray,
    branches: np.ndarray,
    nodes: np.ndarray,
    *,
    max_pu_dev: float,
    min_lift: float,
    cross_at: bool = False,
) -> VMirrorPlan:
    """Select blind clusters and compute the new value of their pseudo-V.

    Args:
        measurements: measurements of the working model AFTER the first solve.
        branches: branches (adjacency + real-TM coverage).
        nodes: nodes AFTER the solve; ``voltage_magnitude`` carries the first-pass V,
            ``voltage_nominal`` is the nominal.
        max_pu_dev: boundary gate ``|median(pu) − 1| ≤ max_pu_dev``.
        min_lift: lift gate ``pu_boundary − pu_node > min_lift`` (a node is touched
            only if it sits systematically below the boundary of its class).

    Returns:
        :class:`VMirrorPlan` (may be ``empty``).
    """
    vn_map, vse_map, active = _node_maps(nodes)

    adj = active_adjacency(branches)

    # Cross-AT (gated): adjacency over active transformers only
    # (branch_type==1, tap>0), for the fallback of blind clusters whose boundary
    # of the same class lies only beyond an AT. pu is invariant to an ideal tap.
    trafo_adj: defaultdict[int, set[int]] = defaultdict(set)
    if cross_at:
        trafo_adj = active_adjacency(
            branches,
            branches["status"].astype(bool)
            & (branches["branch_type"] == BranchType.TRANSFORMER)
            & (branches["tap_ratio"] > 0),
        )

    # pseudo-V placeholders (value == Vnom) per node
    psv_flat = _flat_pseudo_v(measurements, vn_map)

    unmeas = active - _measured_nodes(measurements, branches)

    # connected components of blind nodes over active branches
    seen: set[int] = set()
    new_values: list[tuple[int, float]] = []
    n_clusters = 0
    for start in unmeas:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            x = stack.pop()
            if x in seen or x not in unmeas:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj[x] - seen)
        # boundary = measured neighbors of the component
        boundary = set().union(*(adj[n] for n in comp)) - comp if comp else set()
        # Cross-AT: trafo boundary of the cluster (nodes beyond an active AT, another class).
        trafo_boundary = (
            (set().union(*(trafo_adj[n] for n in comp)) - comp) if (cross_at and comp) else set()
        )
        hit = False
        for n in sorted(comp):
            vn = psv_flat.get(n)
            if not vn:
                continue
            # Gate 1: pu only from the boundary of the SAME voltage class.
            pus = [
                vse_map[b] / vn_map[b]
                for b in boundary
                if abs(vn_map.get(b, 0.0) - vn) < _FLAT_REL_TOL * vn and vse_map.get(b, 0.0) > 0
            ]
            if not pus and cross_at and vn >= _CROSS_AT_MIN_VNOM:
                # Fallback: pu of the solved trafo boundary (another class), pu is invariant
                # to an ideal tap. Only EHV/HV (>=110 kV) on both sides: on
                # gen step-up (10-35 kV) the pu invariance across an AT is grossly violated
                # (large winding drop). The max_pu_dev/lift gates are kept.
                pus = [
                    vse_map[b] / vn_map[b]
                    for b in trafo_boundary
                    if vn_map.get(b, 0.0) >= _CROSS_AT_MIN_VNOM and vse_map.get(b, 0.0) > 0
                ]
            if not pus:
                continue
            pu = float(np.median(pus))
            if abs(pu - 1.0) > max_pu_dev:
                continue
            # Gate 2: lift, the node sits systematically below its boundary.
            pu_node = vse_map.get(n, 0.0) / vn
            if pu - pu_node <= min_lift:
                continue
            new_values.append((n, pu * vn))
            hit = True
        if hit:
            n_clusters += 1

    return VMirrorPlan(tuple(new_values), n_clusters)


def classify_chain_mirror(
    measurements: np.ndarray,
    branches: np.ndarray,
    nodes: np.ndarray,
    *,
    max_pu_dev: float,
    min_lift: float,
    decay: float,
    max_depth: int,
    cross_at: bool = False,
) -> VMirrorPlan:
    """Chain-mirror: pseudo-V of flat placeholders DEEP in the network, from real V.

    :func:`classify_v_mirror` covers only fully blind clusters (without
    a single real TM). Nodes "observable" only through flows do not fall into a cluster,
    but their V level is anchored by nothing except the flat prior ``value = Vnom``,
    so deep chains (d>=2 from the nearest live real-V measurement) sag toward nominal.

    Mechanism: multi-source BFS from nodes with a live real node-V measurement over active
    branches of the SAME voltage class (gate 1 of v_mirror; with ``cross_at``, plus
    active transformers with ``tap>0`` when both ends are >=110 kV, pu-invariant).
    A node at depth ``d`` inherits the median pu of the sources that reached it with the first
    front, with decay toward nominal::

        pu_target = 1 + (pu_src − 1) · decay^(d−1)

    (d=1 is the full pu, like the v_mirror boundary; deeper is a conservative decay,
    NOT an extrapolation). The other v_mirror gates are kept: only
    flat placeholders, ``|pu_src − 1| ≤ max_pu_dev`` (garbage source),
    lift gate ``pu_target − pu_node > min_lift`` (we move only systematically
    sagging nodes). The priors' σ is not touched.
    """
    vn_map, vse_map, active = _node_maps(nodes)

    # pu propagation edges: the same class always; through an active AT >=110 kV
    # on both sides only with cross_at (pu is invariant to an ideal tap)
    adj: dict[int, set[int]] = defaultdict(set)
    for f, t, st, bt, tap in zip(
        branches["from_node"],
        branches["to_node"],
        branches["status"],
        branches["branch_type"],
        branches["tap_ratio"],
        strict=True,
    ):
        if not st:
            continue
        fi, ti = int(f), int(t)
        if fi not in active or ti not in active:
            continue
        vf, vt = vn_map.get(fi, 0.0), vn_map.get(ti, 0.0)
        same_class = vf > 0 and abs(vf - vt) < _FLAT_REL_TOL * vf
        trafo_ok = (
            cross_at
            and int(bt) == BranchType.TRANSFORMER
            and float(tap) > 0
            and vf >= _CROSS_AT_MIN_VNOM
            and vt >= _CROSS_AT_MIN_VNOM
        )
        if same_class or trafo_ok:
            adj[fi].add(ti)
            adj[ti].add(fi)

    # sources: live real node-V measurements with a plausible solved pu
    sel_src = meas_mask(measurements, kinds=KIND_VOLTAGE, obj=OBJ_NODE, pseudo=False)
    level: dict[int, float] = {}
    for j in np.where(sel_src)[0]:
        nid = int(measurements["object_id"][j])
        vn = vn_map.get(nid, 0.0)
        vse = vse_map.get(nid, 0.0)
        if nid not in active or vn <= 0 or vse <= 0:
            continue
        pu = vse / vn
        if abs(pu - 1.0) <= max_pu_dev:
            level[nid] = pu

    # flat pseudo-V placeholders (candidates for targets)
    psv_flat = _flat_pseudo_v(measurements, vn_map)

    # level-synchronous multi-source BFS: a node gets the median pu of the first front
    dist: dict[int, int] = dict.fromkeys(level, 0)
    new_values: list[tuple[int, float]] = []
    max_d = 0
    for d in range(1, max_depth + 1):
        contrib: dict[int, list[float]] = defaultdict(list)
        for u, pu in level.items():
            for w in adj[u]:
                if w not in dist:
                    contrib[w].append(pu)
        if not contrib:
            break
        level = {}
        for w, pus in contrib.items():
            dist[w] = d
            raw_pu = float(np.median(pus))
            level[w] = raw_pu  # further on, pass the UNdecayed source level
            if w not in psv_flat or abs(raw_pu - 1.0) > max_pu_dev:
                continue
            vn = vn_map[w]
            pu_target = 1.0 + (raw_pu - 1.0) * decay ** (d - 1)
            pu_node = vse_map.get(w, 0.0) / vn
            if pu_target - pu_node <= min_lift:
                continue
            new_values.append((w, pu_target * vn))
            max_d = d

    return VMirrorPlan(tuple(new_values), n_clusters=max_d)


def apply_v_mirror_plan(model: Working, plan: VMirrorPlan) -> dict:
    """Reassign pseudo-V measurement values on the plan's nodes."""
    if not plan.new_values:
        return {"clusters": 0, "nodes": 0}
    new_by_node = dict(plan.new_values)
    m = table_copy(model.measurements)
    # Inactive pseudo-V rows are retargeted too (no status filter).
    sel = meas_mask(m, kinds=KIND_VOLTAGE, obj=OBJ_NODE, active=False, pseudo=True) & np.isin(
        m["object_id"], list(new_by_node)
    )
    n = 0
    for j in np.where(sel)[0]:
        nid = int(m["object_id"][j])
        if nid in new_by_node:
            m["value"][j] = new_by_node[nid]
            n += 1
    table_replace(model.measurements, m)
    return {"clusters": plan.n_clusters, "nodes": n}
