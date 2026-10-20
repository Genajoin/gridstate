"""Nodes joined by zero-impedance switches, merged into supernodes (IPM).

A switch (breaker, disconnector, bus section) is usually given to SE as a
branch with a tiny reactance: its two ends get two voltages, and the solver
spends freedom on a voltage drop that physically does not exist. Merging
removes that freedom: the nodes joined by such branches form a *supernode*
with one V and one δ, the merged branches leave the network and the flow
measurements on them are not used. Everything else of the member nodes stays:
their V measurements, their injection measurements and their generation /
load box variables.

How the IPM problem changes (:func:`reduce_ipm_problem`):

* state: V / δ per supernode instead of per node (``x_node = T · x``), the box
  variables are unchanged;
* node balance: one P and one Q row per supernode, the sum of the member rows
  (the flows through the merged branches cancel in the sum);
* net injection row of a member: the injection of a node inside a supernode is
  not a function of V / δ (the flows through the merged branches are not
  modelled), so the row measures the box variables of the node
  (``Pgen − Pnag``), i.e. the injection row minus the node's own balance row;
  on a member without a box variable of that axis the row is a constant and
  is dropped.

A merged transformer keeps its ratio and phase shift between the voltages of
its ends (a member's V / δ is a fixed image of the supernode's).

The flows of the merged branches are recovered after the solve from the
member balances (:func:`merged_branch_flows`): unique in a tree, split as over
equal impedances in a loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from gridstate.constants import MeasurementObjectType
from gridstate.preprocessing.ipm_setup import IPMSetup
from gridstate.state import StateLayout
from gridstate.telemetry.shunts import BREAKER_MIN_X, BREAKER_X_OHM_PER_KV
from gridstate.units import NetworkPU
from gridstate.utils import id_to_pos_map
from gridstate.working import Working, _ArrayCollection
from gridstate.z_vector import (
    KIND_NODE_BALANCE_P,
    KIND_NODE_BALANCE_Q,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    OBJ_NODE,
    MeasurementIndex,
)


#: Default merge threshold: ``|r + jx| ≤ 110·min_x·Vnom`` Ohm with ``min_x = 4e-6``
#: (the same rule that gives a breaker branch its reactance, see
#: :func:`gridstate.telemetry.normalize_breaker_reactance`).
DEFAULT_MERGE_Z_OHM_PER_KV = BREAKER_X_OHM_PER_KV * BREAKER_MIN_X

# Relative slack of the threshold: a breaker normalized by the default rule sits
# exactly on it.
_THRESHOLD_RTOL = 1e-9


def merged_branch_ids(
    nodes_arr: np.ndarray, branches_arr: np.ndarray, z_ohm_per_kv: float
) -> np.ndarray:
    """Ids of the branches merged into supernodes.

    A branch is merged when it is active, both its ends are active nodes and
    its impedance is ``|r + jx| ≤ z_ohm_per_kv · Vnom`` Ohm (``Vnom`` of the
    ``from`` node). Any ratio: across a merged transformer the voltages keep
    its ratio and phase shift (see :class:`Supernodes`).
    """
    if len(branches_arr) == 0:
        return np.zeros(0, dtype=np.int64)
    active = {int(r["id"]): float(r["voltage_nominal"]) for r in nodes_arr if bool(r["status"])}
    out: list[int] = []
    for b in branches_arr:
        if not bool(b["status"]):
            continue
        vn_from = active.get(int(b["from_node"]))
        vn_to = active.get(int(b["to_node"]))
        if vn_from is None or vn_to is None or vn_from <= 0.0 or vn_to <= 0.0:
            continue
        if int(b["from_node"]) == int(b["to_node"]):
            continue
        z = math.hypot(float(b["resistance"]), float(b["reactance"]))
        if z > z_ohm_per_kv * vn_from * (1.0 + _THRESHOLD_RTOL):
            continue
        out.append(int(b["id"]))
    return np.asarray(out, dtype=np.int64)


@dataclass(frozen=True)
class Supernodes:
    """Supernodes over the buses of a :class:`~gridstate.units.NetworkPU`.

    Attributes:
        group: (n_bus,) supernode of every bus position.
        rep: (n_group,) bus position that represents each supernode (the slack
            in its supernode, else the first member).
        branch_ids: ids of the merged branches.
        edges: (k, 3) ``[branch id, from pos, to pos]`` of every merged branch.
        forest: the rows of ``edges`` that span the supernodes (the voltage
            relations run over them).
        v_scale: (n_bus,) ``V_bus = v_scale · V_rep`` in p.u.: 1 across a merged
            branch between equal nominal voltages without a ratio, else the
            ratio in p.u. of the merged transformers on the way.
        delta_shift: (n_bus,) ``δ_bus = δ_rep + delta_shift`` (the phase shifts
            on the way).
    """

    group: np.ndarray
    rep: np.ndarray
    branch_ids: np.ndarray
    edges: np.ndarray
    forest: np.ndarray
    v_scale: np.ndarray
    delta_shift: np.ndarray

    @property
    def n_group(self) -> int:
        return int(self.rep.size)

    @property
    def merged(self) -> np.ndarray:
        """(n_bus,) mask: the bus belongs to a supernode of two or more buses."""
        size = np.bincount(self.group, minlength=self.n_group)
        return np.asarray(size[self.group] > 1)


def build_supernodes(
    network_pu: NetworkPU, branches_arr: np.ndarray, branch_ids: np.ndarray
) -> Supernodes:
    """Group the buses of ``network_pu`` joined by the branches ``branch_ids``.

    Across a merged branch ``from → to`` with the ratio ``K`` in p.u. and the
    phase shift ``φ`` (as :func:`gridstate.units.network_pu_from_tables` gives
    them): ``V_to = V_from / K``, ``δ_to = δ_from − φ``.
    """
    n = network_pu.n_bus
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    pos = id_to_pos_map(network_pu.bus_ids)
    ends = {int(b["id"]): (int(b["from_node"]), int(b["to_node"])) for b in branches_arr}
    ratio = {int(b["id"]): (float(b["tap_ratio"]), float(b["phase_shift"])) for b in branches_arr}
    edges: list[tuple[int, int, int]] = []
    forest: list[tuple[int, int, int]] = []
    for bid in branch_ids.tolist():
        f, t = pos[ends[bid][0]], pos[ends[bid][1]]
        edges.append((bid, f, t))
        rf, rt = find(f), find(t)
        if rf == rt:
            continue
        parent[max(rf, rt)] = min(rf, rt)
        forest.append((bid, f, t))
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    uniq, group = np.unique(roots, return_inverse=True)
    rep = uniq.copy()
    rep[group[network_pu.slack_idx]] = network_pu.slack_idx

    vn = network_pu.bus_vn_kv
    adj: dict[int, list[tuple[int, float, float]]] = {}
    for bid, f, t in forest:
        tap, phi = ratio[bid]
        k = (1.0 if tap <= 0.0 else tap) * float(vn[t]) / float(vn[f])
        adj.setdefault(f, []).append((t, 1.0 / k, -phi))
        adj.setdefault(t, []).append((f, k, phi))
    v_scale = np.ones(n)
    delta_shift = np.zeros(n)
    for root in rep.tolist():
        order, seen = [int(root)], {int(root)}
        for node in order:
            for nxt, scale, shift in adj.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    v_scale[nxt] = v_scale[node] * scale
                    delta_shift[nxt] = delta_shift[node] + shift
                    order.append(nxt)
    return Supernodes(
        group=group.astype(np.int64),
        rep=rep.astype(np.int64),
        branch_ids=np.asarray(branch_ids, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 3),
        forest=np.asarray(forest, dtype=np.int64).reshape(-1, 3),
        v_scale=v_scale,
        delta_shift=delta_shift,
    )


def add_merged_branch_shunts(
    network_pu: NetworkPU, branches_arr: np.ndarray, branch_ids: np.ndarray
) -> NetworkPU:
    """``network_pu`` with the shunt admittances of the merged branches moved to their ends."""
    if branch_ids.size == 0:
        return network_pu
    pos = id_to_pos_map(network_pu.bus_ids)
    merged = set(branch_ids.tolist())
    g = network_pu.bus_g_shunt.copy()
    b = network_pu.bus_b_shunt.copy()
    for row in branches_arr:
        if int(row["id"]) not in merged:
            continue
        f, t = pos[int(row["from_node"])], pos[int(row["to_node"])]
        zb = float(network_pu.bus_vn_kv[f]) ** 2 / network_pu.base_mva
        g[f] += (float(row["conductance_from"]) + 0.5 * float(row["conductance"])) * zb
        b[f] += (float(row["susceptance_from"]) + 0.5 * float(row["susceptance"])) * zb
        g[t] += (float(row["conductance_to"]) + 0.5 * float(row["conductance"])) * zb
        b[t] += (float(row["susceptance_to"]) + 0.5 * float(row["susceptance"])) * zb
    return replace(network_pu, bus_g_shunt=g, bus_b_shunt=b)


# ---------------------------------------------------------------------------
# Reduction of the IPM problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateReduction:
    """``x_node = t · x + c`` between the node state and the supernode state."""

    layout: StateLayout  # supernode layout (box sections as in the node layout)
    t: csr_matrix  # (n_node_state, n_state)
    c: np.ndarray  # (n_node_state,) the δ shifts of the members
    pick: np.ndarray  # (n_state,) node-state index that seeds each coordinate

    def expand(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.t @ x + self.c)

    def reduce(self, x_node: np.ndarray) -> np.ndarray:
        return np.asarray(x_node[self.pick])

    def box_index(self, j_node: np.ndarray) -> np.ndarray:
        """Supernode-state index of box coordinates given in the node state."""
        shift = int(self.t.shape[0]) - self.layout.size
        return np.asarray(j_node, dtype=np.int64) - shift


def reduce_state(layout: StateLayout, sn: Supernodes) -> StateReduction:
    """Supernode state of the node ``layout``: one V / δ per supernode, the same boxes.

    The representative of a supernode carries its V / δ as they are, a member
    the ``v_scale`` / ``delta_shift`` image of them.
    """
    n_node = layout.n_bus
    slack_group = int(sn.group[layout.slack_idx])
    non_slack = np.asarray([g for g in range(sn.n_group) if g != slack_group], dtype=np.int64)
    red = StateLayout(
        n_bus=sn.n_group,
        slack_idx=slack_group,
        non_slack_idx=non_slack,
        pgen_node_pos=sn.group[layout.pgen_node_pos],
        qgen_node_pos=sn.group[layout.qgen_node_pos],
        pnag_node_pos=sn.group[layout.pnag_node_pos],
        qnag_node_pos=sn.group[layout.qnag_node_pos],
    )
    delta_of_group = np.full(sn.n_group, -1, dtype=np.int64)
    delta_of_group[non_slack] = red.offset_delta + np.arange(non_slack.size)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    c = np.zeros(layout.size)
    # δ of a node: the δ of its supernode plus its shift; the slack supernode
    # keeps δ = 0.
    for k, p in enumerate(layout.non_slack_idx.tolist()):
        c[layout.offset_delta + k] = sn.delta_shift[p]
        j = int(delta_of_group[sn.group[p]])
        if j >= 0:
            rows.append(layout.offset_delta + k)
            cols.append(j)
            vals.append(1.0)
    for p in range(n_node):
        rows.append(layout.offset_v + p)
        cols.append(red.offset_v + int(sn.group[p]))
        vals.append(float(sn.v_scale[p]))
    n_box = layout.n_box
    rows.extend(range(layout.offset_v + n_node, layout.offset_v + n_node + n_box))
    cols.extend(range(red.offset_v + sn.n_group, red.offset_v + sn.n_group + n_box))
    vals.extend([1.0] * n_box)
    t = csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(layout.size, red.size),
    )
    delta_node = np.full(n_node, -1, dtype=np.int64)
    delta_node[layout.non_slack_idx] = layout.offset_delta + np.arange(n_node - 1)
    pick = np.empty(red.size, dtype=np.int64)
    pick[red.offset_delta + np.arange(non_slack.size)] = delta_node[sn.rep[non_slack]]
    pick[red.offset_v + np.arange(sn.n_group)] = layout.offset_v + sn.rep
    pick[red.offset_v + sn.n_group :] = np.arange(layout.offset_v + n_node, layout.size)
    return StateReduction(layout=red, t=t, c=c, pick=pick)


@dataclass(frozen=True)
class SolvedRows:
    """The rows and the state a supernode solve ran on (for the quality summary).

    ``residual`` / ``jacobian`` / ``sigma2`` at the solution, ``z`` the value of
    the measured quantity of each row (p.u.), ``meas_index`` the node row each
    row stands for, ``layout`` the supernode state.
    """

    residual: np.ndarray
    jacobian: csr_matrix
    sigma2: np.ndarray
    z: np.ndarray
    meas_index: MeasurementIndex
    layout: StateLayout
    network_pu: NetworkPU
    supernodes: Supernodes


@dataclass(frozen=True)
class RowReduction:
    """``r = rows · r_node`` between the node rows of an IPM setup and the solved rows."""

    rows: csr_matrix  # (m, m_node)
    src: np.ndarray  # (m,) node row each row stands for (the representative balance row)
    plain: np.ndarray  # (m,) bool: the row is one node row as it is
    converted: np.ndarray  # (m,) bool: a member injection row measuring its box variables
    sigma2: np.ndarray  # (m,)
    eq_rows: np.ndarray | None
    eq_tol: np.ndarray | None
    dropped_ids: np.ndarray  # measurement ids of the dropped member injection rows

    def apply(self, r_node: np.ndarray) -> np.ndarray:
        return np.asarray(self.rows @ r_node)


def reduce_rows(setup: IPMSetup, sn: Supernodes) -> RowReduction:
    """Solved rows of a node IPM setup: supernode balances, member injections on the boxes."""
    mi = setup.meas_index
    m_node = len(mi)
    merged = sn.merged
    lay = setup.layout
    has_box = (
        np.isin(np.arange(lay.n_bus), np.concatenate([lay.pgen_node_pos, lay.pnag_node_pos])),
        np.isin(np.arange(lay.n_bus), np.concatenate([lay.qgen_node_pos, lay.qnag_node_pos])),
    )
    nb = setup.n_balance
    b0 = setup.balance_start
    balance_row: list[dict[int, int]] = [{}, {}]
    for axis in (0, 1):
        for k in range(nb):
            i = b0 + axis * nb + k
            balance_row[axis][int(mi.object_pos[i])] = i
    eq = np.zeros(m_node, dtype=bool)
    tol = np.zeros(m_node)
    if setup.eq_rows is not None and setup.eq_tol is not None:
        eq[setup.eq_rows] = True
        tol[setup.eq_rows] = setup.eq_tol
    sigma2_node = setup.r_matrix.diagonal()

    r_rows: list[int] = []
    r_cols: list[int] = []
    r_vals: list[float] = []
    src: list[int] = []
    kind: list[int] = []  # 0 plain, 1 converted, 2 supernode balance
    sigma2: list[float] = []
    eq_out: list[int] = []
    tol_out: list[float] = []
    dropped: list[int] = []
    members: dict[int, list[int]] = {}
    for p in np.flatnonzero(merged).tolist():
        members.setdefault(int(sn.group[p]), []).append(p)

    def emit(cols: list[int], vals: list[float], s: int, k: int, s2: float) -> int:
        row = len(src)
        r_rows.extend([row] * len(cols))
        r_cols.extend(cols)
        r_vals.extend(vals)
        src.append(s)
        kind.append(k)
        sigma2.append(s2)
        return row

    for i in range(m_node):
        pos = int(mi.object_pos[i])
        node_row = int(mi.object_kind[i]) == OBJ_NODE
        k_i = int(mi.kind[i])
        if node_row and k_i in (KIND_NODE_BALANCE_P, KIND_NODE_BALANCE_Q) and merged[pos]:
            g = int(sn.group[pos])
            if pos != int(sn.rep[g]):
                continue  # folded into the representative's row
            axis = 0 if k_i == KIND_NODE_BALANCE_P else 1
            parts = [balance_row[axis][p] for p in members[g] if p in balance_row[axis]]
            all_eq = bool(eq[parts].all())
            loose = [float(sigma2_node[j]) for j in parts if not eq[j]]
            row = emit(parts, [1.0] * len(parts), i, 2, sigma2_node[i] if all_eq else max(loose))
            if all_eq:
                eq_out.append(row)
                tol_out.append(float(tol[i]))
            continue
        if (
            node_row
            and i < b0
            and k_i in (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q)
            and merged[pos]
        ):
            axis = 0 if k_i == KIND_POWER_INJECTION_P else 1
            if not has_box[axis][pos] or pos not in balance_row[axis]:
                dropped.append(int(mi.meas_id[i]))
                continue
            emit([i, balance_row[axis][pos]], [1.0, -1.0], i, 1, float(sigma2_node[i]))
            continue
        row = emit([i], [1.0], i, 0, float(sigma2_node[i]))
        if eq[i]:
            eq_out.append(row)
            tol_out.append(float(tol[i]))

    m = len(src)
    kinds = np.asarray(kind, dtype=np.int64)
    return RowReduction(
        rows=csr_matrix((r_vals, (r_rows, r_cols)), shape=(m, m_node)),
        src=np.asarray(src, dtype=np.int64),
        plain=kinds == 0,
        converted=kinds == 1,
        sigma2=np.asarray(sigma2, dtype=np.float64),
        eq_rows=np.asarray(eq_out, dtype=np.int64) if eq_out else None,
        eq_tol=np.asarray(tol_out, dtype=np.float64) if eq_out else None,
        dropped_ids=np.asarray(dropped, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Flows of the merged branches
# ---------------------------------------------------------------------------


def merged_branch_flows(
    sn: Supernodes, outflow_p: np.ndarray, outflow_q: np.ndarray
) -> dict[int, tuple[float, float]]:
    """``from``-side P / Q (p.u.) of every merged branch.

    ``outflow_*`` (n_bus,): what each bus sends out through the merged
    branches (its injection minus its flow over the network). The flows meet
    these outflows at every member except the representative (which takes the
    remaining imbalance of its supernode) with the least sum of squares: in a
    tree they are unique, in a loop of merged branches they split as over
    equal impedances.
    """
    if sn.edges.size == 0:
        return {}
    bids, f, t = sn.edges[:, 0], sn.edges[:, 1], sn.edges[:, 2]
    m = bids.size
    incidence = csr_matrix(
        (np.r_[np.ones(m), -np.ones(m)], (np.r_[f, t], np.r_[np.arange(m), np.arange(m)])),
        shape=(sn.group.size, m),
    )
    is_rep = np.zeros(sn.group.size, dtype=bool)
    is_rep[sn.rep] = True
    free = np.flatnonzero(sn.merged & ~is_rep)
    a = incidence[free]
    lap = (a @ a.T).tocsc()
    flows_p = a.T @ np.atleast_1d(spsolve(lap, outflow_p[free]))
    flows_q = a.T @ np.atleast_1d(spsolve(lap, outflow_q[free]))
    return {
        int(b): (float(p), float(q))
        for b, p, q in zip(bids.tolist(), flows_p.tolist(), flows_q.tolist(), strict=True)
    }


def write_merged_results(
    model: Working,
    network_pu: NetworkPU,
    sn: Supernodes,
    outflow_p: np.ndarray,
    outflow_q: np.ndarray,
    v_pu: np.ndarray,
) -> None:
    """Write the flows of the merged branches and complete the injections of their ends.

    :func:`gridstate.units.write_results_to_model` sees only the network without
    the merged branches, so the ``p/q_inj_calc`` (and ``imbalance_*``) of a
    member node miss what flows through them; the flows recovered by
    :func:`merged_branch_flows` are added here. The merged branches (an ideal
    transformer included) are lossless: ``power_to = −power_from``.
    """
    flows = merged_branch_flows(sn, outflow_p, outflow_q)
    pos = id_to_pos_map(network_pu.bus_ids)
    branches = model.branches.to_numpy()
    names = set(branches.dtype.names or ())
    row_of = id_to_pos_map(branches["id"])
    extra_p = np.zeros(network_pu.n_bus)
    extra_q = np.zeros(network_pu.n_bus)
    for bid, (p, q) in flows.items():
        i = row_of[bid]
        f, t = pos[int(branches["from_node"][i])], pos[int(branches["to_node"][i])]
        extra_p[f] += p
        extra_q[f] += q
        extra_p[t] -= p
        extra_q[t] -= q
        s_mva = math.hypot(p, q) * network_pu.base_mva
        current_from, current_to = (
            s_mva * 1000.0 / (math.sqrt(3.0) * max(float(v_pu[e] * network_pu.bus_vn_kv[e]), 1e-6))
            for e in (f, t)
        )
        values = {
            "power_from_p": p * network_pu.base_mva,
            "power_from_q": q * network_pu.base_mva,
            "power_to_p": -p * network_pu.base_mva,
            "power_to_q": -q * network_pu.base_mva,
            "current_from": current_from,
            "current_to": current_to,
            "loss_p": 0.0,
            "loss_q": 0.0,
        }
        for col, value in values.items():
            if col in names:
                branches[col][i] = value
    model.branches.update_from_array(branches)

    nodes = model.nodes.to_numpy()
    node_names = set(nodes.dtype.names or ())
    node_row = id_to_pos_map(nodes["id"])
    for p in np.flatnonzero(sn.merged).tolist():
        i = node_row[int(network_pu.bus_ids[p])]
        for col, extra in (
            ("p_inj_calc", extra_p[p]),
            ("q_inj_calc", extra_q[p]),
            ("imbalance_p", extra_p[p]),
            ("imbalance_q", extra_q[p]),
        ):
            if col in node_names:
                nodes[col][i] += extra * network_pu.base_mva
    model.nodes.update_from_array(nodes)


def without_branch_measurements(
    measurements: _ArrayCollection, branch_ids: np.ndarray
) -> _ArrayCollection:
    """A copy of ``measurements`` with the measurements of ``branch_ids`` switched off."""
    out = measurements.copy()
    arr = out.to_numpy()
    on_branch = (arr["object_type"] == int(MeasurementObjectType.BRANCH)) & np.isin(
        arr["object_id"], branch_ids
    )
    if on_branch.any():
        arr["status"][on_branch] = False
        out.update_from_array(arr)
    return out


def solved_base_rows(meas_index: MeasurementIndex, sn: Supernodes) -> np.ndarray:
    """Mask of the measurement rows that are functions of the supernode V / δ.

    The node injection rows of the merged buses are not (the flows through the
    merged branches are not modelled); everything else is.
    """
    node_inj = (meas_index.object_kind == OBJ_NODE) & np.isin(
        meas_index.kind, (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q)
    )
    pos = np.asarray(meas_index.object_pos, dtype=np.int64)
    merged = np.zeros(pos.shape, dtype=bool)
    merged[node_inj] = sn.merged[pos[node_inj]]
    return np.asarray(~merged)


def start_network(network_pu: NetworkPU, sn: Supernodes) -> NetworkPU:
    """``network_pu`` with the merged branches back, for the flat-start voltages.

    :func:`gridstate.state.tap_consistent_voltages` propagates the ratios over
    the branches; without the merged ones a part of the network reached only
    through them would start off its ratios. They come back with zero
    impedance (the start floors it) and the ratio between their ends.
    """
    if sn.edges.size == 0:
        return network_pu
    f, t = sn.edges[:, 1], sn.edges[:, 2]
    zeros = np.zeros(f.size)
    return replace(
        network_pu,
        n_branch=network_pu.n_branch + f.size,
        from_idx=np.r_[network_pu.from_idx, f],
        to_idx=np.r_[network_pu.to_idx, t],
        branch_r=np.r_[network_pu.branch_r, zeros],
        branch_x=np.r_[network_pu.branch_x, zeros],
        tap_ratio=np.r_[network_pu.tap_ratio, sn.v_scale[f] / sn.v_scale[t]],
    )
