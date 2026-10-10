"""Conversion of ``Working`` (named units) ↔ the internal
p.u. representation of SE.

Inside SE everything is computed in the p.u. system with ``base_mva = 100`` and a voltage
base equal to ``NodeCollection.voltage_nominal`` of the corresponding
bus. Only the input data and the final write-back
into ``Working`` use named units.

Conventions:

- Base power ``S_base = 100 MVA`` (3-phase basis).
- Base voltage of each bus ``V_base = voltage_nominal`` (LL, kV).
- Base impedance ``Z_base = V_base² / S_base`` (Ohm).
- Base current ``I_base = S_base · 1000 / (√3 · V_base)`` (A).
- Branch impedances and admittances are interpreted as given on the "from" side
  (``from_node``): for a transformer the impedance is placed at the start of the branch,
  the ideal transformer at its end (see ``gridstate.ybus``).

The SE algorithms use *positional* bus indices (``0..n_bus−1``);
the original ``id``s from ``NODE_DTYPE`` are kept in ``NetworkPU.bus_ids`` —
the results are written back into the ``Working`` collections by them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate.constants import NodeType
from gridstate.utils import id_to_pos_map


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from gridstate.working import Working


BASE_MVA: float = 100.0
"""Base power of SE (fixed, 3-phase)."""


@dataclass
class NetworkPU:
    """Internal p.u. representation of the network for the numerical SE algorithms.

    All indices (``slack_idx``, ``from_idx``, ``to_idx``) are *positional* in the
    bus array (``0..n_bus−1``); the ``id``s themselves are kept in ``bus_ids`` for
    write-back.
    """

    # Topology
    n_bus: int
    n_branch: int
    bus_ids: np.ndarray  # (n_bus,) i8   — NODE_DTYPE.id
    bus_vn_kv: np.ndarray  # (n_bus,) f8   — voltage_nominal, kV
    bus_type: np.ndarray  # (n_bus,) i1   — 0=PQ, 1=PV, 2=SLACK
    slack_idx: int  # positional index of the slack node

    branch_ids: np.ndarray  # (n_branch,) i8 — BRANCH_DTYPE.id
    from_idx: np.ndarray  # (n_branch,) i8 — positional "from" index
    to_idx: np.ndarray  # (n_branch,) i8 — positional "to" index

    # Branch parameters in p.u.
    branch_r: np.ndarray  # series R
    branch_x: np.ndarray  # series X
    branch_g: np.ndarray  # total branch shunt G (Π-model)
    branch_b: np.ndarray  # total branch shunt B (Π-model)
    branch_g_from: np.ndarray  # shunt at the "from" side
    branch_b_from: np.ndarray
    branch_g_to: np.ndarray  # shunt at the "to" side
    branch_b_to: np.ndarray
    tap_ratio: np.ndarray  # magnitude of the forward Ktr (dimensionless)
    phase_shift: np.ndarray  # arg(Ktr), radians

    # Nodal shunts (p.u.)
    bus_g_shunt: np.ndarray
    bus_b_shunt: np.ndarray

    # Injections for the zero-injection classification (P_gen − P_load, p.u.)
    bus_p_injection: np.ndarray
    bus_q_injection: np.ndarray

    base_mva: float = BASE_MVA


# ---------------------------------------------------------------------------
# External → internal (Working → NetworkPU)
# ---------------------------------------------------------------------------


def model_to_pu(model: Working) -> NetworkPU:
    """Build a ``NetworkPU`` from ``Working`` (a thin adapter wrapper).

    Extracts the contract tables of nodes/branches (``to_numpy()``) and delegates
    the numeric conversion to :func:`network_pu_from_tables`. The p.u. conversion itself
    works only on the contract arrays — the model here is merely a source of tables.
    """
    return network_pu_from_tables(model.nodes.to_numpy(), model.branches.to_numpy())


def network_pu_from_tables(nodes_arr: np.ndarray, branches_arr: np.ndarray) -> NetworkPU:
    """Build a ``NetworkPU`` from the contract node/branch tables (numpy).

    **Array kernel** of the input converter: the input is the structured arrays
    of the input layer of the contract (``SEInput`` nodes/branches; columns —
    :data:`gridstate.contract.tables.NODES` / ``BRANCHES``), the output is the internal
    p.u. representation of SE. There is no dependence on ``Working`` — the array path
    calls here directly.

    Only active nodes (``status=True``) and active branches are taken into account.
    Branch impedances and admittances are interpreted as given on the "from" side.

    Raises:
        ValueError: if there are no nodes/slack node, ``voltage_nominal`` is not set,
            or a branch refers to a nonexistent node.
    """
    if len(nodes_arr) == 0:
        raise ValueError("Working is empty: no nodes")

    # Active nodes; indexing is positional over the resulting filter.
    active_nodes = nodes_arr[nodes_arr["status"]]
    if len(active_nodes) == 0:
        raise ValueError("All nodes are marked inactive (status=False)")

    bus_ids = active_nodes["id"].astype(np.int64, copy=True)
    vn_kv = active_nodes["voltage_nominal"].astype(np.float64, copy=True)
    if np.any(vn_kv <= 0):
        bad = bus_ids[vn_kv <= 0].tolist()
        raise ValueError(f"voltage_nominal ≤ 0 at nodes {bad}")

    bus_type = active_nodes["node_type"].astype(np.int8, copy=True)

    slack_positions = np.where(bus_type == NodeType.SLACK)[0]
    if slack_positions.size == 0:
        raise ValueError(
            "Working has no slack node (node_type=SLACK). SE requires exactly one slack."
        )
    if slack_positions.size > 1:
        # If there are several — take the one with the minimal balance_priority (i.e. the primary).
        # NOTE: this rule differs from topology._slack_nodes_to_demote (which
        # skips balance_priority <= 0 and slacks without an active branch, and
        # breaks ties by the smaller id; here ties go to the first position).
        priorities = active_nodes["balance_priority"][slack_positions]
        slack_idx = int(slack_positions[int(np.argmin(priorities))])
    else:
        slack_idx = int(slack_positions[0])

    # id → positional index map
    id_to_pos = id_to_pos_map(bus_ids)

    # Bases (element-wise for each bus)
    z_base = vn_kv**2 / BASE_MVA  # Ohm

    # Nodal shunts (S → p.u.)
    bus_g_shunt = active_nodes["shunt_g"].astype(np.float64) * z_base
    bus_b_shunt = active_nodes["shunt_b"].astype(np.float64) * z_base

    # Injections (MW/MVAr → p.u.)
    bus_p_inj = (active_nodes["generation_p"] - active_nodes["load_p"]).astype(
        np.float64
    ) / BASE_MVA
    bus_q_inj = (active_nodes["generation_q"] - active_nodes["load_q"]).astype(
        np.float64
    ) / BASE_MVA

    # Angles — already in radians per NODE_DTYPE; we numerically check the range.
    voltage_angle = active_nodes["voltage_angle"].astype(np.float64)
    if np.any(np.abs(voltage_angle) > 2 * math.pi):
        raise ValueError(
            "voltage_angle is outside the range [−2π, +2π] — the model may store the angle in "
            "degrees. NODE_DTYPE requires radians."
        )

    # ---- Branches ----
    if len(branches_arr) == 0:
        # A network without branches is formally valid (a single node) — continue.
        active_branches = branches_arr
    else:
        active_branches = branches_arr[branches_arr["status"]]

    # Filter out branches with at least one end outside the active nodes.
    if len(active_branches) > 0:
        keep = np.array(
            [
                int(b["from_node"]) in id_to_pos and int(b["to_node"]) in id_to_pos
                for b in active_branches
            ],
            dtype=bool,
        )
        active_branches = active_branches[keep]

    n_branch = len(active_branches)
    branch_ids = (
        active_branches["id"].astype(np.int64, copy=True)
        if n_branch > 0
        else np.empty(0, dtype=np.int64)
    )
    from_idx = np.empty(n_branch, dtype=np.int64)
    to_idx = np.empty(n_branch, dtype=np.int64)
    branch_r = np.empty(n_branch, dtype=np.float64)
    branch_x = np.empty(n_branch, dtype=np.float64)
    branch_g = np.empty(n_branch, dtype=np.float64)
    branch_b = np.empty(n_branch, dtype=np.float64)
    branch_g_from = np.empty(n_branch, dtype=np.float64)
    branch_b_from = np.empty(n_branch, dtype=np.float64)
    branch_g_to = np.empty(n_branch, dtype=np.float64)
    branch_b_to = np.empty(n_branch, dtype=np.float64)
    tap_ratio = np.empty(n_branch, dtype=np.float64)
    phase_shift = np.empty(n_branch, dtype=np.float64)

    for i, b in enumerate(active_branches):
        f_pos = id_to_pos[int(b["from_node"])]
        t_pos = id_to_pos[int(b["to_node"])]
        from_idx[i] = f_pos
        to_idx[i] = t_pos

        # The impedance is given in Ohm on the "from" side; build_ybus refers it
        # behind the ideal transformer by the actual ratio.
        zb = float(z_base[f_pos])
        branch_r[i] = float(b["resistance"]) / zb
        branch_x[i] = float(b["reactance"]) / zb
        branch_g[i] = float(b["conductance"]) * zb
        branch_b[i] = float(b["susceptance"]) * zb
        branch_g_from[i] = float(b["conductance_from"]) * zb
        branch_b_from[i] = float(b["susceptance_from"]) * zb
        branch_g_to[i] = float(b["conductance_to"]) * zb
        branch_b_to[i] = float(b["susceptance_to"]) * zb

        tap = float(b["tap_ratio"])
        if tap <= 0:
            tap = 1.0
        # Input format convention: ``tap_ratio`` is stored as the **physical**
        # turn ratio HV:LV (e.g. 2.27 for a 750/330 trafo). After
        # base normalization (V_base_node = voltage_nominal_node) the different
        # sides of the trafo have different V_base, and in p.u. the ratio becomes
        # K_pu = K_physical / (V_base_from / V_base_to). For an ideal
        # trafo K_pu = 1.0; an off-nominal tap (~±10%) gives 0.9-1.1.
        # Without this normalization yff/yft in build_ybus get a
        # double-counted turn ratio: on a 750/330 branch at flat init
        # h(P_flow_to) comes out at 2160 p.u. = 216 GW.
        vn_from = float(vn_kv[f_pos])
        vn_to = float(vn_kv[t_pos])
        if vn_from > 0 and vn_to > 0 and not math.isclose(vn_from, vn_to):
            tap = tap / (vn_from / vn_to)
        tap_ratio[i] = tap
        phase_shift[i] = float(b["phase_shift"])

    return NetworkPU(
        n_bus=len(bus_ids),
        n_branch=n_branch,
        bus_ids=bus_ids,
        bus_vn_kv=vn_kv,
        bus_type=bus_type,
        slack_idx=slack_idx,
        branch_ids=branch_ids,
        from_idx=from_idx,
        to_idx=to_idx,
        branch_r=branch_r,
        branch_x=branch_x,
        branch_g=branch_g,
        branch_b=branch_b,
        branch_g_from=branch_g_from,
        branch_b_from=branch_b_from,
        branch_g_to=branch_g_to,
        branch_b_to=branch_b_to,
        tap_ratio=tap_ratio,
        phase_shift=phase_shift,
        bus_g_shunt=bus_g_shunt,
        bus_b_shunt=bus_b_shunt,
        bus_p_injection=bus_p_inj,
        bus_q_injection=bus_q_inj,
    )


# ---------------------------------------------------------------------------
# Internal → external (writing the SE results into Working)
# ---------------------------------------------------------------------------


def compute_node_results_pu(
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    *,
    ybus: csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Numeric kernel of the nodal write-back: pu solution → named quantities (arrays).

    A pure function over arrays (without ``Working``): converts the voltage
    magnitudes to kV and, if ``ybus`` is given, computes the nodal injections
    ``S = V·conj(Ybus·V)·base_mva`` (MW/MVAr).

    Returns:
        ``(voltage_kv, p_inj_mw, q_inj_mvar)`` — arrays of length ``n_bus`` in
        the order of ``network_pu.bus_ids``; ``p_inj_mw``/``q_inj_mvar`` = ``None``
        if ``ybus`` is not passed.
    """
    voltage_kv = v_pu * network_pu.bus_vn_kv
    if ybus is None:
        return voltage_kv, None, None
    v_complex_full = v_pu * np.exp(1j * delta_rad)
    i_bus = ybus @ v_complex_full
    s_bus = v_complex_full * np.conj(i_bus)
    return voltage_kv, s_bus.real * BASE_MVA, s_bus.imag * BASE_MVA


def compute_branch_results_pu(
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    yf: csr_matrix,
    yt: csr_matrix,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Numeric kernel of the branch write-back: pu solution + Yf/Yt → flows/currents/losses.

    A pure function over arrays (without ``Working``). All quantities are in
    named units (MW/MVAr/A), in the order of ``network_pu.branch_ids``.

    Returns:
        ``(p_from_mw, q_from_mvar, p_to_mw, q_to_mvar, i_from_a, i_to_a,
        loss_p, loss_q)``.
    """
    v_complex_full = v_pu * np.exp(1j * delta_rad)
    i_from = yf @ v_complex_full  # (n_branch,) complex, in p.u.
    i_to = yt @ v_complex_full
    s_from = v_complex_full[network_pu.from_idx] * np.conj(i_from)
    s_to = v_complex_full[network_pu.to_idx] * np.conj(i_to)

    # Base currents at the side of each branch (A)
    sqrt3 = math.sqrt(3.0)
    i_base_from = BASE_MVA * 1000.0 / (sqrt3 * network_pu.bus_vn_kv[network_pu.from_idx])
    i_base_to = BASE_MVA * 1000.0 / (sqrt3 * network_pu.bus_vn_kv[network_pu.to_idx])

    p_from_mw = s_from.real * BASE_MVA
    q_from_mvar = s_from.imag * BASE_MVA
    p_to_mw = s_to.real * BASE_MVA
    q_to_mvar = s_to.imag * BASE_MVA
    i_from_a = np.abs(i_from) * i_base_from
    i_to_a = np.abs(i_to) * i_base_to

    # Branch losses: the sign convention in power_from_p/power_to_p is such
    # that P_from > 0 = flow enters the branch at the from side, P_to > 0 = leaves
    # at the to side. Hence losses = P_from + P_to (not P_from − P_to).
    loss_p = p_from_mw + p_to_mw
    loss_q = q_from_mvar + q_to_mvar
    return (
        p_from_mw,
        q_from_mvar,
        p_to_mw,
        q_to_mvar,
        i_from_a,
        i_to_a,
        loss_p,
        loss_q,
    )


def write_results_to_model(
    model: Working,
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    yf: csr_matrix | None = None,
    yt: csr_matrix | None = None,
    ybus: csr_matrix | None = None,
) -> None:
    """Write the SE result back into ``Working`` (named units).

    Args:
        model: ``Working``, updated in place.
        v_pu: (n_bus,) — voltage magnitudes in p.u. (correspond to ``network_pu.bus_ids``).
        delta_rad: (n_bus,) — voltage angles in radians.
        network_pu: the internal representation used for the computation.
        yf: (n_branch × n_bus) — the "from" matrix in p.u. If given together with ``yt``,
            the branch flows, currents and losses are recomputed.
        yt: (n_branch × n_bus) — the "to" matrix.
        ybus: (n_bus × n_bus) — the nodal matrix in p.u. If given, the node fields
            ``p_inj_calc``/``q_inj_calc``/``imbalance_p``/``imbalance_q`` are filled.

    Fills:

    Nodes (``NodeCollection``):
        - ``voltage_magnitude`` (kV = v_pu × voltage_nominal),
        - ``voltage_angle`` (rad);
        - with ``ybus`` — ``p_inj_calc``/``q_inj_calc`` (MW/MVAr) and
          ``imbalance_p`` = ``p_inj_calc`` − (``generation_p`` − ``load_p``),
          ``imbalance_q`` likewise.

    Branches (``BranchCollection``) — with ``yf, yt``:
        - ``power_from_p/q`` (MW/MVAr), ``power_to_p/q``,
        - ``current_from/to`` (A);
        - ``loss_p`` = ``power_from_p`` + ``power_to_p`` (MW),
          ``loss_q`` likewise;
        - ``loading_pct`` = max(I_from, I_to) / ``current_limit_normal`` × 100,
          if the limit is set (otherwise 0).
    """
    if v_pu.shape != (network_pu.n_bus,):
        raise ValueError(f"v_pu must have length {network_pu.n_bus}, got {v_pu.shape}")
    if delta_rad.shape != (network_pu.n_bus,):
        raise ValueError(f"delta_rad must have length {network_pu.n_bus}, got {delta_rad.shape}")

    # ---- Nodes ---- (numeric kernel — on arrays; writing into the model — below)
    voltage_kv, p_inj_mw, q_inj_mvar = compute_node_results_pu(
        v_pu, delta_rad, network_pu, ybus=ybus
    )

    # Vectorized write-back through the backing structured array (same
    # technique as write_measurement_estimates): rows are resolved once via
    # an id map instead of a per-node .update() call. Columns absent from
    # the carrier dtype are skipped — matching the silent-ignore semantics
    # of _ArrayCollection.update().
    nodes_arr = model.nodes.to_numpy()
    node_names = set(nodes_arr.dtype.names or ())
    node_rows_map = id_to_pos_map(nodes_arr["id"])
    node_rows = np.fromiter(
        (node_rows_map.get(int(nid), -1) for nid in network_pu.bus_ids.tolist()),
        dtype=np.int64,
        count=network_pu.n_bus,
    )
    nv = node_rows >= 0
    nr = node_rows[nv]

    def _set_node(col: str, values: np.ndarray) -> None:
        if col in node_names:
            nodes_arr[col][nr] = values[nv]

    _set_node("voltage_magnitude", voltage_kv)
    _set_node("voltage_angle", delta_rad)
    # "Node solved" marker: only the network_pu nodes get 1;
    # nodes cut off by preprocessing stay solved=0 — their zeros in
    # OUTPUT must not be taken for an estimate (promote/PF filter on it).
    _set_node("solved", np.ones(network_pu.n_bus, dtype=np.int64))
    if p_inj_mw is not None and q_inj_mvar is not None:
        net_p_meas = (nodes_arr["generation_p"][node_rows] - nodes_arr["load_p"][node_rows]).astype(
            np.float64
        )
        net_q_meas = (nodes_arr["generation_q"][node_rows] - nodes_arr["load_q"][node_rows]).astype(
            np.float64
        )
        _set_node("p_inj_calc", p_inj_mw)
        _set_node("q_inj_calc", q_inj_mvar)
        _set_node("imbalance_p", p_inj_mw - net_p_meas)
        _set_node("imbalance_q", q_inj_mvar - net_q_meas)
    model.nodes.update_from_array(nodes_arr)

    if yf is None or yt is None or network_pu.n_branch == 0:
        return

    # ---- Branches: Sf/St in p.u., then conversion to MW/MVAr and A (numeric kernel) ----
    (
        p_from_mw,
        q_from_mvar,
        p_to_mw,
        q_to_mvar,
        i_from_a,
        i_to_a,
        loss_p,
        loss_q,
    ) = compute_branch_results_pu(v_pu, delta_rad, network_pu, yf, yt)

    # Vectorized write-back, same scheme as the nodes above.
    branches_arr = model.branches.to_numpy()
    branch_names = set(branches_arr.dtype.names or ())
    branch_rows_map = id_to_pos_map(branches_arr["id"])
    branch_rows = np.fromiter(
        (branch_rows_map.get(int(bid), -1) for bid in network_pu.branch_ids.tolist()),
        dtype=np.int64,
        count=network_pu.n_branch,
    )
    bv = branch_rows >= 0
    br = branch_rows[bv]

    def _set_branch(col: str, values: np.ndarray) -> None:
        if col in branch_names:
            branches_arr[col][br] = values[bv]

    _set_branch("power_from_p", p_from_mw)
    _set_branch("power_from_q", q_from_mvar)
    _set_branch("power_to_p", p_to_mw)
    _set_branch("power_to_q", q_to_mvar)
    _set_branch("current_from", i_from_a)
    _set_branch("current_to", i_to_a)
    _set_branch("loss_p", loss_p)
    _set_branch("loss_q", loss_q)
    # loading_pct: only when current_limit_normal (> 0) is set; other
    # rows are left untouched (as in the former row-by-row variant).
    if "loading_pct" in branch_names and "current_limit_normal" in branch_names:
        i_limit = branches_arr["current_limit_normal"][br].astype(np.float64)
        has_limit = i_limit > 0
        if np.any(has_limit):
            i_max_used = np.maximum(i_from_a[bv], i_to_a[bv])
            rows_lim = br[has_limit]
            branches_arr["loading_pct"][rows_lim] = (
                i_max_used[has_limit] / i_limit[has_limit] * 100.0
            )
    model.branches.update_from_array(branches_arr)
