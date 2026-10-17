"""Plausibility check of a solved state and hints for the log.

Any solver can converge to a state that fits the data but is physically
wrong. A typical case is a node whose voltage magnitude is barely observable
(no voltage measurement, measurements at zero, no injection): the estimation
problem then has a second, low-voltage root, and a change of the solver path
can land on it while the solver reports success. :func:`low_voltage_nodes`
finds such nodes; :func:`low_voltage_hints` and :func:`residual_hints` turn
the solved state into readable lines for the log.
"""

from __future__ import annotations

import numpy as np

from gridstate.units import BASE_MVA, NetworkPU
from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_BOX_PRIOR_QGEN,
    KIND_BOX_PRIOR_QNAG,
    KIND_NODE_BALANCE_P,
    KIND_NODE_BALANCE_Q,
    MeasurementIndex,
)


#: Row kinds by name (``MeasurementType`` plus the rows the IPM setup appends).
KIND_NAMES = {
    0: "P flow",
    1: "Q flow",
    2: "V",
    3: "I",
    4: "P injection",
    5: "Q injection",
    KIND_NODE_BALANCE_P: "P balance",
    KIND_NODE_BALANCE_Q: "Q balance",
    KIND_BOX_PRIOR_PGEN: "Pgen",
    KIND_BOX_PRIOR_QGEN: "Qgen",
    KIND_BOX_PRIOR_PNAG: "Pload",
    KIND_BOX_PRIOR_QNAG: "Qload",
}
_KIND_V = 2
#: |z| below this (p.u.) counts as a zero measurement.
_ZERO_PU = 1e-6


def describe_row(
    meas_index: MeasurementIndex, network_pu: NetworkPU, row: int, *, part: bool = False
) -> str:
    """``"<kind> measurement <id> at <object>"`` for row ``row`` of the measurement vector."""
    kind = int(meas_index.kind[row])
    name = KIND_NAMES.get(kind, f"kind {kind}")
    okind, opos = int(meas_index.object_kind[row]), int(meas_index.object_pos[row])
    if okind == 1:
        side = "from" if int(meas_index.branch_side[row]) == 0 else "to"
        obj = f"branch {int(network_pu.branch_ids[opos])} ({side})"
    else:
        obj = f"node {int(network_pu.bus_ids[opos])}"
    mid = int(meas_index.meas_id[row])
    what = "part measurement" if part else ("measurement" if mid > 0 else "row")
    return f"{name} {what} {mid} at {obj}" if mid > 0 else f"{name} {what} at {obj}"


def low_voltage_nodes(v_pu: np.ndarray, threshold: float) -> np.ndarray:
    """Positions of the nodes with ``V/Vnom`` below ``threshold`` (none when it is ≤ 0)."""
    if threshold <= 0.0:
        return np.zeros(0, dtype=np.int64)
    v = np.asarray(v_pu, dtype=np.float64)
    return np.flatnonzero(np.isfinite(v) & (v < threshold))


def _rows_at_node(meas_index: MeasurementIndex, network_pu: NetworkPU, pos: int) -> np.ndarray:
    """Rows measuring node ``pos`` or a branch incident to it."""
    branches = np.flatnonzero((network_pu.from_idx == pos) | (network_pu.to_idx == pos))
    node_rows = (meas_index.object_kind == 0) & (meas_index.object_pos == pos)
    branch_rows = (meas_index.object_kind == 1) & np.isin(meas_index.object_pos, branches)
    return np.flatnonzero(node_rows | branch_rows)


def low_voltage_hints(
    positions: np.ndarray,
    *,
    v_pu: np.ndarray,
    network_pu: NetworkPU,
    meas_index: MeasurementIndex,
    z: np.ndarray,
    top: int = 5,
) -> list[str]:
    """One line per low-voltage node (at most ``top``) with the likely cause."""
    lines: list[str] = []
    for pos in positions[np.argsort(v_pu[positions])][:top].tolist():
        node = int(network_pu.bus_ids[pos])
        vn = float(network_pu.bus_vn_kv[pos])
        rows = _rows_at_node(meas_index, network_pu, pos)
        real = rows[meas_index.meas_id[rows] > 0]
        v_rows = real[(meas_index.kind[real] == _KIND_V) & (meas_index.object_pos[real] == pos)]
        v_measured = v_rows[np.abs(z[v_rows]) > _ZERO_PU]
        zero = real[np.abs(z[real]) <= _ZERO_PU]
        inj = abs(float(network_pu.bus_p_injection[pos])) + abs(
            float(network_pu.bus_q_injection[pos])
        )
        nbr = {
            int(t)
            for f, t in zip(network_pu.from_idx.tolist(), network_pu.to_idx.tolist(), strict=True)
            if f == pos
        } | {
            int(f)
            for f, t in zip(network_pu.from_idx.tolist(), network_pu.to_idx.tolist(), strict=True)
            if t == pos
        }
        facts = [
            f"{len(real)} measurement(s) around it, {len(zero)} at zero",
            "V measured" if v_measured.size else "V not measured",
            f"{len(nbr - {pos})} neighbour node(s)",
        ]
        if inj == 0.0:
            facts.append("no injection")
        if v_measured.size:
            hint = (
                "the estimate contradicts the V measurement: check the measurement, "
                "the topology and the transformer ratios around the node"
            )
        else:
            hint = (
                "V is barely observable here, so the solver likely took the low-voltage "
                "root; add a V measurement or a V pseudo-measurement, or check the "
                "topology (a dead bus left energized?)"
            )
        lines.append(
            f"  node {node} ({vn:g} kV): V = {v_pu[pos]:.3f} p.u.; {', '.join(facts)} — {hint}"
        )
    return lines


def residual_hints(
    residual: np.ndarray,
    sigma: np.ndarray,
    *,
    meas_index: MeasurementIndex,
    network_pu: NetworkPU,
    skip: np.ndarray | None = None,
    part_rows: np.ndarray | None = None,
    top: int = 5,
) -> list[str]:
    """The ``top`` largest normalized residuals ``|r|/σ`` (rows of ``skip`` left out)."""
    rn = np.abs(residual) / np.where(sigma > 0, sigma, np.inf)
    if skip is not None:
        rn[skip] = 0.0
    is_part = np.zeros(len(residual), dtype=bool)
    if part_rows is not None:
        is_part[part_rows] = True
    order = np.argsort(-rn)[:top]
    if not order.size or rn[order[0]] <= 0:
        return []
    lines = ["largest normalized residuals |r|/σ:"]
    for i in order.tolist():
        kind = int(meas_index.kind[i])
        r_txt = f"{residual[i]:.4f} p.u." if kind in (2, 3) else f"{residual[i] * BASE_MVA:.2f}"
        desc = describe_row(meas_index, network_pu, i, part=bool(is_part[i]))
        lines.append(f"  {desc}: {rn[i]:.1f}σ (r = {r_txt})")
    return lines


__all__ = [
    "KIND_NAMES",
    "describe_row",
    "low_voltage_hints",
    "low_voltage_nodes",
    "residual_hints",
]
