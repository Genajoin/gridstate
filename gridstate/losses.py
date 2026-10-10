"""System active/reactive power loss report computed from an estimated state.

The module is *diagnostic*: it is not part of the main pipeline (``estimate``).
It is used for system loss reports and for cross-checking the estimated state.

Conventions:

* ``branch.susceptance`` in ``Working`` is the **full** line-charging
  susceptance B (S). In the Pi-model it is split in half: B/2 into the "from"
  shunt and B/2 into the "to" shunt. The same holds for ``branch.conductance``
  (corona active losses).
* ``branch.tap_ratio`` is stored as the **physical** HV:LV turn ratio (e.g.
  18.58 for 110/6 kV, 2.22 for 500/220). Inside SE it is normalised to p.u.:
  ``K_pu = K_phys / (V_base_from / V_base_to)`` (see ``gridstate/units.py``).
  An off-nominal tap (+-10 %) gives ``K_pu`` in ``[0.9..1.1]``. Losses are
  computed in p.u. by reusing ``model_to_pu`` + ``build_ybus`` (the same Y-bus
  as SE).
* For a transformer *all* of g+jb is usually placed on the "from" side via
  ``conductance_from`` / ``susceptance_from`` (``branch_g/b`` stay zero).
* Node shunts are ``node.shunt_g + j*node.shunt_b`` (S). Reactors folded into
  the node shunt by ``apply_reactors_to_node_shunt`` are already part of
  ``node.shunt_b`` and are not summed separately.

Formulas (per branch, everything in p.u. after ``model_to_pu``):

.. code::

    V_f, V_t - complex node voltages (V_pu * exp(j*delta));
    t     = tap_ratio_pu * exp(j*phase_shift);
    y_ser = |t|^2 / (r + j*x);   # impedance behind the ideal transformer

    # Power at the "from"/"to" side - same formulas as in build_ybus:
    S_from = V_f * conj(Y_ff * V_f + Y_ft * V_t);
    S_to   = V_t * conj(Y_tf * V_f + Y_tt * V_t);

    # Series losses (R + jX part, without the Pi-model shunt):
    I_ser = (V_f / t - V_t) * y_ser;
    S_series = |I_ser|^2 * (r + j*x) / |t|^2;

    # Total branch losses (including the shunts):
    S_loss_total = S_from + S_to;

    # Shunt part, by difference:
    S_loss_shunt = S_loss_total - S_series;

Node shunt (p.u.):

.. code::

    S_shunt_node = |V|^2 * conj(g_sh + j*b_sh)

All ``S`` are multiplied by ``BASE_MVA=100`` to get MW / Mvar.

Sign convention. The library uses the standard admittance convention
everywhere: ``Y = g + jb`` with ``b > 0`` for a capacitive element (capacitor
bank, line charging) and ``b < 0`` for an inductive one (shunt reactor). The
reported ``S = V * conj(I) = |V|^2 * conj(Y)`` is the power *absorbed* by the
element, hence:

* P of any element is positive (dissipation, ``g > 0``);
* series Q is positive for an inductive series branch (``x > 0``);
* shunt Q, node and branch alike, is ``-|V|^2 * b``: negative for capacitive
  elements (they *generate* Q, i.e. negative reactive "losses"), positive for
  reactors (they absorb Q).

No sign flip is applied anywhere. Consequently ``total_p_loss_mw`` and
``total_q_loss_mvar`` equal the sum of the complex nodal injections
``sum_i V_i * conj((Y_bus V)_i)`` of the solved state (Kirchhoff balance).
Input data that stores susceptance with the opposite sign must be converted by
the caller before it reaches ``Working``.

Breakdown identities (``SystemLosses`` fields):

* ``total_p_loss_mw = p_line_series + p_trafo_series + p_branch_shunt
  + p_node_shunt``;
* ``total_q_loss_mvar`` is the same sum for Q;
* ``p_branch_shunt`` / ``q_branch_shunt`` -- Pi-model shunts of all branches
  (no-load losses and magnetising Q of transformers, line charging);
* ``p_node_shunt`` / ``q_node_shunt`` -- node shunts (reactors, capacitors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gridstate.units import BASE_MVA, model_to_pu
from gridstate.utils import id_to_pos_map
from gridstate.ybus import build_ybus, series_admittance


if TYPE_CHECKING:
    from gridstate.working import Working


@dataclass
class BranchLossRow:
    """Per-branch loss breakdown (physical units, MW/MVAr)."""

    branch_id: int
    from_node: int
    to_node: int
    tip: str  # "line" / "trafo"
    p_loss_series: float
    q_loss_series: float
    p_loss_shunt: float
    q_loss_shunt: float
    p_loss_total: float
    q_loss_total: float


@dataclass
class SystemLosses:
    """System summary."""

    total_p_loss_mw: float
    total_q_loss_mvar: float
    # Breakdown by type, MW/MVAr
    p_line_series: float
    q_line_series: float
    p_trafo_series: float
    q_trafo_series: float
    p_branch_shunt: float
    q_branch_shunt: float
    p_node_shunt: float
    q_node_shunt: float
    # Per-branch list
    per_branch: list[BranchLossRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_p_loss_mw": self.total_p_loss_mw,
            "total_q_loss_mvar": self.total_q_loss_mvar,
            "breakdown": {
                "line_series_p": self.p_line_series,
                "line_series_q": self.q_line_series,
                "trafo_series_p": self.p_trafo_series,
                "trafo_series_q": self.q_trafo_series,
                "branch_shunt_p": self.p_branch_shunt,
                "branch_shunt_q": self.q_branch_shunt,
                "node_shunt_p": self.p_node_shunt,
                "node_shunt_q": self.q_node_shunt,
            },
            "per_branch": [
                {
                    "branch_id": r.branch_id,
                    "from_node": r.from_node,
                    "to_node": r.to_node,
                    "tip": r.tip,
                    "p_loss_series_mw": r.p_loss_series,
                    "q_loss_series_mvar": r.q_loss_series,
                    "p_loss_shunt_mw": r.p_loss_shunt,
                    "q_loss_shunt_mvar": r.q_loss_shunt,
                    "p_loss_total_mw": r.p_loss_total,
                    "q_loss_total_mvar": r.q_loss_total,
                }
                for r in self.per_branch
            ],
        }


def compute_system_losses(
    model: Working,
    v_pu: np.ndarray | None = None,
    delta_rad: np.ndarray | None = None,
) -> SystemLosses:
    """Compute the active/reactive system losses from the SE results.

    The computation is done in p.u. via ``gridstate.units.model_to_pu``, which
    guarantees consistency with the y-bus used in SE
    (see ``gridstate/ybus.py``). The final totals are converted to MW/MVAr
    via ``BASE_MVA=100``.

    If ``v_pu``/``delta_rad`` are not given — V/δ are taken from
    ``model.nodes.voltage_magnitude / voltage_angle`` (i.e. the already written
    SE results; see ``write_results_to_model``).

    Args:
        model: ``Working`` after ``estimate()``.
        v_pu: (n_bus_active,) — V magnitudes in p.u. (optional).
        delta_rad: (n_bus_active,) — angles in radians (optional).

    Returns:
        ``SystemLosses`` — totals + breakdown by type + per-branch list.
    """
    network_pu = model_to_pu(model)
    n_bus = network_pu.n_bus
    n_branch = network_pu.n_branch

    if v_pu is None:
        # From model.nodes.voltage_magnitude (after write_results_to_model).
        # Vectorized read: resolve bus_ids -> rows once (missing id -> 0.0,
        # matching the former per-node get_by_id fallback).
        nodes_arr = model.nodes.to_numpy()
        rows_map = id_to_pos_map(nodes_arr["id"])
        rows = np.fromiter(
            (rows_map.get(int(nid), -1) for nid in network_pu.bus_ids.tolist()),
            dtype=np.int64,
            count=n_bus,
        )
        valid = rows >= 0
        v_kv = np.where(valid, nodes_arr["voltage_magnitude"][rows], 0.0).astype(np.float64)
        delta_arr = np.where(valid, nodes_arr["voltage_angle"][rows], 0.0).astype(np.float64)
        v_pu_arr = np.where(network_pu.bus_vn_kv > 0, v_kv / network_pu.bus_vn_kv, 1.0)
        delta = delta_arr
    else:
        v_pu_arr = np.asarray(v_pu, dtype=np.float64)
        if v_pu_arr.shape != (n_bus,):
            raise ValueError(f"v_pu length {v_pu_arr.shape} != n_bus_active {n_bus}")
        if delta_rad is None:
            raise ValueError("delta_rad is required if v_pu is given")
        delta = np.asarray(delta_rad, dtype=np.float64)

    # Complex V in p.u.
    v_complex = v_pu_arr * np.exp(1j * delta)

    # ---- Nodal shunts (in p.u., S = |V|² · conj(Y)) ----
    y_sh = network_pu.bus_g_shunt + 1j * network_pu.bus_b_shunt  # p.u.
    s_node_shunt_pu = (v_pu_arr**2) * np.conj(y_sh)
    p_node_shunt = float(s_node_shunt_pu.real.sum()) * BASE_MVA
    # Standard convention (b > 0 capacitive): Im(S) = -|V|^2 * b is negative for a
    # capacitor (it generates Q) and positive for a reactor (it absorbs Q).
    q_node_shunt = float(s_node_shunt_pu.imag.sum()) * BASE_MVA

    # ---- Branches (vectorized) ----
    if n_branch == 0:
        return SystemLosses(
            total_p_loss_mw=p_node_shunt,
            total_q_loss_mvar=q_node_shunt,
            p_line_series=0.0,
            q_line_series=0.0,
            p_trafo_series=0.0,
            q_trafo_series=0.0,
            p_branch_shunt=0.0,
            q_branch_shunt=0.0,
            p_node_shunt=p_node_shunt,
            q_node_shunt=q_node_shunt,
            per_branch=[],
        )

    # Use the same y-bus that SE builds — guarantees consistency
    # with branch.loss_p/q (written in write_results_to_model).
    _, yf, yt = build_ybus(network_pu)
    i_from = yf @ v_complex
    i_to = yt @ v_complex
    s_from = v_complex[network_pu.from_idx] * np.conj(i_from)
    s_to = v_complex[network_pu.to_idx] * np.conj(i_to)
    s_loss_total = s_from + s_to  # p.u.

    # Series part separately — for the breakdown. With the same tap.
    z_ser = network_pu.branch_r + 1j * network_pu.branch_x
    if np.any(z_ser == 0):
        raise ValueError(
            "Branches with r=x=0 in the active set; they must be switched off before SE."
        )
    # Resistance referred behind the ideal transformer (as in build_ybus).
    y_ser = series_admittance(network_pu)
    z_ser = z_ser / network_pu.tap_ratio**2
    tap_complex = network_pu.tap_ratio * np.exp(1j * network_pu.phase_shift)
    v_f = v_complex[network_pu.from_idx]
    v_t = v_complex[network_pu.to_idx]
    v_f_eff = v_f / tap_complex
    i_ser = (v_f_eff - v_t) * y_ser
    s_series = (np.abs(i_ser) ** 2) * z_ser  # p.u.

    # Shunt part — by difference (s_loss_total = s_series + s_shunt).
    s_shunt = s_loss_total - s_series

    # Per-branch transformer flag (vectorized _branch_is_trafo_pu)
    off_nominal_tap = np.abs(network_pu.tap_ratio - 1.0) > 1e-6
    shunt_one_sided = (
        (np.abs(network_pu.branch_g) < 1e-12)
        & (np.abs(network_pu.branch_b) < 1e-12)
        & ((np.abs(network_pu.branch_g_from) > 1e-12) | (np.abs(network_pu.branch_b_from) > 1e-12))
    )
    is_trafo = off_nominal_tap | shunt_one_sided

    p_ser_mw = s_series.real * BASE_MVA
    q_ser_mvar = s_series.imag * BASE_MVA
    p_sh_mw = s_shunt.real * BASE_MVA
    # Line charging (b > 0) generates Q, so the branch shunt Q is negative.
    q_sh_mvar = s_shunt.imag * BASE_MVA

    p_line_ser = float(p_ser_mw[~is_trafo].sum())
    q_line_ser = float(q_ser_mvar[~is_trafo].sum())
    p_trafo_ser = float(p_ser_mw[is_trafo].sum())
    q_trafo_ser = float(q_ser_mvar[is_trafo].sum())
    p_branch_shunt = float(p_sh_mw.sum())
    q_branch_shunt = float(q_sh_mvar.sum())

    # Per-branch list. Use the original node ids (not positions).
    from_node_ids = network_pu.bus_ids[network_pu.from_idx]
    to_node_ids = network_pu.bus_ids[network_pu.to_idx]
    per_branch: list[BranchLossRow] = []
    for i in range(n_branch):
        per_branch.append(
            BranchLossRow(
                branch_id=int(network_pu.branch_ids[i]),
                from_node=int(from_node_ids[i]),
                to_node=int(to_node_ids[i]),
                tip="trafo" if is_trafo[i] else "line",
                p_loss_series=float(p_ser_mw[i]),
                q_loss_series=float(q_ser_mvar[i]),
                p_loss_shunt=float(p_sh_mw[i]),
                q_loss_shunt=float(q_sh_mvar[i]),
                p_loss_total=float(p_ser_mw[i] + p_sh_mw[i]),
                q_loss_total=float(q_ser_mvar[i] + q_sh_mvar[i]),
            )
        )

    total_p = p_line_ser + p_trafo_ser + p_branch_shunt + p_node_shunt
    total_q = q_line_ser + q_trafo_ser + q_branch_shunt + q_node_shunt

    return SystemLosses(
        total_p_loss_mw=total_p,
        total_q_loss_mvar=total_q,
        p_line_series=p_line_ser,
        q_line_series=q_line_ser,
        p_trafo_series=p_trafo_ser,
        q_trafo_series=q_trafo_ser,
        p_branch_shunt=p_branch_shunt,
        q_branch_shunt=q_branch_shunt,
        p_node_shunt=p_node_shunt,
        q_node_shunt=q_node_shunt,
        per_branch=per_branch,
    )
