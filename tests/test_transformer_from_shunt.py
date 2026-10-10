"""Transformer branch shunts: the "from" shunt sits at the "from" node.

Branch layout (see ``gridstate.ybus``): the series impedance at the "from" end,
the ideal transformer ``t : 1`` at the "to" end, the whole "from" shunt at the
"from" node in the "from"-side base (before the ideal transformer, independent
of the tap), the "to" shunt at the "to" node::

    from ──┬── r + j·x ──( t : 1 )──┬── to
         Yc_from                   Yc_to

The network below has transformers with off-nominal taps, a phase shift,
different nominal voltages on the two ends and non-zero shunts on both ends.
The branch flows, currents and injections of the measurement function ``h(x)``
are compared with a direct evaluation of this circuit, the analytic Jacobian
with finite differences of ``h(x)``, and the loss breakdown with the shunt
powers of the circuit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gridstate.algebra.base import BaseAlgebra
from gridstate.constants import BranchType, NodeType
from gridstate.losses import compute_system_losses
from gridstate.state import StateLayout, pack, unpack
from gridstate.units import BASE_MVA, NetworkPU, compute_branch_results_pu, model_to_pu
from gridstate.working import Working
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
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
    build_z_and_r,
)


_NODES = ((1, 220.0, NodeType.SLACK), (2, 110.0, NodeType.PQ), (3, 110.0, NodeType.PQ))

_TRAFO = int(BranchType.TRANSFORMER)

# Branches: series R/X in Ohm on the from side, physical ratio HV:LV, shunts in S
# (from-side base). Both transformers have an off-nominal ratio in p.u.
_BRANCHES: tuple[dict[str, Any], ...] = (
    {
        "id": 10,
        "from_node": 1,
        "to_node": 2,
        "branch_type": _TRAFO,
        "resistance": 2.0,
        "reactance": 40.0,
        "tap_ratio": 2.0 * 1.06,
        "phase_shift": 0.03,
        "conductance_from": 2e-6,
        "susceptance_from": -1.5e-5,
        "conductance_to": 1e-6,
        "susceptance_to": -4e-6,
    },
    {
        "id": 20,
        "from_node": 2,
        "to_node": 3,
        "branch_type": int(BranchType.LINE),
        "resistance": 5.0,
        "reactance": 20.0,
        "tap_ratio": 1.0,
        "susceptance": 1e-4,
    },
    {
        "id": 30,
        "from_node": 1,
        "to_node": 3,
        "branch_type": _TRAFO,
        "resistance": 3.0,
        "reactance": 55.0,
        "tap_ratio": 2.0 * 0.94,
        "conductance_from": 3e-6,
        "susceptance_from": -2.5e-5,
    },
)


def _network(*, tap_scale: float = 1.0) -> Working:
    m = Working.empty()
    for node_id, vn, node_type in _NODES:
        m.nodes.add(
            {"id": node_id, "voltage_nominal": vn, "status": True, "node_type": int(node_type)}
        )
    for branch in _BRANCHES:
        row = {"status": True, **branch}
        if row["branch_type"] == _TRAFO:
            row["tap_ratio"] *= tap_scale
        m.branches.add(row)
    mid = 1
    for node_id, _, _ in _NODES:
        kinds = [KIND_VOLTAGE]
        if node_id != 1:
            kinds += [KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q]
        for kind in kinds:
            m.measurements.add(_meas(mid, OBJ_NODE, node_id, kind, -1))
            mid += 1
    for bid in (branch["id"] for branch in _BRANCHES):
        for side in (SIDE_FROM, SIDE_TO):
            for kind in (KIND_POWER_P, KIND_POWER_Q, KIND_CURRENT):
                m.measurements.add(_meas(mid, OBJ_BRANCH, bid, kind, side))
                mid += 1
    return m


def _meas(mid: int, obj_type: int, obj_id: int, kind: int, side: int) -> dict[str, Any]:
    return {
        "id": mid,
        "object_type": obj_type,
        "object_id": obj_id,
        "measurement_type": kind,
        "branch_side": side,
        "value": 1.0,
        "variance": 1.0,
        "status": True,
        "quality": 0,
    }


def _circuit_currents(pu: NetworkPU, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Branch-end currents (p.u., into the branch) of the circuit drawn above."""
    f, t = pu.from_idx, pu.to_idx
    z = pu.branch_r + 1j * pu.branch_x
    ratio = pu.tap_ratio * np.exp(1j * pu.phase_shift)
    half = 0.5 * (pu.branch_g + 1j * pu.branch_b)
    yc_from = pu.branch_g_from + 1j * pu.branch_b_from + half
    yc_to = pu.branch_g_to + 1j * pu.branch_b_to + half
    i_series = (v[f] - ratio * v[t]) / z
    i_from = i_series + yc_from * v[f]
    i_to = -np.conj(ratio) * i_series + yc_to * v[t]
    return i_from, i_to


def _state(pu: NetworkPU) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([1.03, 0.97, 0.985])[: pu.n_bus]
    delta = np.array([0.0, -0.06, -0.045])[: pu.n_bus]
    return v, delta


def _algebra(m: Working) -> tuple[BaseAlgebra, NetworkPU, StateLayout, Any]:
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    _z, _r, idx = build_z_and_r(m, m.measurements, pu)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    return BaseAlgebra(ybus, yf, yt, idx, layout, pu), pu, layout, idx


def test_network_exercises_off_nominal_taps_and_from_shunts() -> None:
    pu = model_to_pu(_network())
    trafo = np.isin(pu.branch_ids, [10, 30])
    assert np.all(np.abs(pu.tap_ratio[trafo] - 1.0) > 0.05)
    assert np.all(pu.branch_b_from[trafo] != 0)


@pytest.mark.parametrize("tap_scale", [0.9, 1.0, 1.1])
def test_from_shunt_does_not_depend_on_the_tap(tap_scale: float) -> None:
    """``Yff = 1/z + Yc_from`` for any tap: the "from" shunt is not divided by |t|²."""
    pu = model_to_pu(_network(tap_scale=tap_scale))
    _, yf, yt = build_ybus(pu)
    rows = np.arange(pu.n_branch)
    yff = yf.toarray()[rows, pu.from_idx]
    ytt = yt.toarray()[rows, pu.to_idx]
    half = 0.5 * (pu.branch_g + 1j * pu.branch_b)
    yc_from = pu.branch_g_from + 1j * pu.branch_b_from + half
    yc_to = pu.branch_g_to + 1j * pu.branch_b_to + half
    ysf = 1.0 / (pu.branch_r + 1j * pu.branch_x)
    np.testing.assert_allclose(yff, ysf + yc_from, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(ytt, ysf * pu.tap_ratio**2 + yc_to, rtol=1e-13, atol=0.0)


def test_h_matches_the_circuit() -> None:
    """Branch P/Q/|I| at both ends and nodal injections of ``h(x)`` equal the
    direct evaluation of the circuit; the nodal injection is the sum of the
    branch-end powers (the Y-bus and Yf/Yt agree)."""
    algebra, pu, _layout, idx = _algebra(_network())
    v, delta = _state(pu)
    h = algebra.evaluate_h(v, delta)

    vc = v * np.exp(1j * delta)
    i_from, i_to = _circuit_currents(pu, vc)
    s_from = vc[pu.from_idx] * np.conj(i_from)
    s_to = vc[pu.to_idx] * np.conj(i_to)
    s_bus = np.zeros(pu.n_bus, dtype=complex)
    np.add.at(s_bus, pu.from_idx, s_from)
    np.add.at(s_bus, pu.to_idx, s_to)

    expected = np.empty_like(h)
    for k in range(len(h)):
        kind, pos, side = int(idx.kind[k]), int(idx.object_pos[k]), int(idx.branch_side[k])
        if int(idx.object_kind[k]) == OBJ_NODE:
            expected[k] = {
                KIND_VOLTAGE: v[pos],
                KIND_POWER_INJECTION_P: s_bus[pos].real,
                KIND_POWER_INJECTION_Q: s_bus[pos].imag,
            }[kind]
            continue
        s = s_from[pos] if side == SIDE_FROM else s_to[pos]
        i = i_from[pos] if side == SIDE_FROM else i_to[pos]
        expected[k] = {KIND_POWER_P: s.real, KIND_POWER_Q: s.imag, KIND_CURRENT: abs(i)}[kind]
    assert len(h) == 3 + 2 * 2 + len(_BRANCHES) * 6
    np.testing.assert_allclose(h, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("path", ["pattern", "generic"])
def test_jacobian_matches_finite_differences(path: str) -> None:
    algebra, pu, layout, _idx = _algebra(_network())
    v, delta = _state(pu)
    e = pack(delta, v, layout)
    if path == "pattern":
        jac = algebra.evaluate_jacobian(v, delta).toarray()
    else:
        jac = algebra._evaluate_jacobian_generic(v, delta).toarray()

    eps = 1e-6
    numeric = np.zeros_like(jac)
    for j in range(e.size):
        hi, lo = e.copy(), e.copy()
        hi[j] += eps
        lo[j] -= eps
        d_hi, v_hi = unpack(hi, layout)
        d_lo, v_lo = unpack(lo, layout)
        numeric[:, j] = (algebra.evaluate_h(v_hi, d_hi) - algebra.evaluate_h(v_lo, d_lo)) / (
            2 * eps
        )
    np.testing.assert_allclose(jac, numeric, rtol=1e-6, atol=1e-7)


def test_branch_results_and_losses_match_the_circuit() -> None:
    """Branch write-back flows and the loss breakdown: the shunt part of a
    transformer is ``|V_from|²·conj(Yc_from) + |V_to|²·conj(Yc_to)``."""
    m = _network()
    pu = model_to_pu(m)
    _, yf, yt = build_ybus(pu)
    v, delta = _state(pu)
    vc = v * np.exp(1j * delta)
    i_from, i_to = _circuit_currents(pu, vc)
    s_from = vc[pu.from_idx] * np.conj(i_from) * BASE_MVA
    s_to = vc[pu.to_idx] * np.conj(i_to) * BASE_MVA

    p_from, q_from, p_to, q_to, *_ = compute_branch_results_pu(v, delta, pu, yf, yt)
    np.testing.assert_allclose(p_from + 1j * q_from, s_from, rtol=1e-12)
    np.testing.assert_allclose(p_to + 1j * q_to, s_to, rtol=1e-12)

    half = 0.5 * (pu.branch_g + 1j * pu.branch_b)
    yc_from = pu.branch_g_from + 1j * pu.branch_b_from + half
    yc_to = pu.branch_g_to + 1j * pu.branch_b_to + half
    s_shunt = (
        np.abs(vc[pu.from_idx]) ** 2 * np.conj(yc_from)
        + np.abs(vc[pu.to_idx]) ** 2 * np.conj(yc_to)
    ) * BASE_MVA

    losses = compute_system_losses(m, v_pu=v, delta_rad=delta)
    by_id = {row.branch_id: row for row in losses.per_branch}
    for k, bid in enumerate(pu.branch_ids.tolist()):
        row = by_id[bid]
        assert row.p_loss_total == pytest.approx((s_from[k] + s_to[k]).real, rel=1e-10)
        assert row.p_loss_shunt == pytest.approx(s_shunt[k].real, rel=1e-9, abs=1e-12)
        assert row.q_loss_shunt == pytest.approx(s_shunt[k].imag, rel=1e-9, abs=1e-12)
