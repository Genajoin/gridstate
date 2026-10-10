"""``compute_system_losses`` on a 2-bus line with analytically derived numbers.

110 kV line, ``R = 6.05 Ohm``, ``X = 30.25 Ohm``, charging ``B = 2e-4 S``; with
``Z_base = 110^2 / 100 = 121 Ohm``: ``r = 0.05``, ``x = 0.25``,
``b = 0.0242`` p.u. State ``V1 = 1.0 /_0``, ``V2 = 0.98 /_-0.03 rad``.

* series current ``I = (V1 - V2) / (r + jx)``; series losses ``|I|^2 (r + jx)``;
* line charging ``-(|V1|^2 + |V2|^2) * b/2`` (Q generated, negative);
* a node shunt ``g + jb`` absorbs ``|V|^2 * (g - jb)``.

The values are multiplied by ``BASE_MVA = 100`` to get MW / Mvar.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.constants import BranchType, NodeType
from gridstate.losses import SystemLosses, compute_system_losses
from gridstate.working import Working


_Z_BASE = 110.0**2 / 100.0
_R, _X, _B = 6.05 / _Z_BASE, 30.25 / _Z_BASE, 2e-4 * _Z_BASE
_V = np.array([1.0, 0.98])
_DELTA = np.array([0.0, -0.03])


def _model(*, susceptance: float = 2e-4, node2_shunt_b: float = 0.0, with_line: bool = True):
    m = Working.empty()
    for nid, ntype, v, d, sb in (
        (1, NodeType.SLACK, _V[0], _DELTA[0], 0.0),
        (2, NodeType.PQ, _V[1], _DELTA[1], node2_shunt_b),
    ):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": v * 110.0,
                "voltage_angle": d,
                "status": True,
                "node_type": int(ntype),
                "shunt_b": sb,
            }
        )
    if with_line:
        m.branches.add(
            {
                "id": 5,
                "from_node": 1,
                "to_node": 2,
                "resistance": 6.05,
                "reactance": 30.25,
                "susceptance": susceptance,
                "tap_ratio": 1.0,
                "phase_shift": 0.0,
                "status": True,
                "branch_type": int(BranchType.LINE),
            }
        )
    return m


def _series_loss_mva() -> complex:
    v = _V * np.exp(1j * _DELTA)
    i_ser = (v[0] - v[1]) / (_R + 1j * _X)
    return complex(abs(i_ser) ** 2 * (_R + 1j * _X) * 100.0)


def test_series_losses_equal_i_squared_r_and_i_squared_x() -> None:
    res = compute_system_losses(_model(susceptance=0.0))
    s_ser = _series_loss_mva()
    assert res.p_line_series == pytest.approx(s_ser.real, rel=1e-10)
    assert res.q_line_series == pytest.approx(s_ser.imag, rel=1e-10)
    assert res.q_line_series / res.p_line_series == pytest.approx(_X / _R, rel=1e-10)
    assert res.p_trafo_series == 0.0
    assert res.p_branch_shunt == pytest.approx(0.0, abs=1e-10)
    assert res.total_p_loss_mw == pytest.approx(s_ser.real, rel=1e-10)
    (row,) = res.per_branch
    assert (row.branch_id, row.from_node, row.to_node, row.tip) == (5, 1, 2, "line")


def test_line_charging_is_the_shunt_part() -> None:
    res = compute_system_losses(_model())
    q_charging = -(_V[0] ** 2 + _V[1] ** 2) * _B / 2.0 * 100.0
    assert res.q_branch_shunt == pytest.approx(q_charging, rel=1e-10)
    assert res.p_branch_shunt == pytest.approx(0.0, abs=1e-10)
    # The series part does not depend on the charging.
    assert res.p_line_series == pytest.approx(_series_loss_mva().real, rel=1e-10)
    assert res.total_q_loss_mvar == pytest.approx(_series_loss_mva().imag + q_charging, rel=1e-10)


def test_explicit_state_matches_the_state_stored_in_the_model() -> None:
    m = _model(node2_shunt_b=1e-3)
    from_model = compute_system_losses(m)
    explicit = compute_system_losses(m, v_pu=_V, delta_rad=_DELTA)
    for name in ("total_p_loss_mw", "total_q_loss_mvar", "p_line_series", "q_branch_shunt"):
        assert getattr(explicit, name) == pytest.approx(getattr(from_model, name), rel=1e-12)
    # A capacitor of 1e-3 S on node 2: -|V2|^2 * b = -0.98^2 * 0.121 * 100 Mvar.
    assert explicit.q_node_shunt == pytest.approx(-(0.98**2) * 1e-3 * _Z_BASE * 100.0)


def test_explicit_state_input_checks() -> None:
    m = _model()
    with pytest.raises(ValueError, match="v_pu"):
        compute_system_losses(m, v_pu=np.ones(3), delta_rad=np.zeros(3))
    with pytest.raises(ValueError, match="delta_rad"):
        compute_system_losses(m, v_pu=_V)


def test_network_without_branches_reports_node_shunts_only() -> None:
    res = compute_system_losses(_model(with_line=False, node2_shunt_b=-2e-3))
    q_reactor = 0.98**2 * 2e-3 * _Z_BASE * 100.0  # reactor absorbs Q
    assert res.total_q_loss_mvar == pytest.approx(q_reactor)
    assert res.q_node_shunt == pytest.approx(q_reactor)
    assert res.total_p_loss_mw == 0.0
    assert res.per_branch == []


def test_zero_impedance_branch_is_rejected() -> None:
    """``R = X = 0`` is rejected (by the shared Y-bus builder, before the series split)."""
    m = _model()
    m.branches.update(5, {"resistance": 0.0, "reactance": 0.0})
    with pytest.raises(ValueError):
        compute_system_losses(m)


def test_to_dict_layout() -> None:
    res = compute_system_losses(_model(node2_shunt_b=1e-3))
    d = res.to_dict()
    assert d["total_p_loss_mw"] == res.total_p_loss_mw
    assert d["total_q_loss_mvar"] == res.total_q_loss_mvar
    assert d["breakdown"] == {
        "line_series_p": res.p_line_series,
        "line_series_q": res.q_line_series,
        "trafo_series_p": res.p_trafo_series,
        "trafo_series_q": res.q_trafo_series,
        "branch_shunt_p": res.p_branch_shunt,
        "branch_shunt_q": res.q_branch_shunt,
        "node_shunt_p": res.p_node_shunt,
        "node_shunt_q": res.q_node_shunt,
    }
    (row,) = d["per_branch"]
    br = res.per_branch[0]
    assert row == {
        "branch_id": 5,
        "from_node": 1,
        "to_node": 2,
        "tip": "line",
        "p_loss_series_mw": br.p_loss_series,
        "q_loss_series_mvar": br.q_loss_series,
        "p_loss_shunt_mw": br.p_loss_shunt,
        "q_loss_shunt_mvar": br.q_loss_shunt,
        "p_loss_total_mw": br.p_loss_total,
        "q_loss_total_mvar": br.q_loss_total,
    }
    # The breakdown adds up to the totals.
    assert isinstance(res, SystemLosses)
    assert sum(v for k, v in d["breakdown"].items() if k.endswith("_p")) == pytest.approx(
        d["total_p_loss_mw"]
    )
    assert sum(v for k, v in d["breakdown"].items() if k.endswith("_q")) == pytest.approx(
        d["total_q_loss_mvar"]
    )
