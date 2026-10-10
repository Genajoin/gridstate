"""Physics kernels of the telemetry layer with hand-checkable numbers.

* :mod:`gridstate.telemetry.loss_filter` - expected ``|Q_beg + Q_end|`` of a
  pi-model line (charging ``V^2*B`` minus series losses ``S^2/V^2*X``) and the
  consistency check built on it;
* :mod:`gridstate.telemetry.rpn` - application of the selected tap steps
  (tap ratio / phase shift, shunt rescaling) and its effect on the Y-bus;
* :mod:`gridstate.telemetry.voltage_nominal` - filling missing nominal voltages.

Units: kV, S, Ohm, MW / Mvar; ``kV^2 * S = MVA`` and ``MVA^2 / kV^2 * Ohm = MVA``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.constants import BranchType, NodeType
from gridstate.telemetry.loss_filter import (
    compute_expected_q_imbalance_mvar,
    is_branch_q_consistent_with_physics,
)
from gridstate.telemetry.rpn import _apply_tap_steps_on_arrays, apply_rpn_resolved
from gridstate.telemetry.voltage_nominal import apply_voltage_nominal_resolved
from gridstate.units import model_to_pu
from gridstate.working import Working
from gridstate.ybus import build_ybus


# ---------------------------------------------------------------------------
# loss_filter
# ---------------------------------------------------------------------------


def test_expected_q_is_charging_only_without_flow() -> None:
    """750 kV line, B = 1.895 mS: V^2*B = 750^2 * 1.895e-3 = 1065.9375 Mvar."""
    q = compute_expected_q_imbalance_mvar(vn_kv=750.0, susceptance_si=1.895e-3, reactance_si=80.0)
    assert q == pytest.approx(1065.9375, rel=1e-12)


def test_expected_q_is_series_loss_only_without_charging() -> None:
    """220 kV, X = 40 Ohm, P = 200 MW, Q = 0: S^2/V^2*X = 40000/48400*40 = 33.0579 Mvar."""
    q = compute_expected_q_imbalance_mvar(
        vn_kv=220.0, susceptance_si=0.0, reactance_si=40.0, p_typical_mw=200.0
    )
    assert q == pytest.approx(40000.0 / 48400.0 * 40.0, rel=1e-12)


def test_expected_q_combines_charging_and_losses() -> None:
    """|V^2*|B| - (P^2+Q^2)/V^2*|X||: 110 kV, B = 3e-4 S, X = -20 Ohm (sign ignored),
    P = 60 MW, Q = 80 Mvar -> |3.63 - 10000/12100*20| = |3.63 - 16.5289| = 12.8989."""
    q = compute_expected_q_imbalance_mvar(
        vn_kv=110.0,
        susceptance_si=-3e-4,
        reactance_si=-20.0,
        p_typical_mw=60.0,
        q_typical_mvar=80.0,
    )
    expected = abs(110.0**2 * 3e-4 - (60.0**2 + 80.0**2) / 110.0**2 * 20.0)
    assert q == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("vn_kv", "x_ohm"), [(0.0, 40.0), (220.0, 0.0), (220.0, 1e-13)])
def test_expected_q_drops_the_series_term_without_voltage_or_reactance(
    vn_kv: float, x_ohm: float
) -> None:
    """No nominal voltage or a (near) zero reactance: only the charging term remains."""
    q = compute_expected_q_imbalance_mvar(
        vn_kv=vn_kv, susceptance_si=1e-4, reactance_si=x_ohm, p_typical_mw=500.0
    )
    assert q == pytest.approx(vn_kv * vn_kv * 1e-4, abs=1e-15)


def test_q_consistency_keeps_a_charged_750kv_line() -> None:
    """Docstring case: |Q_obs| = 1100 vs V^2*B ~ 1066 -> within max(50, 30 % of 1066)."""
    ok, expected, observed, excess = is_branch_q_consistent_with_physics(
        q_observed_mvar=-1100.0,
        vn_kv=750.0,
        susceptance_si=1.895e-3,
        reactance_si=0.0,
    )
    assert ok
    assert expected == pytest.approx(1065.9375)
    assert observed == 1100.0
    assert excess == pytest.approx(1100.0 - 1065.9375)


def test_q_consistency_drops_an_imbalance_without_physical_cause() -> None:
    """220 kV line with B = 1e-4 S (4.84 Mvar expected): |Q_obs| = 80 -> excess 75.16 > 50."""
    ok, expected, _observed, excess = is_branch_q_consistent_with_physics(
        q_observed_mvar=80.0,
        vn_kv=220.0,
        susceptance_si=1e-4,
        reactance_si=0.0,
    )
    assert not ok
    assert expected == pytest.approx(4.84)
    assert excess == pytest.approx(80.0 - 4.84)


def test_q_consistency_tolerance_is_relative_for_large_expectations() -> None:
    """tol = max(floor, rel_pct % of expected): 30 % of 1065.94 = 319.78 Mvar."""
    common = {"vn_kv": 750.0, "susceptance_si": 1.895e-3, "reactance_si": 0.0}
    tol = 0.3 * 1065.9375
    assert is_branch_q_consistent_with_physics(q_observed_mvar=1065.9375 + tol - 1e-6, **common)[0]
    assert not is_branch_q_consistent_with_physics(
        q_observed_mvar=1065.9375 + tol + 1e-6, **common
    )[0]
    # The floor dominates for a small expectation.
    assert is_branch_q_consistent_with_physics(
        q_observed_mvar=49.0, vn_kv=110.0, susceptance_si=0.0, reactance_si=0.0
    )[0]
    assert not is_branch_q_consistent_with_physics(
        q_observed_mvar=49.0, vn_kv=110.0, susceptance_si=0.0, reactance_si=0.0, floor_mvar=10.0
    )[0]


def test_q_consistency_uses_half_the_observed_sum_as_typical_q() -> None:
    """The series-loss term takes Q_typical = Q_observed / 2 and P_typical as given:
    110 kV, X = 50 Ohm, P = 100 MW, Q_obs = 40 -> S^2 = 100^2 + 20^2 = 10400,
    expected = 10400 / 12100 * 50 = 42.975 Mvar."""
    _, expected, _, _ = is_branch_q_consistent_with_physics(
        q_observed_mvar=40.0,
        vn_kv=110.0,
        susceptance_si=0.0,
        reactance_si=50.0,
        p_typical_mw=100.0,
    )
    assert expected == pytest.approx(10400.0 / 12100.0 * 50.0, rel=1e-12)


def test_q_consistency_ignores_the_sign_of_the_imbalance() -> None:
    """Current behaviour: the check compares magnitudes only, so an observed net Q
    generation and an observed net Q absorption of the same size are both accepted
    against a charging-dominated expectation."""
    common = {"vn_kv": 750.0, "susceptance_si": 1.895e-3, "reactance_si": 0.0}
    assert is_branch_q_consistent_with_physics(q_observed_mvar=-1066.0, **common)[0]
    assert is_branch_q_consistent_with_physics(q_observed_mvar=+1066.0, **common)[0]


# ---------------------------------------------------------------------------
# rpn: tap steps
# ---------------------------------------------------------------------------


def _transformer_model(**branch_overrides: float) -> Working:
    """Two 110 kV nodes joined by a transformer branch (nominal ratio 1.0)."""
    m = Working.empty()
    for nid, ntype in ((1, NodeType.SLACK), (2, NodeType.PQ)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "status": True,
                "node_type": int(ntype),
            }
        )
    row = {
        "id": 7,
        "from_node": 1,
        "to_node": 2,
        "resistance": 2.0,
        "reactance": 40.0,
        "tap_ratio": 1.0,
        "phase_shift": 0.0,
        "status": True,
        "branch_type": int(BranchType.TRANSFORMER),
    }
    row.update(branch_overrides)
    m.branches.add(row)
    return m


def _tap_step(branch_id: int, tap: float, shift: float, factor: float) -> dict:
    return {
        "id": 0,
        "branch_id": branch_id,
        "ktr_re": tap * np.cos(shift),
        "ktr_im": tap * np.sin(shift),
        "tap_ratio": tap,
        "phase_shift": shift,
        "shunt_factor": factor,
    }


def test_tap_kernel_writes_tap_and_rescales_only_nonzero_shunts() -> None:
    m = _transformer_model(susceptance_from=-2e-5, conductance_from=1e-6, susceptance=0.0)
    m.tap_steps.add(_tap_step(7, 1.05, 0.1, 1.05**2))
    branches = m.branches.to_numpy().copy()
    stats = _apply_tap_steps_on_arrays(branches, m.tap_steps.to_numpy())
    assert stats == {"applied": 1, "shunt_recalc": 1}
    row = branches[0]
    assert row["tap_ratio"] == 1.05
    assert row["phase_shift"] == 0.1
    assert row["susceptance_from"] == pytest.approx(-2e-5 * 1.1025, rel=1e-15)
    assert row["conductance_from"] == pytest.approx(1e-6 * 1.1025, rel=1e-15)
    # Zero fields stay zero (no 0 * factor writes).
    for name in ("conductance", "susceptance", "conductance_to", "susceptance_to"):
        assert row[name] == 0.0


def test_tap_kernel_factor_one_and_unknown_branch() -> None:
    """``shunt_factor == 1`` leaves the shunt alone; an unknown branch id is skipped;
    a factor on a branch without shunts applies the tap but recalculates nothing."""
    m = _transformer_model(susceptance_from=-2e-5)
    branches = m.branches.to_numpy().copy()
    steps = np.concatenate(
        [
            _steps_array([_tap_step(7, 0.95, 0.0, 1.0)]),
            _steps_array([_tap_step(999, 1.1, 0.0, 1.21)]),
        ]
    )
    stats = _apply_tap_steps_on_arrays(branches, steps)
    assert stats == {"applied": 1, "shunt_recalc": 0}
    assert branches[0]["tap_ratio"] == 0.95
    assert branches[0]["susceptance_from"] == -2e-5

    bare = _transformer_model().branches.to_numpy().copy()
    stats = _apply_tap_steps_on_arrays(bare, _steps_array([_tap_step(7, 1.1, 0.0, 1.21)]))
    assert stats == {"applied": 1, "shunt_recalc": 0}
    assert bare[0]["tap_ratio"] == 1.1


def _steps_array(rows: list[dict]) -> np.ndarray:
    coll = Working.empty().tap_steps
    for r in rows:
        coll.add(r)
    return coll.to_numpy()


def test_apply_rpn_without_tap_steps_is_a_noop() -> None:
    m = _transformer_model()
    before = m.branches.to_numpy().copy()
    assert apply_rpn_resolved(m) == {"applied": 0, "shunt_recalc": 0}
    np.testing.assert_array_equal(m.branches.to_numpy(), before)

    class _NoDomainTables:
        branches = m.branches

    assert apply_rpn_resolved(_NoDomainTables()) == {"applied": 0, "shunt_recalc": 0}  # type: ignore[arg-type]


def test_apply_rpn_keeps_the_tap_referred_from_shunt_invariant() -> None:
    """Y-bus effect of a tap change 1.0 -> 1.05 with ``shunt_factor = 1.05^2``.

    In ``build_ybus`` the from-shunt enters ``Yff`` divided by ``|t|^2`` and the
    series admittance referred to the from side is ``Ysf = 1/(r+jx)``:
    ``Yff = Ysf + Yc_from/|t|^2``, ``Yft = -Ysf*t``, ``Ytt = Ysf*|t|^2``. With the
    shunt rescaled by ``|t|^2`` the shunt part of ``Yff`` stays the same.
    """
    b_from = -4e-5  # S, magnetising susceptance on the from side
    m = _transformer_model(susceptance_from=b_from)
    y0 = build_ybus(model_to_pu(m))[0].toarray()

    m.tap_steps.add(_tap_step(7, 1.05, 0.0, 1.05**2))
    stats = apply_rpn_resolved(m)
    assert stats == {"applied": 1, "shunt_recalc": 1}
    row = m.branches.get_by_id(7)
    assert row is not None
    assert row.tap_ratio == 1.05
    assert row.susceptance_from == pytest.approx(b_from * 1.1025)

    y1 = build_ybus(model_to_pu(m))[0].toarray()
    z_base = 110.0**2 / 100.0
    ysf = z_base / (2.0 + 40.0j)
    yc_from_pu = 1j * b_from * z_base
    t = 1.05
    assert y0[0, 0] == pytest.approx(ysf + yc_from_pu, rel=1e-12)
    assert y1[0, 0] == pytest.approx(ysf + yc_from_pu, rel=1e-12)  # invariant
    assert y1[0, 1] == pytest.approx(-ysf * t, rel=1e-12)
    assert y1[1, 0] == pytest.approx(-ysf * t, rel=1e-12)
    assert y1[1, 1] == pytest.approx(ysf * t * t, rel=1e-12)


# ---------------------------------------------------------------------------
# voltage_nominal
# ---------------------------------------------------------------------------


def test_voltage_nominal_fills_only_missing_values() -> None:
    m = Working.empty()
    for nid, vn in ((1, 110.0), (2, 0.0), (3, 0.0), (4, -1.0)):
        m.nodes.add({"id": nid, "voltage_nominal": vn, "status": True})
    stats = apply_voltage_nominal_resolved(m, {1: 220.0, 2: 35.0, 4: 10.0, 99: 500.0})
    assert stats == {"applied": 2, "already_set": 1, "missing": 1, "total": 4}
    vn = {int(r["id"]): float(r["voltage_nominal"]) for r in m.nodes.to_numpy()}
    # Node 1 already had a value (not overwritten); 3 has no entry in the map;
    # node 4 had a non-positive value, which counts as missing.
    assert vn == {1: 110.0, 2: 35.0, 3: 0.0, 4: 10.0}


def test_voltage_nominal_empty_map_and_empty_table() -> None:
    m = Working.empty()
    m.nodes.add({"id": 1, "voltage_nominal": 0.0, "status": True})
    assert apply_voltage_nominal_resolved(m, {}) == {
        "applied": 0,
        "already_set": 0,
        "missing": 1,
        "total": 1,
    }
    assert apply_voltage_nominal_resolved(Working.empty(), {1: 110.0}) == {
        "applied": 0,
        "already_set": 0,
        "missing": 0,
        "total": 0,
    }
