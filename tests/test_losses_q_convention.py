"""Sign convention of the reactive losses in ``compute_system_losses``.

The library uses the standard admittance convention everywhere: ``Y = g + jb``,
``b > 0`` capacitive (capacitor bank, line charging), ``b < 0`` inductive (shunt
reactor). The loss report shows the power absorbed by each element, so a
capacitive element *generates* Q (negative reactive "losses") and a reactor
absorbs it (positive). No sign flip is applied; the totals therefore satisfy
the Kirchhoff balance against the nodal injections of the solved state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridstate.constants import BranchType, NodeType
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.losses import compute_system_losses
from gridstate.pipeline import PipelineConfig
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.working import Working
from gridstate.ybus import build_ybus


_DATA_DIR = Path(__file__).parent / "test_data" / "ieee"


def _node_model(shunt_b: float) -> Working:
    m = Working.empty()
    # SLACK -- model_to_pu requires exactly one slack node.
    m.nodes.add(
        {
            "id": 1,
            "name": "N1",
            "voltage_nominal": 500.0,
            "voltage_magnitude": 500.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
            "shunt_g": 0.0,
            "shunt_b": shunt_b,
        }
    )
    return m


def _line_model(susceptance: float) -> Working:
    """Two flat-voltage 110 kV nodes joined by a line with the given charging B."""
    m = Working.empty()
    for nid, ntype in ((1, NodeType.SLACK), (2, NodeType.PQ)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
            }
        )
    m.branches.add(
        {
            "id": 1,
            "from_node": 1,
            "to_node": 2,
            "resistance": 1.0,
            "reactance": 10.0,
            "susceptance": susceptance,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    return m


def test_inductive_shunt_reactor_absorbs_positive():
    """Reactor (b < 0) absorbs Q: q_node_shunt > 0."""
    r = compute_system_losses(_node_model(-50.0))
    assert r.q_node_shunt > 0, f"reactor must absorb (>0), got {r.q_node_shunt}"


def test_capacitive_shunt_generates_negative():
    """Capacitor bank (b > 0) generates Q: q_node_shunt < 0."""
    r = compute_system_losses(_node_model(50.0))
    assert r.q_node_shunt < 0, f"capacitor must generate (<0), got {r.q_node_shunt}"


def test_node_shunt_q_value_and_symmetry():
    """Q = -b * V^2 exactly (flat 500 kV, V = Vn), antisymmetric in b."""
    b = 50e-6  # S
    cap = compute_system_losses(_node_model(b)).q_node_shunt
    ind = compute_system_losses(_node_model(-b)).q_node_shunt
    # b [S] * V^2 [kV^2] = Mvar directly.
    assert cap == pytest.approx(-b * 500.0**2, rel=1e-12)
    assert ind == pytest.approx(b * 500.0**2, rel=1e-12)


def test_line_charging_generates_negative_q():
    """Line charging (susceptance > 0) generates Q: branch shunt Q < 0."""
    cap = compute_system_losses(_line_model(2e-4))
    assert cap.q_branch_shunt < 0, cap.q_branch_shunt
    assert cap.per_branch[0].q_loss_shunt < 0
    # Flat voltage: all of the branch shunt Q is -B * V^2.
    assert cap.q_branch_shunt == pytest.approx(-2e-4 * 110.0**2, rel=1e-9)
    # A negative B (inductive) flips the sign and nothing else.
    ind = compute_system_losses(_line_model(-2e-4))
    assert ind.q_branch_shunt == pytest.approx(2e-4 * 110.0**2, rel=1e-9)


@pytest.mark.parametrize("case", ["case14", "case30", "case118"])
def test_total_losses_equal_sum_of_injections(case: str) -> None:
    """KCL balance on a solved IEEE case.

    Total P/Q losses (series + branch shunts + node shunts) must equal the sum
    of the complex nodal injections ``V_i * conj((Ybus V)_i)`` of the solved
    state, to numerical precision.
    """
    se_in = load_se_input_npz(_DATA_DIR / f"{case}.npz")
    out = run_se(se_in, config=PipelineConfig(algorithm="wls"), validate=False)
    assert out.success

    model = se_in.model
    losses = compute_system_losses(model)

    network_pu = model_to_pu(model)
    nodes = model.nodes.to_numpy()
    row_of = {int(i): k for k, i in enumerate(nodes["id"])}
    rows = np.array([row_of[int(i)] for i in network_pu.bus_ids])
    v_pu = nodes["voltage_magnitude"][rows] / network_pu.bus_vn_kv
    v_complex = v_pu * np.exp(1j * nodes["voltage_angle"][rows])
    ybus, _, _ = build_ybus(network_pu)
    s_inj = v_complex * np.conj(ybus @ v_complex)
    sum_p = float(s_inj.real.sum()) * BASE_MVA
    sum_q = float(s_inj.imag.sum()) * BASE_MVA

    assert losses.total_p_loss_mw == pytest.approx(sum_p, abs=1e-9)
    assert losses.total_q_loss_mvar == pytest.approx(sum_q, abs=1e-9)

    # Breakdown parts add up to the totals.
    parts_q = (
        losses.q_line_series + losses.q_trafo_series + losses.q_branch_shunt + losses.q_node_shunt
    )
    assert parts_q == pytest.approx(losses.total_q_loss_mvar, abs=1e-9)


def test_case14_capacitor_is_not_a_consumer() -> None:
    """case14 has a 19 Mvar capacitor at a bus: the node-shunt Q is negative."""
    se_in = load_se_input_npz(_DATA_DIR / "case14.npz")
    run_se(se_in, config=PipelineConfig(algorithm="wls"), validate=False)
    losses = compute_system_losses(se_in.model)
    assert losses.q_node_shunt < 0
    # 19 Mvar at nominal voltage, scaled by V^2 of the solved state.
    assert -25.0 < losses.q_node_shunt < -17.0
