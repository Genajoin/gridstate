"""Line / transformer split of the series losses in ``compute_system_losses``.

The contract column ``branch_type`` decides which branches count as
transformers. A heuristic on the p.u. parameters (off-nominal tap, one-sided
shunt) is only a fallback for branch tables without that column: it cannot see
a transformer with a nominal tap, and line charging split between the two
sides via ``*_from`` / ``*_to`` must not be mistaken for a magnetising branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.lib import recfunctions as rfn

from gridstate.constants import BranchType, NodeType
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.losses import SystemLosses, _branch_is_trafo, compute_system_losses
from gridstate.pipeline import PipelineConfig
from gridstate.units import model_to_pu
from gridstate.working import Working


_DATA_DIR = Path(__file__).parent / "test_data" / "ieee"

# (case, number of active branches, number of transformers by branch_type)
_CASES = [("case14", 20, 5), ("case118", 186, 13)]


def _solved_losses(case: str) -> tuple[Working, SystemLosses]:
    se_in = load_se_input_npz(_DATA_DIR / f"{case}.npz")
    out = run_se(se_in, config=PipelineConfig(algorithm="wls"), validate=False)
    assert out.success
    model = out.result.model  # the solved state is written to the output model
    return model, compute_system_losses(model)


def _type_mask(model: Working, branch_ids: np.ndarray) -> np.ndarray:
    br = model.branches.to_numpy()
    type_by_id = {int(i): int(t) for i, t in zip(br["id"], br["branch_type"], strict=True)}
    return np.array([type_by_id[int(b)] == BranchType.TRANSFORMER for b in branch_ids])


@pytest.mark.parametrize(("case", "n_branch", "n_trafo"), _CASES)
def test_trafo_mask_equals_branch_type(case: str, n_branch: int, n_trafo: int) -> None:
    model, losses = _solved_losses(case)
    ids = np.array([r.branch_id for r in losses.per_branch])
    reported = np.array([r.tip == "trafo" for r in losses.per_branch])
    expected = _type_mask(model, ids)

    assert len(ids) == n_branch
    assert int(expected.sum()) == n_trafo
    np.testing.assert_array_equal(reported, expected)
    assert {r.tip for r in losses.per_branch} == {"line", "trafo"}

    # The direct helper agrees with the report, in NetworkPU order.
    network_pu = model_to_pu(model)
    np.testing.assert_array_equal(
        _branch_is_trafo(model.branches.to_numpy(), network_pu),
        _type_mask(model, network_pu.branch_ids),
    )


@pytest.mark.parametrize(("case", "n_branch", "n_trafo"), _CASES)
def test_series_split_follows_branch_type(case: str, n_branch: int, n_trafo: int) -> None:
    model, losses = _solved_losses(case)
    ids = np.array([r.branch_id for r in losses.per_branch])
    is_trafo = _type_mask(model, ids)
    p_ser = np.array([r.p_loss_series for r in losses.per_branch])
    q_ser = np.array([r.q_loss_series for r in losses.per_branch])

    assert losses.p_trafo_series == pytest.approx(p_ser[is_trafo].sum(), abs=1e-9)
    assert losses.q_trafo_series == pytest.approx(q_ser[is_trafo].sum(), abs=1e-9)
    assert losses.p_line_series == pytest.approx(p_ser[~is_trafo].sum(), abs=1e-9)
    assert losses.q_line_series == pytest.approx(q_ser[~is_trafo].sum(), abs=1e-9)

    # The split never changes the totals.
    parts_p = (
        losses.p_line_series + losses.p_trafo_series + losses.p_branch_shunt + losses.p_node_shunt
    )
    assert parts_p == pytest.approx(losses.total_p_loss_mw, abs=1e-9)


def test_case118_active_series_losses_sit_in_the_lines() -> None:
    """case118: 173 charged lines and 13 low-resistance transformers.

    Nearly all active series losses are in the lines; the transformers carry a
    sizeable share of the reactive series losses only.
    """
    _, losses = _solved_losses("case118")
    assert losses.p_line_series > 100.0
    assert losses.p_trafo_series < 1.0
    assert losses.q_trafo_series > 50.0
    assert losses.p_line_series > 100.0 * losses.p_trafo_series


# ---------------------------------------------------------------------------
# Synthetic 2-bus networks
# ---------------------------------------------------------------------------


def _two_bus(branch: dict) -> Working:
    m = Working.empty()
    for nid, ntype, v, d in ((1, NodeType.SLACK, 1.0, 0.0), (2, NodeType.PQ, 0.98, -0.03)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": v * 110.0,
                "voltage_angle": d,
                "status": True,
                "node_type": int(ntype),
            }
        )
    row = {
        "id": 7,
        "from_node": 1,
        "to_node": 2,
        "resistance": 1.21,
        "reactance": 12.1,
        "tap_ratio": 1.0,
        "phase_shift": 0.0,
        "status": True,
    }
    row.update(branch)
    m.branches.add(row)
    return m


def test_nominal_tap_transformer_is_reported_as_trafo() -> None:
    """A transformer with tap 1 and no magnetising branch counts by its type."""
    res = compute_system_losses(_two_bus({"branch_type": int(BranchType.TRANSFORMER)}))
    (row,) = res.per_branch
    assert row.tip == "trafo"
    assert res.p_line_series == 0.0
    assert res.p_trafo_series == pytest.approx(row.p_loss_series)


@pytest.mark.parametrize("btype", [BranchType.LINE, BranchType.REACTOR])
def test_off_nominal_tap_non_transformer_is_reported_as_line(btype: BranchType) -> None:
    """``branch_type`` wins over the tap value."""
    res = compute_system_losses(_two_bus({"branch_type": int(btype), "tap_ratio": 1.05}))
    (row,) = res.per_branch
    assert row.tip == "line"
    assert res.p_trafo_series == 0.0


def _heuristic_flag(branch: dict) -> bool:
    """Flag from a branch table with the ``branch_type`` column removed."""
    model = _two_bus({"branch_type": int(BranchType.LINE), **branch})
    branches = rfn.drop_fields(model.branches.to_numpy(), "branch_type", usemask=False)
    assert "branch_type" not in (branches.dtype.names or ())
    (flag,) = _branch_is_trafo(branches, model_to_pu(model)).tolist()
    return bool(flag)


def test_fallback_heuristic_without_branch_type_column() -> None:
    # Plain line with total charging.
    assert not _heuristic_flag({"susceptance": 2e-4})
    # Line charging split between the two sides (as the pandapower adapter writes it).
    assert not _heuristic_flag({"susceptance_from": 1e-4, "susceptance_to": 1e-4})
    assert not _heuristic_flag({"conductance_from": 1e-6, "conductance_to": 1e-6})
    # Magnetising branch on the "from" side only.
    assert _heuristic_flag({"conductance_from": 1e-6, "susceptance_from": -1e-5})
    # Off-nominal tap.
    assert _heuristic_flag({"tap_ratio": 1.05})
    # The heuristic's blind spot: nominal tap, no shunt -> taken for a line.
    assert not _heuristic_flag({})
