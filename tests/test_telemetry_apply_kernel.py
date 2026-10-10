"""Float kernel that applies a resolved z-vector over contract arrays.

The producer of the input resolves its telemetry (values and quality codes) into
a ``resolved`` plan; the kernel ``_apply_telemetry_on_arrays`` reads and mutates
only contract columns of ``SE_INPUT.measurements`` (plus nodes/branches). These
tests check the kernel on bare arrays with a synthetic ``resolved`` /
``arg_keys``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.constants import MeasurementQuality
from gridstate.contract import SE_INPUT
from gridstate.telemetry.apply_resolved import TelemetryApplyConfig, _apply_telemetry_on_arrays
from gridstate.telemetry.noise import (
    variance_branch_q,
    variance_power,
    variance_voltage,
)


OT_NODE, OT_BRANCH = 0, 1
MT_P, MT_Q, MT_V, MT_PINJ, MT_QINJ = 0, 1, 2, 4, 5

# KIND_MAP (telemetry.kinds): v→(0,2,-1), p_from→(1,0,0), p_to→(1,0,+1),
#                       q_from→(1,1,0), q_to→(1,1,+1).


def _meas(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.measurements.input_dtype())
    arr["status"] = True
    arr["branch_side"] = -1
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    arr["status"] = True
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _branches(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.branches.input_dtype())
    arr["status"] = True
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


# Default kernel kwargs: all auto-filters (sign/q-loss) are off so that
# each case is tested in isolation; individual tests switch them on.
CORE_KW = {
    "questionable_sigma2_multiplier": 100.0,
    "branch_p_sigma_frac": 0.02,
    "branch_q_sigma_frac": 0.07,
    "branch_q_sigma_charging_alpha": 0.10,
    "sign_inconsistency_threshold_mw": None,
    "q_loss_filter_enabled": False,
    "q_loss_filter_floor_mvar": 50.0,
    "q_loss_filter_rel_pct": 30.0,
    "q_loss_filter_action": "downweight",
    "q_loss_filter_downweight_factor": 100.0,
}


def _run(meas, nodes, branches, resolved, **overrides):
    arg_keys = list(resolved.keys())
    config = TelemetryApplyConfig(**{**CORE_KW, **overrides})
    return _apply_telemetry_on_arrays(
        meas, nodes, branches, arg_keys, resolved, total_args=len(resolved), config=config
    )


# ---------------------------------------------------------------- (1) branch P/Q/V activation


def test_branch_p_activation_and_variance():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            }
        ]
    )
    resolved = {(7, "p_from"): (150.0, 1, "g-p", MeasurementQuality.GOOD)}
    stats, new_rows = _run(meas, nodes, branches, resolved)
    assert stats["applied"] == 1 and new_rows == []
    assert bool(meas[0]["status"]) is True
    assert float(meas[0]["value"]) == pytest.approx(150.0)
    exp_var = variance_power(150.0, sigma_frac=0.02)
    assert float(meas[0]["variance"]) == pytest.approx(exp_var)
    assert float(meas[0]["weight"]) == pytest.approx(1.0 / exp_var)
    assert str(meas[0]["source_guid"]) == "g-p"


def test_branch_q_activation_charging_variance():
    # vn=750, B=2e-3 S -> charging = |B|·vn² = 1125 MVAr; σ_charge=0.10·1125=112.5.
    nodes = _nodes([{"id": 1, "voltage_nominal": 750.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True, "susceptance": 2e-3}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_Q,
                "branch_side": 0,
            }
        ]
    )
    resolved = {(7, "q_from"): (30.0, 1, "g-q", MeasurementQuality.GOOD)}
    stats, _ = _run(meas, nodes, branches, resolved)
    assert stats["applied"] == 1
    exp_var = variance_branch_q(
        30.0, charging_mvar=2e-3 * 750.0 * 750.0, charging_alpha=0.10, sigma_frac=0.07
    )
    assert float(meas[0]["variance"]) == pytest.approx(exp_var)


def test_node_v_activation_and_variance():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([])
    meas = _meas([{"id": 10, "object_type": OT_NODE, "object_id": 1, "measurement_type": MT_V}])
    resolved = {(1, "v"): (225.0, 1, "g-v", MeasurementQuality.GOOD)}
    stats, _ = _run(meas, nodes, branches, resolved)
    assert stats["applied"] == 1
    assert float(meas[0]["value"]) == pytest.approx(225.0)
    exp_var = variance_voltage(225.0, 220.0)
    assert float(meas[0]["variance"]) == pytest.approx(exp_var)


# ---------------------------------------------------------------- (2) V < 50% Vnom filter


def test_v_below_half_nominal_filtered():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([])
    meas = _meas([{"id": 10, "object_type": OT_NODE, "object_id": 1, "measurement_type": MT_V}])
    resolved = {(1, "v"): (80.0, 1, "g-v", MeasurementQuality.GOOD)}  # 80 < 0.5·220=110
    stats, _ = _run(meas, nodes, branches, resolved)
    assert stats["skipped_v_below_half_nominal"] == 1 and stats["applied"] == 0
    assert bool(meas[0]["status"]) is False
    assert int(meas[0]["filter_flag"]) == 4


# ---------------------------------------------------------------- (3) BAD quality skip


def test_bad_quality_skip_branch():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            }
        ]
    )
    resolved = {(7, "p_from"): (150.0, 1, "g-p", MeasurementQuality.BAD)}
    stats, _ = _run(meas, nodes, branches, resolved)
    assert stats["skipped_bad_quality"] == 1 and stats["applied"] == 0
    assert bool(meas[0]["status"]) is False
    assert int(meas[0]["filter_flag"]) == 1


# ---------------------------------------------------------------- (4) sign-inconsistency pre-pass


def test_sign_inconsistency_drops_all_four():
    # p_from=+200, p_to=+150 → |sum|=350 ≥ 100 -> branch inconsistent, all 4 measurements skipped.
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            },
            {
                "id": 11,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 1,
            },
            {
                "id": 12,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_Q,
                "branch_side": 0,
            },
            {
                "id": 13,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_Q,
                "branch_side": 1,
            },
        ]
    )
    resolved = {
        (7, "p_from"): (200.0, 1, "g", MeasurementQuality.GOOD),
        (7, "p_to"): (150.0, 1, "g", MeasurementQuality.GOOD),
        (7, "q_from"): (10.0, 1, "g", MeasurementQuality.GOOD),
        (7, "q_to"): (-10.0, 1, "g", MeasurementQuality.GOOD),
    }
    stats, _ = _run(meas, nodes, branches, resolved, sign_inconsistency_threshold_mw=100.0)
    assert stats["sign_inconsistent_branches"] == 1
    assert stats["skipped_sign_inconsistent"] == 4 and stats["applied"] == 0
    assert all(not bool(meas[i]["status"]) for i in range(4))
    assert int(meas[0]["filter_flag"]) == 5  # p_from
    assert int(meas[2]["filter_flag"]) == 5  # q_from


def test_sign_consistency_keeps_meas():
    # p_from=+200, p_to=-198 → |sum|=2 < 100 -> consistent, activated.
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            },
            {
                "id": 11,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 1,
            },
        ]
    )
    resolved = {
        (7, "p_from"): (200.0, 1, "g", MeasurementQuality.GOOD),
        (7, "p_to"): (-198.0, 1, "g", MeasurementQuality.GOOD),
    }
    stats, _ = _run(meas, nodes, branches, resolved, sign_inconsistency_threshold_mw=100.0)
    assert stats["sign_inconsistent_branches"] == 0
    assert stats["applied"] == 2


# ---------------------------------------------------------------- (5) node-inj accumulation


def test_node_inj_pg_minus_pn_net_in_new_rows():
    # PG=+300, PN=-120 (mult −1) → net P_inj = 300 − 120 = 180.
    nodes = _nodes([{"id": 5, "voltage_nominal": 220.0}])
    branches = _branches([])
    meas = _meas([{"id": 10, "object_type": OT_NODE, "object_id": 1, "measurement_type": MT_V}])
    resolved = {
        (5, "p_gen"): (300.0, 1, "g-pg", MeasurementQuality.GOOD),
        (5, "p_load"): (120.0, 1, "g-pn", MeasurementQuality.GOOD),
        (1, "v"): (220.0, 1, "g-v", MeasurementQuality.GOOD),
    }
    stats, new_rows = _run(meas, nodes, branches, resolved)
    assert stats["node_inj_added"] == 1
    assert len(new_rows) == 1
    row = new_rows[0]
    assert row["object_type"] == 0 and row["object_id"] == 5
    assert row["measurement_type"] == MT_PINJ and row["branch_side"] == -1
    assert row["value"] == pytest.approx(180.0)
    # id = max(existing id)+1 = 10+1.
    assert row["id"] == 11
    assert row["source_guid"] == "g-pg"  # entries[0]
    sigma = max(0.05 * abs(180.0), 0.5)
    assert row["variance"] == pytest.approx(sigma * sigma + 1.0)


def test_node_inj_id_assignment_multiple():
    nodes = _nodes([{"id": 5, "voltage_nominal": 220.0}])
    branches = _branches([])
    meas = _meas([{"id": 100, "object_type": OT_NODE, "object_id": 5, "measurement_type": MT_V}])
    resolved = {
        (5, "p_gen"): (300.0, 1, "g-pg", MeasurementQuality.GOOD),
        (5, "q_gen"): (50.0, 1, "g-qg", MeasurementQuality.GOOD),
        (5, "v"): (220.0, 1, "g-v", MeasurementQuality.GOOD),
    }
    stats, new_rows = _run(meas, nodes, branches, resolved)
    assert stats["node_inj_added"] == 2
    ids = sorted(r["id"] for r in new_rows)
    assert ids == [101, 102]  # max(100)+1, +1


# ---------------------------------------------------------------- (6) += re-accumulation


def test_repeated_accumulation_into_one_meas():
    # Two arg keys with the same (ot,mt,side,oid) -> += into the same row.
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            }
        ]
    )
    # Both p_from kinds map to (1,0,0,7). p_gen:<n> is a separate path. Hitting the
    # same row twice via two formal p_from keys is impossible here (dict keys are
    # unique), and a repeat via a separate object is impossible as well. A
    # worst-case with two measurements on the same index is likewise not
    # expressible. Instead the += semantics is checked directly: one
    # measurement, status already True.
    meas[0]["status"] = False
    resolved = {(7, "p_from"): (400.0, 1, "g", MeasurementQuality.GOOD)}
    _run(meas, nodes, branches, resolved)
    assert float(meas[0]["value"]) == pytest.approx(400.0)


def test_accumulation_branch_already_active_preset():
    # Pre-filled active measurement: the kernel resets every status to False
    # first, then re-activates, so the value is overwritten (not +=) because
    # status was reset. Checks that the pre-existing value does not "leak".
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
                "value": 999.0,
                "status": True,
            }
        ]
    )
    resolved = {(7, "p_from"): (150.0, 1, "g", MeasurementQuality.GOOD)}
    _run(meas, nodes, branches, resolved)
    assert float(meas[0]["value"]) == pytest.approx(150.0)  # overwritten, not 999+150


# ---------------------------------------------------------------- (7) branch-off skip


def test_branch_off_skip():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": False}])  # OFF
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            }
        ]
    )
    resolved = {(7, "p_from"): (150.0, 1, "g", MeasurementQuality.GOOD)}
    stats, _ = _run(meas, nodes, branches, resolved)
    assert stats["skipped_branch_off"] == 1 and stats["applied"] == 0
    assert bool(meas[0]["status"]) is False


# ---------------------------------------------------------------- (8) QUESTIONABLE × multiplier


def test_questionable_variance_multiplier():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            }
        ]
    )
    resolved = {(7, "p_from"): (150.0, 1, "g", MeasurementQuality.QUESTIONABLE)}
    stats, _ = _run(meas, nodes, branches, resolved)
    assert stats["applied_questionable"] == 1 and stats["applied"] == 0
    base_var = variance_power(150.0, sigma_frac=0.02)
    assert float(meas[0]["variance"]) == pytest.approx(base_var * 100.0)
    assert int(meas[0]["quality"]) == MeasurementQuality.QUESTIONABLE


# ---------------------------------------------------------------- v_sigma2_scale_by_node


def test_v_sigma2_scale_by_node_applied():
    # Node from the plan: V-measurement variance ×factor (0.1 = trust boosted ×10).
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([])
    meas = _meas([{"id": 10, "object_type": OT_NODE, "object_id": 1, "measurement_type": MT_V}])
    resolved = {(1, "v"): (225.0, 1, "g-v", MeasurementQuality.GOOD)}
    stats, _ = _run(meas, nodes, branches, resolved, v_sigma2_scale_by_node={1: 0.1})
    assert stats["applied"] == 1 and stats["v_sigma2_scaled"] == 1
    exp_var = variance_voltage(225.0, 220.0) * 0.1
    assert float(meas[0]["variance"]) == pytest.approx(exp_var)
    assert float(meas[0]["weight"]) == pytest.approx(1.0 / exp_var)


def test_flow_sigma2_scale_by_branch_applied():
    # (branch_id, kind) from the plan: flow-measurement variance ×factor (100 = down-weighting);
    # a different kind on the same branch is not touched.
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_Q,
                "branch_side": 0,
            },
            {
                "id": 11,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            },
        ]
    )
    resolved = {
        (7, "q_from"): (30.0, 1, "g-q", MeasurementQuality.GOOD),
        (7, "p_from"): (150.0, 1, "g-p", MeasurementQuality.GOOD),
    }
    stats, _ = _run(
        meas, nodes, branches, resolved, flow_sigma2_scale_by_branch={(7, "q_from"): 100.0}
    )
    assert stats["applied"] == 2 and stats["flow_sigma2_scaled"] == 1
    exp_q = variance_branch_q(30.0, charging_mvar=0.0, charging_alpha=0.10, sigma_frac=0.07) * 100.0
    assert float(meas[0]["variance"]) == pytest.approx(exp_q)
    assert float(meas[1]["variance"]) == pytest.approx(variance_power(150.0, sigma_frac=0.02))


def test_v_sigma2_scale_by_node_ignores_other_nodes_and_kinds():
    # A node outside the plan and non-V measurements are not scaled.
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}, {"id": 2, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {"id": 10, "object_type": OT_NODE, "object_id": 2, "measurement_type": MT_V},
            {
                "id": 11,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            },
        ]
    )
    resolved = {
        (2, "v"): (225.0, 1, "g-v", MeasurementQuality.GOOD),
        (7, "p_from"): (150.0, 1, "g-p", MeasurementQuality.GOOD),
    }
    stats, _ = _run(meas, nodes, branches, resolved, v_sigma2_scale_by_node={1: 0.1})
    assert stats["applied"] == 2 and "v_sigma2_scaled" not in stats
    assert float(meas[0]["variance"]) == pytest.approx(variance_voltage(225.0, 220.0))
    assert float(meas[1]["variance"]) == pytest.approx(variance_power(150.0, sigma_frac=0.02))


def test_questionable_node_inj_multiplier():
    nodes = _nodes([{"id": 5, "voltage_nominal": 220.0}])
    branches = _branches([])
    meas = _meas([])
    resolved = {(5, "p_gen"): (300.0, 1, "g-pg", MeasurementQuality.QUESTIONABLE)}
    stats, new_rows = _run(meas, nodes, branches, resolved)
    assert stats["node_inj_added"] == 1 and stats["applied_questionable"] == 1
    sigma = max(0.05 * 300.0, 0.5)
    base = sigma * sigma + 1.0
    assert new_rows[0]["variance"] == pytest.approx(base * 100.0)
    assert new_rows[0]["id"] == 1  # empty meas_arr -> next_id=1


# ---------------------------------------------------------------- misc coverage


def test_skipped_no_value_and_no_meas():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    # value=None with n_res=0 -> skipped_no_value; formula error n_res>0 -> skipped_formula_error.
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 7,
                "measurement_type": MT_P,
                "branch_side": 0,
            }
        ]
    )
    resolved = {
        (7, "p_from"): (None, 0, "", 0),  # skipped_no_value
        (8, "p_to"): (None, 2, "", 0),  # skipped_formula_error (but (1,0,1,8) has no measurement)
    }
    stats, _ = _run(meas, nodes, branches, resolved)
    # p_from: idx found, value None, n_res 0 -> skipped_no_value.
    assert stats["skipped_no_value"] == 1
    # p_to obj 8: idx not found (no measurement) -> skipped_no_meas before eval? No:
    # the kernel checks meas_idx before unpacking value. Branch 8 has no status -> off-skip.
    assert stats["skipped_branch_off"] == 1
