"""Tests of the measurement vector assembly (``gridstate.z_vector``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.units import BASE_MVA, model_to_pu
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    SIDE_FROM,
    SIDE_TO,
    build_z_and_r,
)


# ----------------------------------------------------------- helpers
def _build_two_bus_with_measurements():
    """Network of 2 nodes and 1 branch + a set of measurements of all types."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 0.0,
            "load_q": 0.0,
            "generation_p": 50.0,
            "generation_q": 20.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 30.0,
            "load_q": 10.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )

    # Link measurements <-> branches via the ti_* fields
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 1001,  # link to the measurement id=1001
            "ti_q_from": 1002,
            "ti_p_to": 1003,
            "ti_q_to": 1004,
        }
    )

    # Node measurements
    m.measurements.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": 110.5,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": 109.7,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 3,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 50.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 4,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_Q,
            "value": -10.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
        }
    )
    # Branch measurements with known side
    m.measurements.add(
        {
            "id": 1001,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_P,
            "value": 28.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 1003,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_P,
            "value": -28.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    return m


# --------------------------------------------------------------- basic
def test_basic_assembly_lengths_consistent() -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    z, R, idx = build_z_and_r(m, m.measurements, pu)
    assert len(z) == 6
    assert R.shape == (6, 6)
    assert len(idx) == 6


# ------------------------------------------------------- unit conversions
class TestUnitConversion:
    def test_voltage_kv_to_pu(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        v_pos = np.where(idx.kind == KIND_VOLTAGE)[0]
        # Nodes 1 and 2: vn=110 kV, value=110.5 -> 1.00454
        for i in v_pos:
            assert 0.99 < z[i] < 1.01

    def test_voltage_variance_scaled(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        _, R, idx = build_z_and_r(m, m.measurements, pu)
        v_pos = np.where(idx.kind == KIND_VOLTAGE)[0][0]
        # variance=0.01 kV², vn=110 -> σ²_pu = 0.01 / 110² ≈ 8.26e-7
        assert R.diagonal()[v_pos] == pytest.approx(0.01 / (110.0**2))

    def test_power_mw_to_pu(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        # POWER_INJECTION_P=50 MW -> 0.5 p.u.
        p_inj_pos = np.where(idx.kind == KIND_POWER_INJECTION_P)[0][0]
        assert z[p_inj_pos] == pytest.approx(50.0 / BASE_MVA)
        # Flow POWER_P (id=1001, value=28) -> 0.28 p.u.
        p_branch = np.where((idx.kind == KIND_POWER_P) & (idx.branch_side == SIDE_FROM))[0][0]
        assert z[p_branch] == pytest.approx(28.0 / BASE_MVA)


# -------------------------------------------------------- side detection
class TestBranchSide:
    def test_from_and_to_resolved_via_ti_fields(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        _, _, idx = build_z_and_r(m, m.measurements, pu)
        # id=1001 ↔ ti_p_from → SIDE_FROM
        # id=1003 ↔ ti_p_to → SIDE_TO
        for k, mid in zip(idx.kind, idx.meas_id, strict=False):
            if mid == 1001:
                assert k == KIND_POWER_P
            if mid == 1003:
                assert k == KIND_POWER_P
        side_for = {int(mid): int(s) for mid, s in zip(idx.meas_id, idx.branch_side, strict=False)}
        assert side_for[1001] == SIDE_FROM
        assert side_for[1003] == SIDE_TO

    def test_unresolvable_side_skipped(self, caplog) -> None:
        m = _build_two_bus_with_measurements()
        # Add a branch measurement not linked to any ti_*
        m.measurements.add(
            {
                "id": 9999,
                "object_type": OBJ_BRANCH,
                "object_id": 100,
                "measurement_type": KIND_POWER_P,
                "value": 0.0,
                "variance": 1.0,
                "status": True,
                "quality": 0,
            }
        )
        pu = model_to_pu(m)
        with caplog.at_level("WARNING"):
            _, _, idx = build_z_and_r(m, m.measurements, pu)
        # 6 valid + 0 invalid must come back
        assert 9999 not in idx.meas_id.tolist()


# ------------------------------------------------------- filtering
class TestFiltering:
    def test_inactive_status_skipped(self) -> None:
        m = _build_two_bus_with_measurements()
        # Mark measurement id=1 as inactive.
        m.measurements.get_by_id(1).status = False
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        assert 1 not in idx.meas_id.tolist()
        assert len(z) == 5

    def test_bad_quality_skipped(self) -> None:
        m = _build_two_bus_with_measurements()
        m.measurements.get_by_id(2).quality = 2  # BAD
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        assert 2 not in idx.meas_id.tolist()
        assert len(z) == 5

    def test_negative_variance_skipped(self) -> None:
        m = _build_two_bus_with_measurements()
        m.measurements.get_by_id(1).variance = -0.1
        pu = model_to_pu(m)
        _z, _, idx = build_z_and_r(m, m.measurements, pu)
        assert 1 not in idx.meas_id.tolist()


# ------------------------------------------------------- index integrity
def test_meas_index_uses_positional_indices() -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    _, _, idx = build_z_and_r(m, m.measurements, pu)
    # Node measurements must have object_pos in [0, n_bus)
    for k, op in zip(idx.object_kind, idx.object_pos, strict=False):
        if k == OBJ_NODE:
            assert 0 <= op < pu.n_bus
        elif k == OBJ_BRANCH:
            assert 0 <= op < pu.n_branch


# --------------------------------------------------------- empty input
def test_empty_collection_returns_zeros() -> None:
    """Network without measurements: z is empty, R is 0x0."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    pu = model_to_pu(m)
    z, R, idx = build_z_and_r(m, m.measurements, pu)
    assert len(z) == 0
    assert R.shape == (0, 0)
    assert len(idx) == 0


# --------------------------------------------------- unknown bus filtered
def test_measurement_on_unknown_node_skipped() -> None:
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    m.measurements.add(
        {
            "id": 5,
            "object_type": OBJ_NODE,
            "object_id": 999,  # there is no node 999
            "measurement_type": KIND_VOLTAGE,
            "value": 110.0,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    pu = model_to_pu(m)
    z, _, idx = build_z_and_r(m, m.measurements, pu)
    assert len(z) == 0
    assert 5 not in idx.meas_id.tolist()


# ------------------------------------------- current, generator, NaN variance
def test_current_side_from_q_reference_and_base_current() -> None:
    from gridstate.z_vector import KIND_CURRENT

    m = _build_two_bus_with_measurements()
    m.measurements.add(
        {
            "id": 1004,  # ti_q_to of branch 100
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_CURRENT,
            "value": 150.0,
            "variance": 4.0,
            "status": True,
            "quality": 0,
        }
    )
    pu = model_to_pu(m)
    z, R, idx = build_z_and_r(m, m.measurements, pu)
    k = idx.meas_id.tolist().index(1004)
    i_base = BASE_MVA * 1000.0 / (np.sqrt(3.0) * 110.0)
    assert idx.branch_side[k] == SIDE_TO
    assert z[k] == 150.0 / i_base
    assert R.diagonal()[k] == 4.0 / (i_base * i_base)


def test_generator_measurement_goes_to_its_node() -> None:
    m = _build_two_bus_with_measurements()
    m.generators.add({"id": 7, "node_id": 2, "status": True})
    m.measurements.add(
        {
            "id": 70,
            "object_type": 2,  # generator
            "object_id": 7,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 12.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
        }
    )
    pu = model_to_pu(m)
    z, _, idx = build_z_and_r(m, m.measurements, pu)
    k = idx.meas_id.tolist().index(70)
    assert idx.object_kind[k] == OBJ_NODE
    assert idx.object_pos[k] == int(np.where(pu.bus_ids == 2)[0][0])
    assert z[k] == 12.0 / BASE_MVA


def test_measurements_keep_collection_order() -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    _, _, idx = build_z_and_r(m, m.measurements, pu)
    assert idx.meas_id.tolist() == [1, 2, 3, 4, 1001, 1003]


# ------------------------------------------- non-finite value / variance
@pytest.mark.parametrize(
    ("field", "bad"),
    [("value", np.nan), ("value", np.inf), ("variance", np.nan), ("variance", np.inf)],
)
def test_nonfinite_measurement_skipped(field: str, bad: float) -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    z0, _, idx0 = build_z_and_r(m, m.measurements, pu)
    arr = m.measurements.to_numpy()
    bad_id = int(arr["id"][0])
    assert bad_id in idx0.meas_id.tolist()
    arr[field][0] = bad
    m.measurements.update_from_array(arr)

    z, R, idx = build_z_and_r(m, m.measurements, pu)

    assert len(z) == len(z0) - 1
    assert bad_id not in idx.meas_id.tolist()
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(R.diagonal()))


def test_nonfinite_measurement_warned_once_per_collection(caplog) -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    arr = m.measurements.to_numpy()
    arr["value"][0] = np.nan
    m.measurements.update_from_array(arr)

    with caplog.at_level("WARNING", logger="gridstate.z_vector"):
        build_z_and_r(m, m.measurements, pu)
        build_z_and_r(m, m.measurements, pu)
    assert sum("non-finite" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
@pytest.mark.parametrize(("field", "bad"), [("value", np.nan), ("variance", np.nan)])
def test_estimate_survives_nonfinite_measurement(algorithm: str, field: str, bad: float) -> None:
    """One NaN measurement must not fail the whole estimate (it is skipped)."""
    from pathlib import Path

    from gridstate.api import estimate
    from gridstate.contract.serialize import load_se_input_npz

    model = load_se_input_npz(Path(__file__).parent / "test_data" / "ieee" / "case14.npz").model
    arr = model.measurements.to_numpy()
    arr[field][5] = bad
    model.measurements.update_from_array(arr)

    res = estimate(model, algorithm=algorithm)

    assert res.success
    assert np.all(np.isfinite(res.v_pu))
    assert res.chi2 is not None and np.isfinite(res.chi2.value)
